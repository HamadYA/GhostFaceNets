import sys
import select
import os
import json
from tkinter.ttk import Separator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils


class Gently_stop_callback(keras.callbacks.Callback):
    def __init__(self, prompt="Continue? ([Y]/n)", time_out=3):
        super(Gently_stop_callback, self).__init__()
        self.yes_or_no = lambda: "n" not in self.timeout_input(prompt, time_out, default="y")[1].lower()

    def on_epoch_end(self, epoch, logs={}):
        print()
        if not self.yes_or_no():
            self.model.stop_training = True

    def timeout_input(self, prompt, timeout=3, default=""):
        print(prompt, end=": ", flush=True)
        inputs, outputs, errors = select.select([sys.stdin], [], [], timeout)
        print()
        return (0, sys.stdin.readline().strip()) if inputs else (-1, default)


class ExitOnNaN(keras.callbacks.Callback):
    """ Callback that exit directly when a NaN loss is encountered, avoiding saving model """

    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is not None:
            if not tf.math.is_finite(loss):
                print("\nError: Invalid loss, terminating training")
                self.model.stop_training = True
                sys.exit()


class My_history(keras.callbacks.Callback):
    def __init__(self, initial_file=None, evals=[]):
        super(My_history, self).__init__()
        if initial_file and os.path.exists(initial_file):
            with open(initial_file, "r") as ff:
                self.history = json.load(ff)
        else:
            self.history = {}
        self.evals = evals
        self.initial_file = initial_file
        self.custom_obj = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.pop("lr", None)
        lr = self.model.optimizer.lr
        if hasattr(lr, "value"):
            lr = lr.value()

        self.history.setdefault("lr", []).append(float(lr))
        for k, v in logs.items():
            k = "accuracy" if "accuracy" in k else k
            self.history.setdefault(k, []).append(float(v))
        for ee in self.evals:
            self.history.setdefault(ee.test_names, []).append(float(ee.cur_acc))
            self.history.setdefault(ee.test_names + "_thresh", []).append(float(ee.acc_thresh))
        for kk, vv in self.custom_obj.items():
            tt = losses_utils.compute_weighted_loss(vv())
            self.history.setdefault(kk, []).append(tt)
        if len(self.model.losses) != 0:
            regular_loss = K.sum(self.model.losses).numpy()
            self.history.setdefault("regular_loss", []).append(float(regular_loss))
            self.history["loss"][-1] -= regular_loss

        if self.initial_file:
            with open(self.initial_file, "w") as ff:
                json.dump(self.history, ff)

    def print_hist(self):
        print("{")
        for kk, vv in self.history.items():
            print("  '%s': %s," % (kk, vv))
        print("}")


class VPLUpdateQueue(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        batch_labels_back_up = self.model.loss[0].batch_labels_back_up
        update_label_pos = tf.expand_dims(batch_labels_back_up, 1)
        vpl_norm_dense_layer = self.model.layers[-1]

        updated_queue = tf.tensor_scatter_nd_update(vpl_norm_dense_layer.queue_features, update_label_pos, vpl_norm_dense_layer.norm_features)
        vpl_norm_dense_layer.queue_features.assign(updated_queue)

        iters = tf.repeat(vpl_norm_dense_layer.iters, tf.shape(batch_labels_back_up)[0])
        updated_queue_iters = tf.tensor_scatter_nd_update(vpl_norm_dense_layer.queue_iters, update_label_pos, iters)
        vpl_norm_dense_layer.queue_iters.assign(updated_queue_iters)


class OptimizerWeightDecay(keras.callbacks.Callback):
    def __init__(self, lr_base, wd_base, is_lr_on_batch=False):
        super(OptimizerWeightDecay, self).__init__()
        self.wd_m = wd_base / lr_base
        self.lr_base, self.wd_base = lr_base, wd_base
        # self.model.optimizer.weight_decay = lambda: wd_m * self.model.optimizer.lr
        self.is_lr_on_batch = is_lr_on_batch
        if is_lr_on_batch:
            self.on_train_batch_begin = self.__update_wd__
        else:
            self.on_epoch_begin = self.__update_wd__

    def __update_wd__(self, step, log=None):
        if self.model is not None:
            wd = self.wd_m * K.get_value(self.model.optimizer.lr)
            # wd = self.wd_base * K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.weight_decay, wd)
        # wd = self.model.optimizer.weight_decay
        if not self.is_lr_on_batch or step == 0:
            print("Weight decay is {}".format(wd))


class CosineLrSchedulerEpoch(keras.callbacks.Callback):
    def __init__(self, lr_base, first_restart_step, m_mul=0.5, t_mul=2.0, lr_min=1e-6, lr_warmup=-1, warmup_steps=0, cooldown_steps=1):
        super(CosineLrSchedulerEpoch, self).__init__()
        self.warmup_steps, self.cooldown_steps, self.lr_min = warmup_steps, cooldown_steps, lr_min

        if lr_min == lr_base * m_mul:
            self.schedule = keras.experimental.CosineDecay(lr_base, first_restart_step, alpha=lr_min / lr_base)
            self.cooldown_steps_start, self.cooldown_steps_end = np.array([]), np.array([])
        else:
            self.schedule = keras.experimental.CosineDecayRestarts(lr_base, first_restart_step, t_mul=t_mul, m_mul=m_mul, alpha=lr_min / lr_base)
            aa = [first_restart_step * (t_mul ** ii) for ii in range(5)]
            self.cooldown_steps_start = np.array([warmup_steps + int(sum(aa[:ii]) + cooldown_steps * (ii - 1)) for ii in range(1, 5)])
            self.cooldown_steps_end = np.array([ii + cooldown_steps for ii in self.cooldown_steps_start])

        if warmup_steps != 0:
            self.lr_warmup = lr_warmup if lr_warmup > 0 else lr_min
            self.warmup_lr_func = lambda ii: self.lr_warmup + (lr_base - self.lr_warmup) * ii / warmup_steps

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_steps:
            lr = self.warmup_lr_func(epoch)
        elif self.cooldown_steps_end.shape[0] != 0:
            cooldown_end_pos = (self.cooldown_steps_end > epoch).argmax()
            if epoch >= self.cooldown_steps_end[cooldown_end_pos] - self.cooldown_steps:
                lr = self.lr_min  # cooldown
            else:
                lr = self.schedule(epoch - self.cooldown_steps * cooldown_end_pos - self.warmup_steps)
                # lr = self.schedule(epoch - self.cooldown_steps * cooldown_end_pos)
        else:
            lr = self.schedule(epoch)

        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)

        print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
        return lr


class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, first_restart_step, steps_per_epoch=-1, m_mul=0.5, t_mul=2.0, lr_min=1e-5, lr_warmup=-1, warmup_steps=0, cooldown_steps=1):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.m_mul, self.t_mul, self.lr_min, self.steps_per_epoch = lr_base, m_mul, t_mul, lr_min, steps_per_epoch
        self.first_restart_step, self.warmup_steps, self.cooldown_steps, self.lr_warmup = first_restart_step, warmup_steps, cooldown_steps, lr_warmup
        self.init_step_num, self.cur_epoch, self.is_cooldown_epoch, self.previous_cooldown_steps = 0, 0, False, 0
        self.is_built = False
        if steps_per_epoch != -1:
            self.build(steps_per_epoch)

    def build(self, steps_per_epoch=-1):
        if steps_per_epoch != -1:
            self.steps_per_epoch = steps_per_epoch

        first_restart_batch_step = self.first_restart_step * self.steps_per_epoch
        alpha = self.lr_min / self.lr_base
        if self.lr_min == self.lr_base * self.m_mul:  # Without restart
            self.schedule = keras.experimental.CosineDecay(self.lr_base, first_restart_batch_step, alpha=alpha)
            self.cooldown_steps_start, self.cooldown_steps_end = np.array([]), np.array([])
        else:
            self.schedule = keras.experimental.CosineDecayRestarts(self.lr_base, first_restart_batch_step, t_mul=self.t_mul, m_mul=self.m_mul, alpha=alpha)
            aa = [first_restart_batch_step / self.steps_per_epoch * (self.t_mul ** ii) for ii in range(5)]
            self.cooldown_steps_start = np.array([self.warmup_steps + int(sum(aa[:ii]) + self.cooldown_steps * (ii - 1)) for ii in range(1, 5)])
            self.cooldown_steps_end = np.array([ii + self.cooldown_steps for ii in self.cooldown_steps_start])

        if self.warmup_steps != 0:
            self.warmup_batch_steps = self.warmup_steps * self.steps_per_epoch
            self.lr_warmup = self.lr_warmup if self.lr_warmup > 0 else self.lr_min
            self.warmup_lr_func = lambda ii: self.lr_warmup + (self.lr_base - self.lr_warmup) * ii / self.warmup_batch_steps
        else:
            self.warmup_batch_steps = 0
        self.is_built = True

    def on_epoch_begin(self, cur_epoch, logs=None):
        if not self.is_built:
            self.build()

        self.init_step_num = int(self.steps_per_epoch * cur_epoch)
        self.cur_epoch = cur_epoch

        if self.cooldown_steps_end.shape[0] != 0:
            cooldown_end_pos = (self.cooldown_steps_end > cur_epoch).argmax()
            self.previous_cooldown_steps = self.cooldown_steps * cooldown_end_pos * self.steps_per_epoch
            if cur_epoch >= self.cooldown_steps_end[cooldown_end_pos] - self.cooldown_steps:
                self.is_cooldown_epoch = True
            else:
                self.is_cooldown_epoch = False

    def on_train_batch_begin(self, iterNum, logs=None):
        global_iterNum = iterNum + self.init_step_num
        if global_iterNum < self.warmup_batch_steps:
            lr = self.warmup_lr_func(global_iterNum)
        elif self.is_cooldown_epoch:
            lr = self.lr_min  # cooldown
        else:
            lr = self.schedule(global_iterNum - self.warmup_batch_steps - self.previous_cooldown_steps)
            # lr = self.schedule(global_iterNum - self.previous_cooldown_steps)

        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        if iterNum == 0:
            print("\nLearning rate for iter {} is {}, global_iterNum is {}".format(self.cur_epoch + 1, lr, global_iterNum))
        return lr


def exp_scheduler(epoch, lr_base, decay_rate=0.05, lr_min=0, warmup_steps=0):
    if epoch < warmup_steps:
        lr = lr_min + (lr_base - lr_min) * epoch / warmup_steps
    else:
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decaysteps)
        lr = lr_base * np.exp(decay_rate * (warmup_steps - epoch))
    lr = lr_min if lr < lr_min else lr
    print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def constant_scheduler(epoch, lr_base, lr_decay_steps, decay_rate=0.1, warmup_steps=0):
    if epoch < warmup_steps:
        lr_min = lr_base * decay_rate ** len(lr_decay_steps)
        lr = lr_min + (lr_base - lr_min) * epoch / warmup_steps
    else:
        lr = lr_base * decay_rate ** np.sum(epoch >= np.array(lr_decay_steps))
    print("\nLearning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def basic_callbacks(checkpoint="keras_checkpoints.h5", evals=[], lr=0.001, lr_decay=0.05, lr_min=0, lr_decay_steps=0, lr_warmup_steps=0):
    checkpoint_base = "checkpoints"
    if not os.path.exists(checkpoint_base):
        os.mkdir(checkpoint_base)
    checkpoint = os.path.join(checkpoint_base, checkpoint)
    model_checkpoint = ModelCheckpoint(checkpoint, verbose=1, save_weights_only=True)
    # model_checkpoint = ModelCheckpoint(checkpoint, verbose=1)
    model_checkpoint = ModelCheckpoint(os.path.splitext(checkpoint)[0] + "_epoch{epoch:02d}.h5", verbose=1)
    # model_checkpoint = keras.callbacks.experimental.BackupAndRestore(checkpoint_base)

    if isinstance(lr_decay_steps, list):
        # Constant decay on epoch
        lr_scheduler = LearningRateScheduler(lambda epoch: constant_scheduler(epoch, lr, lr_decay_steps, lr_decay, lr_warmup_steps))
    elif lr_decay_steps > 1:
        # Cosine decay on epoch / batch
        lr_scheduler = CosineLrScheduler(lr, first_restart_step=lr_decay_steps, m_mul=lr_decay, lr_min=lr_min, lr_warmup=lr_min, warmup_steps=lr_warmup_steps)
    else:
        # Exponential decay
        lr_scheduler = LearningRateScheduler(lambda epoch: exp_scheduler(epoch, lr, lr_decay, lr_min, warmup_steps=lr_warmup_steps))
    my_history = My_history(os.path.splitext(checkpoint)[0] + "_hist.json", evals=evals)
    history_logger = keras.callbacks.CSVLogger('log.csv', separator = ',', append = True)
    # tensor_board_log = keras.callbacks.TensorBoard(log_dir=os.path.splitext(checkpoint)[0] + '_logs')
    return [my_history, model_checkpoint, lr_scheduler, Gently_stop_callback()]

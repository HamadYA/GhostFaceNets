import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras


# ArcfaceLoss class
class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        # reduction = tf.keras.losses.Reduction.NONE if tf.distribute.has_strategy() else tf.keras.losses.Reduction.AUTO
        # super(ArcfaceLoss, self).__init__(**kwargs, reduction=reduction)
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2
        self.batch_labels_back_up = None
        # self.reduction_func = tf.keras.losses.CategoricalCrossentropy(
        #     from_logits=from_logits, label_smoothing=label_smoothing, reduction=reduction
        # )
        # self.norm_logits = tf.Variable(tf.zeros([512, 93431]), dtype="float32", trainable=False)
        # self.y_true = tf.Variable(tf.zeros([512, 93431], dtype="int32"), dtype="int32", trainable=False)

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # norm_logits = y_pred
        # self.norm_logits.assign(norm_logits)
        # self.y_true.assign(y_true)
        # pick_cond = tf.cast(y_true, dtype=tf.bool)
        # y_pred_vals = norm_logits[pick_cond]
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        # tf.print(", y_true.sum:", tf.reduce_sum(y_true), ", y_pred_vals.shape:", K.shape(y_pred_vals), ", y_true.shape:", K.shape(y_true), end="")
        # tf.assert_equal(tf.reduce_sum(y_true), K.shape(y_true)[0])
        # tf.assert_equal(K.shape(y_pred_vals)[0], K.shape(y_true)[0])
        # y_pred_vals = tf.clip_by_value(y_pred_vals, -1, 1)
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = y_pred_vals
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3
            # Grad(theta) == 0
            #   ==> np.sin(np.math.acos(xx) * margin1 + margin2) == 0
            #   ==> np.math.acos(xx) * margin1 + margin2 == np.pi
            #   ==> xx == np.cos((np.pi - margin2) / margin1)
            #   ==> theta_min == theta(xx) == -1 - margin3
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        # theta_one_hot = tf.expand_dims(theta_valid - y_pred_vals, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = (theta_one_hot + norm_logits) * self.scale
        # theta_one_hot = tf.expand_dims(theta_valid, 1) * tf.cast(y_true, dtype=tf.float32)
        # arcface_logits = tf.where(pick_cond, theta_one_hot, norm_logits) * self.scale
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        # tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
        # arcface_logits = tf.cond(tf.math.is_finite(tf.reduce_mean(arcface_logits)), lambda: arcface_logits, lambda: tf.cast(y_true, "float32"))
        # arcface_logits = tf.where(tf.math.is_finite(arcface_logits), arcface_logits, tf.zeros_like(arcface_logits))
        # cond = tf.repeat(tf.math.is_finite(tf.reduce_sum(arcface_logits, axis=-1, keepdims=True)), arcface_logits.shape[-1], axis=-1)
        # arcface_logits = tf.where(cond, arcface_logits, tf.zeros_like(arcface_logits))
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)
        # return self.reduction_func(y_true, arcface_logits)

    def get_config(self):
        config = super(ArcfaceLoss, self).get_config()
        config.update(
            {
                "margin1": self.margin1,
                "margin2": self.margin2,
                "margin3": self.margin3,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ArcfaceLoss simple
class ArcfaceLossSimple(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossSimple, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        # self.low_pred_punish = tf.sin(np.pi - margin) * margin
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(y_true != 0)
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLossSimple, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



# [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/pdf/1801.09414.pdf)
class CosFaceLoss(ArcfaceLossSimple):
    def __init__(self, margin=0.35, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(CosFaceLoss, self).__init__(margin, scale, from_logits, label_smoothing, **kwargs)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.cast(y_true, dtype=tf.bool)
        logits = tf.where(pick_cond, norm_logits - self.margin, norm_logits) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

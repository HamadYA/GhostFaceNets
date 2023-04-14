#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import normalize
from data import pre_process_folder, tf_imread

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        import mxnet as mx

        self.mx = mx
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [self.mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [self.mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = self.mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = self.mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = self.mx.nd.array(imgs)
        db = self.mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


class Torch_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        self.model = self.torch.jit.load(model_file, map_location=device_name)

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
        return output.cpu().detach().numpy()


def teacher_model_interf_wrapper(model_file):
    if model_file.endswith(".h5"):
        # Keras model file
        mm = tf.keras.models.load_model(model_file, compile=False)
        mm.trainable = False
        interf_func = lambda imm: mm((imm - 127.5) * 0.0078125)
        return interf_func

    if model_file.endswith(".pth") or model_file.endswith(".pt"):
        # Try pytorch
        mm = Torch_model_interf(model_file)
        emb_shape = mm(np.ones([1, 112, 112, 3])).shape[-1]
    else:
        # MXNet model file, like models/r50-arcface-emore/model,1
        mm = Mxnet_model_interf(model_file)
        emb_shape = mm.model.output_shapes[0][-1][-1]

    def interf_func(imm):
        emb = tf.numpy_function(mm, [imm], tf.float32)
        emb.set_shape([None, emb_shape])
        return emb

    return interf_func


class Data_distiller:
    def __init__(self, data_path, model_file=None, dest_file=None, save_npz=False, batch_size=256, use_fp16=False, limit=-1):
        self.data_path, self.model_file, self.batch_size = data_path, model_file, batch_size
        self.dest_file, self.save_npz, self.use_fp16, self.limit = dest_file, save_npz, use_fp16, limit
        if model_file == None and data_path.endswith(".npz"):
            image_names, image_classes, embeddings = self.__init_from_saved_npz__()
            self.emb_gen = ([image_names, image_classes, embeddings],)
            self.save_npz = False
            self.tqdm_desc = "Converting"
        elif self.model_file != None:
            self.__init_ds_model_dest__()
            self.emb_gen = self.__extract_emb_gen__()
            self.tqdm_desc = "Embedding"
        else:
            return

        if save_npz:
            self.__save_to_npz__()
        else:
            self.__save_to_tfrecord_by_batch__()
        print(">>>> Output:", self.dest_file)

    def __init_ds_model_dest__(self):
        """ Init dataset """
        image_names, image_classes, _, classes, dataset_pickle_file_src = pre_process_folder(self.data_path)
        print(">>>> Image length: %d, Image class length: %d, classes: %d" % (len(image_names), len(image_classes), classes))
        if self.limit > 0:
            image_names, image_classes = image_names[: self.limit], image_classes[: self.limit]
        total = len(image_names)
        ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
        ds = ds.batch(self.batch_size)

        """ Init model, it could be PyTorch model file / MXNet model file / keras model file """
        interf_func = teacher_model_interf_wrapper(self.model_file)
        emb_shape = interf_func(np.ones([1, 112, 112, 3])).shape[-1]

        """ Init dest filename """
        if self.dest_file is None:
            src_name = os.path.splitext(dataset_pickle_file_src)[0]
            self.dest_file = src_name + "_label_embs_{}".format(emb_shape)
            if self.use_fp16:
                self.dest_file += "_fp16"
        ext_format = ".npz" if self.save_npz else ".tfrecord"
        self.dest_file = self.dest_file if self.dest_file.endswith(ext_format) else self.dest_file + ext_format
        self.interf_func, self.ds, self.classes, self.emb_shape, self.total = interf_func, ds, classes, emb_shape, total

    def __init_from_saved_npz__(self):
        print(">>>> Reload data from:", self.data_path)
        aa = np.load(self.data_path)
        image_names, image_classes, embeddings = aa["image_names"], aa["image_classes"], aa["embeddings"]

        classes = np.max(image_classes) + 1
        total = len(image_names)
        emb_shape = embeddings.shape[-1]
        self.use_fp16 = True if embeddings.dtype == np.float16 else self.use_fp16
        embeddings = embeddings.astype("float16") if self.use_fp16 else embeddings.astype("float32")
        print(">>>> [Base info] total:", total, "classes:", classes, "emb_shape:", emb_shape, "use_fp16:", self.use_fp16)

        if self.dest_file is None:
            self.dest_file = os.path.splitext(self.data_path)[0]
            if self.use_fp16 and "_fp16" not in self.data_path:
                self.dest_file += "_fp16"
        self.dest_file = self.dest_file if self.dest_file.endswith(".tfrecord") else self.dest_file + ".tfrecord"
        self.classes, self.total, self.emb_shape = classes, total, emb_shape
        return image_names, image_classes, embeddings

    def __extract_emb_gen__(self):
        for imm, label in self.ds:
            imgs = tf.stack([tf_imread(ii) for ii in imm])
            emb = self.interf_func(imgs)
            emb = np.array(emb, dtype="float16") if self.use_fp16 else np.array(emb, dtype="float32")
            yield imm.numpy(), label.numpy(), emb

    def __save_to_npz__(self):
        """ Extract embeddings """
        steps = int(np.ceil(self.total // self.batch_size)) + 1
        image_names, image_classes, embeddings = [], [], []
        for imm, label, emb in tqdm(self.emb_gen, self.tqdm_desc, total=steps):
            image_names.extend(imm)
            image_classes.extend(label)
            embeddings.extend(emb)
        # imms, labels, embeddings = np.array(imms), np.array(labels), np.array(embeddings)
        np.savez_compressed(self.dest_file, image_names=image_names, image_classes=image_classes, embeddings=embeddings)

    def __save_to_tfrecord_by_batch__(self):
        """ Encode feature definations, save also `classes` and `emb_shape` """
        self.encode_base_info = {
            "classes": tf.train.Feature(int64_list=tf.train.Int64List(value=[self.classes])),
            "emb_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=[self.emb_shape])),
            "total": tf.train.Feature(int64_list=tf.train.Int64List(value=[self.total])),
            "use_fp16": tf.train.Feature(int64_list=tf.train.Int64List(value=[self.use_fp16])),
        }
        self.encode_feature = {
            "image_names": lambda vv: tf.train.Feature(bytes_list=tf.train.BytesList(value=[vv])),
            "image_classes": lambda vv: tf.train.Feature(int64_list=tf.train.Int64List(value=[vv])),
            # "embeddings": lambda vv: tf.train.Feature(float_list=tf.train.FloatList(value=vv.tolist())),
            "embeddings": lambda vv: tf.train.Feature(bytes_list=tf.train.BytesList(value=[vv.tobytes()])),
        }

        is_first_line = True
        with tf.io.TFRecordWriter(self.dest_file) as file_writer, tqdm(desc=self.tqdm_desc, total=self.total) as pbar:
            for imm, label, emb in self.emb_gen:
                data = {"image_names": imm, "image_classes": label, "embeddings": emb}
                batch_steps = range(len(data["image_names"]))
                for ii in batch_steps:
                    feature = {kk: self.encode_feature[kk](data[kk][ii]) for kk in data}
                    if is_first_line:  # Save base_info in the first line
                        is_first_line = False
                        feature.update(self.encode_base_info)
                    record_bytes = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
                    file_writer.write(record_bytes)
                    pbar.update(1)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-D", "--data_path", type=str, required=True, help="Data path, or npz file converting to tfrecord")
    parser.add_argument("-M", "--model_file", type=str, default=None, help="Model file, keras h5 / pytorch pth / mxnet")
    parser.add_argument("-d", "--dest_file", type=str, default=None, help="Dest file path to save the processed dataset")
    parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("-L", "--limit", type=int, default=-1, help="Test parameter, limit converting only the first [NUM]")
    parser.add_argument("--use_fp16", action="store_true", help="Save using float16")
    parser.add_argument("--save_npz", action="store_true", help="Save as npz file, default is tfrecord")
    args = parser.parse_known_args(sys.argv[1:])[0]

    Data_distiller(args.data_path, args.model_file, args.dest_file, args.save_npz, args.batch_size, args.use_fp16, args.limit)

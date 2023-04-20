#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from glob2 import glob
from skimage import transform
from skimage.io import imread, imsave

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

FILE_HASH = {"yolov5s_face_dynamic": "e7854a5cae48ded05b3b31aa93765f0d"}

DEFAULT_DETECTOR = "https://github.com/leondgarse/Keras_insightface/releases/download/v1.0.0/yolov5s_face_dynamic.h5"
DEFAULT_ANCHORS = np.array(
    [
        [[0.5, 0.625], [1.0, 1.25], [1.625, 2.0]],
        [[1.4375, 1.8125], [2.6875, 3.4375], [4.5625, 6.5625]],
        [[4.5625, 6.781199932098389], [7.218800067901611, 9.375], [10.468999862670898, 13.531000137329102]],
    ],
    dtype="float32",
)
DEFAULT_STRIDES = np.array([8, 16, 32], dtype="float32")


class BaseDetector:
    def face_align_landmarks(self, img, landmarks, image_size=(112, 112), method="similar"):
        tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
        src = np.array(
            [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
            dtype=np.float32,
        )
        ret = []
        landmarks = landmarks if landmarks.shape[1] == 5 else tf.reshape(landmarks, [-1, 5, 2]).numpy()
        for landmark in landmarks:
            # landmark = np.array(landmark).reshape(2, 5)[::-1].T
            tform.estimate(landmark, src)
            # M = tform.params[0:2, :]
            # ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
            ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
            if len(ndimage.shape) == 2:
                ndimage = np.stack([ndimage, ndimage, ndimage], -1)
            ret.append(ndimage)
        # return np.array(ret)
        return (np.array(ret) * 255).astype(np.uint8)

    def detect_in_image(self, image, max_output_size=15, iou_threshold=0.45, score_threshold=0.25, image_format="RGB"):
        if isinstance(image, str):
            image = imread(image)[:, :, :3]
            image_format = "RGB"
        bbs, pps, ccs = self.__call__(image, max_output_size, iou_threshold, score_threshold, image_format)
        # print(bbs.shape, pps.shape, ccs.shape)
        if len(bbs) != 0:
            image_RGB = image if image_format == "RGB" else image[:, :, ::-1]
            return bbs, pps, ccs, self.face_align_landmarks(image_RGB, pps)
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])

    def detect_in_folder(self, data_path, max_output_size=15, iou_threshold=0.45, score_threshold=0.25):
        while data_path.endswith(os.sep):
            data_path = data_path[:-1]
        imms = glob(os.path.join(data_path, "*", "*"))
        use_class = True
        if len(imms) == 0:
            imms = glob(os.path.join(data_path, "*"))
            use_class = False
        dest_path = data_path + "_aligned_112_112"

        for imm in tqdm(imms, "Detecting"):
            _, _, _, nimages = self.detect_in_image(imm, max_output_size, iou_threshold, score_threshold, image_format="RGB")
            if nimages.shape[0] != 0:
                file_name = os.path.basename(imm)
                if use_class:
                    class_name = os.path.basename(os.path.dirname(imm))
                    save_dir = os.path.join(dest_path, class_name)
                else:
                    save_dir = dest_path
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                imsave(os.path.join(save_dir, file_name), nimages[0])  # Use only the first one
            else:
                print(">>>> None face detected in image:", imm)
        print(">>>> Saved aligned face images in:", dest_path)
        return dest_path

    def show_result(self, image, bbs, pps=[], ccs=[]):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(image)
        for id, bb in enumerate(bbs):
            plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]], [bb[1], bb[1], bb[3], bb[3], bb[1]])
            if len(ccs) != 0:
                plt.text(bb[0], bb[1], "{:.4f}".format(ccs[id]))
            if len(pps) != 0:
                pp = pps[id]
                if len(pp.shape) == 2:
                    plt.scatter(pp[:, 0], pp[:, 1], s=8)
                else:
                    plt.scatter(pp[::2], pp[1::2], s=8)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


class YoloV5FaceDetector(BaseDetector):
    """ Yolov5-face Ported from https://github.com/deepcam-cn/yolov5-face """

    def __init__(self, model_path=DEFAULT_DETECTOR, anchors=DEFAULT_ANCHORS, strides=DEFAULT_STRIDES):
        if isinstance(model_path, str) and model_path.startswith("http"):
            file_name = os.path.basename(model_path)
            file_hash = FILE_HASH.get(os.path.splitext(file_name)[0], None)
            model_path = tf.keras.utils.get_file(file_name, model_path, cache_subdir="models", file_hash=file_hash)
            self.model = tf.keras.models.load_model(model_path)
        elif isinstance(model_path, str) and model_path.endswith(".h5"):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = model_path

        self.anchors, self.strides = anchors, strides
        self.num_anchors = anchors.shape[1]
        self.anchor_grids = tf.math.ceil((anchors * strides[:, tf.newaxis, tf.newaxis])[:, tf.newaxis, :, tf.newaxis, :])

    def make_grid(self, nx=20, ny=20, dtype=tf.float32):
        xv, yv = tf.meshgrid(tf.range(nx), tf.range(ny))
        return tf.cast(tf.reshape(tf.stack([xv, yv], 2), [1, 1, -1, 2]), dtype=dtype)

    def pre_process_32(self, image):
        hh, ww, _ = image.shape
        pad_hh = (32 - hh % 32) % 32  # int(tf.math.ceil(hh / 32) * 32) - hh
        pad_ww = (32 - ww % 32) % 32  # int(tf.math.ceil(ww / 32) * 32) - ww
        if pad_ww != 0 or pad_hh != 0:
            image = tf.pad(image, [[0, pad_hh], [0, pad_ww], [0, 0]])
        return tf.expand_dims(image, 0)

    def post_process(self, outputs, image_height, image_width):
        post_outputs = []
        for output, stride, anchor, anchor_grid in zip(outputs, self.strides, self.anchors, self.anchor_grids):
            hh, ww = image_height // stride, image_width // stride
            anchor_width = output.shape[-1] // self.num_anchors
            output = tf.reshape(output, [-1, output.shape[1] * output.shape[2], self.num_anchors, anchor_width])
            output = tf.transpose(output, [0, 2, 1, 3])

            cls = tf.sigmoid(output[:, :, :, :5])
            cur_grid = self.make_grid(ww, hh, dtype=output.dtype) * stride
            xy = cls[:, :, :, 0:2] * (2 * stride) - 0.5 * stride + cur_grid
            wh = (cls[:, :, :, 2:4] * 2) ** 2 * anchor_grid

            mm = [1, 1, 1, 5]
            landmarks = output[:, :, :, 5:15] * tf.tile(anchor_grid, mm) + tf.tile(cur_grid, mm)

            # print(output.shape, cls.shape, xy.shape, wh.shape, landmarks.shape)
            post_out = tf.concat([xy, wh, landmarks, cls[:, :, :, 4:]], axis=-1)
            post_outputs.append(tf.reshape(post_out, [-1, output.shape[1] * output.shape[2], anchor_width - 1]))
        return tf.concat(post_outputs, axis=1)

    def yolo_nms(self, inputs, max_output_size=15, iou_threshold=0.35, score_threshold=0.25):
        inputs = inputs[0][inputs[0, :, -1] > score_threshold]
        xy_center, wh, ppt, cct = inputs[:, :2], inputs[:, 2:4], inputs[:, 4:14], inputs[:, 14]
        xy_start = xy_center - wh / 2
        xy_end = xy_start + wh
        bbt = tf.concat([xy_start, xy_end], axis=-1)
        rr = tf.image.non_max_suppression(bbt, cct, max_output_size=max_output_size, iou_threshold=iou_threshold, score_threshold=0.0)
        bbs, pps, ccs = tf.gather(bbt, rr, axis=0), tf.gather(ppt, rr, axis=0), tf.gather(cct, rr, axis=0)
        pps = tf.reshape(pps, [-1, 5, 2])
        return bbs.numpy(), pps.numpy(), ccs.numpy()

    def __call__(self, image, max_output_size=15, iou_threshold=0.45, score_threshold=0.25, image_format="RGB"):
        imm_RGB = image if image_format == "RGB" else image[:, :, ::-1]
        imm_RGB = self.pre_process_32(imm_RGB)
        outputs = self.model(imm_RGB)
        post_outputs = self.post_process(outputs, imm_RGB.shape[1], imm_RGB.shape[2])
        return self.yolo_nms(post_outputs, max_output_size, iou_threshold, score_threshold)


class SCRFD(BaseDetector):
    """ SCRFD from https://github.com/deepinsight/insightface """

    def __init__(self, det_shape=640):
        self.model = self.download_and_prepare_det()
        self.det_shape = (det_shape, det_shape)

    def __call__(self, image, max_output_size=15, iou_threshold=0.45, score_threshold=0.25, image_format="RGB"):
        imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
        bboxes, pps = self.model.detect(imm_BGR, self.det_shape)
        bbs, ccs = bboxes[:, :4], bboxes[:, -1]
        return bbs, pps, ccs

    def download_and_prepare_det(self):
        import insightface

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        ctx = 0 if len(cvd) > 0 and int(cvd) != -1 else -1

        model_file = os.path.expanduser("~/.insightface/models/antelope/scrfd_10g_bnkps.onnx")
        if not os.path.exists(model_file):
            import zipfile

            model_url = "http://storage.insightface.ai/files/models/antelope.zip"
            zip_file = os.path.expanduser("~/.insightface/models/antelope.zip")
            zip_extract_path = os.path.splitext(zip_file)[0]
            if not os.path.exists(os.path.dirname(zip_file)):
                os.makedirs(os.path.dirname(zip_file))
            insightface.utils.storage.download_file(model_url, path=zip_file, overwrite=True)
            with zipfile.ZipFile(zip_file) as zf:
                zf.extractall(zip_extract_path)
            os.remove(zip_file)

        model = insightface.model_zoo.SCRFD(model_file=model_file)
        model.prepare(ctx)
        return model


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "input_path",
        type=str,
        default=None,
        help="Could be: 1. Data path, containing images in class folders; 2. image folder path, containing multiple images; 3. jpg / png image path",
    )
    parser.add_argument("--use_scrfd", action="store_true", help="Use SCRFD instead of YoloV5FaceDetector")
    args = parser.parse_known_args(sys.argv[1:])[0]

    det = SCRFD() if args.use_scrfd else YoloV5FaceDetector()
    if args.input_path.endswith(".jpg") or args.input_path.endswith(".png"):
        print(">>>> Detection in image:", args.input_path)
        imm = imread(args.input_path)
        bbs, pps, ccs, nimgs = det.detect_in_image(imm)
        det.show_result(imm, bbs, pps, ccs)
    else:
        print(">>>> Detection in folder:", args.input_path)
        det.detect_in_folder(args.input_path)

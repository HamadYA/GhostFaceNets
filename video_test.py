#!/usr/bin/env python3

import os
import cv2
import glob2

# import insightface
import numpy as np
import pandas as pd
import tensorflow as tf

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import normalize
from skimage.io import imread
from skimage import transform
from tqdm import tqdm
from face_detector import YoloV5FaceDetector


def init_det_and_emb_model(model_file):
    # det = insightface.model_zoo.face_detection.retinaface_mnet025_v1()
    # det = insightface.model_zoo.SCRFD(model_file=os.path.expanduser("~/.insightface/models/antelope/scrfd_10g_bnkps.onnx"))
    # det.prepare(-1)
    det = YoloV5FaceDetector()
    if model_file is not None:
        face_model = tf.keras.models.load_model(model_file, compile=False)
    else:
        face_model = None
    return det, face_model


def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
    ret = []
    for landmark in landmarks:
        # landmark = np.array(landmark).reshape(2, 5)[::-1].T
        tform.estimate(landmark, src)
        ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
    return (np.array(ret) * 255).astype(np.uint8)


def do_detect_in_image(image, det, image_format="BGR"):
    imm_BGR = image if image_format == "BGR" else image[:, :, ::-1]
    imm_RGB = image[:, :, ::-1] if image_format == "BGR" else image
    # bboxes, pps = det.detect(imm_BGR, (640, 640))

    bboxes, pps, ccs = det.__call__(imm_BGR)
    #OR
    # bboxes, pps, ccs, _ = det.detect_in_image(imm_BGR)

    nimgs = face_align_landmarks_sk(imm_RGB, pps)
    bbs, ccs = bboxes[:, :4].astype("int"), bboxes[:, -1]
    return bbs, ccs, nimgs


def embedding_images(det, face_model, known_user, batch_size=32, force_reload=False):
    while known_user.endswith("/"):
        known_user = known_user[:-1]
    dest_pickle = os.path.join(known_user, os.path.basename(known_user) + "_embedding.npz")

    if force_reload == False and os.path.exists(dest_pickle):
        aa = np.load(dest_pickle)
        image_classes, embeddings = aa["image_classes"], aa["embeddings"]
    else:
        if not os.path.exists(known_user):
            return [], [], None
        # data_gen = ImageDataGenerator(preprocessing_function=lambda img: (img - 127.5) * 0.0078125)
        # img_gen = data_gen.flow_from_directory(known_user, target_size=(112, 112), batch_size=1, class_mode='binary')
        image_names = glob2.glob(os.path.join(known_user, "*/*.jpg"))

        """ Detct faces in images, keep only those have exactly one face. """
        nimgs, image_classes = [], []
        for image_name in tqdm(image_names, "Detect"):
            img = imread(image_name)
            nimg = do_detect_in_image(img, det, image_format="RGB")[-1]
            if nimg.shape[0] > 0:
                nimgs.append(nimg[0])
                image_classes.append(os.path.basename(os.path.dirname(image_name)))

        """ Extract embedding info from aligned face images """
        steps = int(np.ceil(len(image_classes) / batch_size))
        nimgs = (np.array(nimgs) - 127.5) * 0.0078125
        embeddings = [face_model(nimgs[ii * batch_size : (ii + 1) * batch_size]) for ii in tqdm(range(steps), "Embedding")]

        embeddings = normalize(np.concatenate(embeddings, axis=0))
        image_classes = np.array(image_classes)
        np.savez_compressed(dest_pickle, embeddings=embeddings, image_classes=image_classes)

    print(">>>> image_classes info:")
    print(pd.value_counts(image_classes))
    return image_classes, embeddings, dest_pickle


def image_recognize(image_classes, embeddings, det, face_model, frame, image_format="BGR"):
    if isinstance(frame, str):
        frame = imread(frame)
        image_format = "RGB"
    bbs, ccs, nimgs = do_detect_in_image(frame, det, image_format=image_format)
    if len(bbs) == 0:
        return [], [], [], []

    emb_unk = face_model((nimgs - 127.5) * 0.0078125).numpy()
    emb_unk = normalize(emb_unk)
    dists = np.dot(embeddings, emb_unk.T).T
    rec_idx = dists.argmax(-1)
    rec_dist = [dists[id, ii] for id, ii in enumerate(rec_idx)]
    rec_class = [image_classes[ii] for ii in rec_idx]

    return rec_dist, rec_class, bbs, ccs


def draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh):
    for dist, label, bb, cc in zip(rec_dist, rec_class, bbs, ccs):
        # Red color for unknown, green for Recognized
        color = (0, 0, 255) if dist < dist_thresh else (0, 255, 0)
        label = "Unknown" if dist < dist_thresh else label

        left, up, right, down = bb
        cv2.line(frame, (left, up), (right, up), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, up), (right, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (right, down), (left, down), color, 3, cv2.LINE_AA)
        cv2.line(frame, (left, down), (left, up), color, 3, cv2.LINE_AA)

        xx, yy = np.max([bb[0] - 10, 10]), np.max([bb[1] - 10, 10])
        cv2.putText(frame, "Label: {}, dist: {:.4f}".format(label, dist), (xx, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return frame


def video_recognize(image_classes, embeddings, det, face_model, video_source=0, frames_per_detect=5, dist_thresh=0.6):
    cap = cv2.VideoCapture(video_source)
    cur_frame_idx = 0
    while True:
        grabbed, frame = cap.read()
        if grabbed != True:
            break
        if cur_frame_idx % frames_per_detect == 0:
            rec_dist, rec_class, bbs, ccs = image_recognize(image_classes, embeddings, det, face_model, frame)
            cur_frame_idx = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite("{}.jpg".format(cur_frame_idx), frame)
        if key == ord("q"):
            break

        draw_polyboxes(frame, rec_dist, rec_class, bbs, ccs, dist_thresh)
        cv2.imshow("", frame)
        cur_frame_idx += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    import argparse

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, required=True, help="Saved basic_model file path, NOT model")
    parser.add_argument("-k", "--known_user", type=str, default=None, help="Folder containing user images data")
    parser.add_argument("-K", "--known_user_force", type=str, default=None, help="Folder containing user images data, force reload")
    parser.add_argument("-b", "--embedding_batch_size", type=int, default=4, help="Batch size for extracting known user embedding data")
    parser.add_argument("-s", "--video_source", type=str, default="0", help="Video source")
    parser.add_argument("-t", "--dist_thresh", type=float, default=0.6, help="Cosine dist thresh, dist lower than this will be Unknown")
    parser.add_argument("-p", "--frames_per_detect", type=int, default=5, help="Do detect every [NUM] frame")
    args = parser.parse_known_args(sys.argv[1:])[0]

    det, face_model = init_det_and_emb_model(args.model_file)
    if args.known_user_force != None:
        force_reload = True
        known_user = args.known_user_force
    else:
        force_reload = False
        known_user = args.known_user

    if known_user != None and face_model is not None:
        image_classes, embeddings, _ = embedding_images(det, face_model, known_user, args.embedding_batch_size, force_reload)
        video_source = int(args.video_source) if str.isnumeric(args.video_source) else args.video_source
        video_recognize(image_classes, embeddings, det, face_model, video_source, args.frames_per_detect, args.dist_thresh)

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


def data_drop_top_k(model, data_path, dest_file=None, deg_thresh=75, limit=0):
    cos_thresh = np.cos(np.pi * deg_thresh / 180)  # 0.25881904510252074

    image_names, image_classes, _, _, dataset_pickle_file_src = pre_process_folder(data_path)  # Reload from pickle file
    sorted_idx = np.argsort(image_classes)
    image_names, image_classes = np.array(image_names)[sorted_idx], np.array(image_classes)[sorted_idx]

    if isinstance(model, str):
        from train import NormDense

        mm = tf.keras.models.load_model(model, custom_objects={"NormDense": NormDense}, compile=False)
    else:
        mm = model
    basic_model = tf.keras.models.Model(mm.inputs[0], mm.layers[-2].output)

    output_layer = mm.layers[-1]
    centers = normalize(output_layer.weights[0].numpy(), axis=0)
    emb_num = output_layer.input_shape[-1]
    class_num = output_layer.output_shape[-1]
    top_k = centers.shape[-1] // class_num

    print(">>>> [Before] emb_num = %d, class_num = %d, top_k = %d, images = %d" % (emb_num, class_num, top_k, len(image_classes)))
    # >>>> [Before] emb_num = 256, class_num = 10572, top_k = 3, images = 490623

    cur_idx = 0  # The new index to save.
    new_image_classes, new_image_names = [], []
    total_idxes = class_num if limit == 0 else limit
    for ii in tqdm(range(total_idxes)):
        imms = image_names[image_classes == ii]
        imgs = tf.stack([tf_imread(imm) for imm in imms])
        imgs = (imgs - 127.5) * 0.0078125
        embs = normalize(basic_model(imgs).numpy(), axis=1)

        """ Find the best center """
        sub_centers = centers[:, ii * top_k : (ii + 1) * top_k]  # (256, 3)
        dists = np.dot(embs, sub_centers)
        max_sub_center_idxes = np.argmax(dists, axis=1)
        max_sub_center_count = [(max_sub_center_idxes == idx).sum() for idx in range(top_k)]
        dominant_index = np.argmax(max_sub_center_count)
        dominant_center = sub_centers[:, dominant_index]  # (256)

        """ Drop those dists too large """
        dominant_dist = dists[:, dominant_index]
        keep_idxes = dominant_dist > cos_thresh
        if keep_idxes.sum() == 0:
            print(">>>> All False, drop this index:", ii)
            continue

        new_imgs = imms[keep_idxes]
        new_image_names.extend(new_imgs)
        new_image_classes.extend([cur_idx] * len(new_imgs))
        cur_idx += 1

    """ Do shuffle again """
    new_image_classes, new_image_names = np.array(new_image_classes), np.array(new_image_names)
    shuffle_idxes = np.random.permutation(len(new_image_names))
    new_image_classes, new_image_names = new_image_classes[shuffle_idxes].tolist(), new_image_names[shuffle_idxes].tolist()

    """ Save to npz """
    if dest_file is None:
        src_name = os.path.splitext(os.path.basename(dataset_pickle_file_src))[0]
        dest_file = src_name + "_topK{}_deg{}.npz".format(top_k, deg_thresh)
    np.savez_compressed(dest_file, image_names=new_image_names, image_classes=new_image_classes)
    # with open(dest_file, "wb") as ff:
    #     pickle.dump({"image_names": new_image_names, "image_classes": new_image_classes}, ff)

    print(">>>> [After] Total classes: %d, total images: %d" % (np.max(new_image_classes) + 1, len(new_image_names)))
    # >>>> [After] Total classes: 10572, total images: 466276
    return dest_file


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-M", "--model_file", type=str, required=True, help="Saved model file path, NOT basic_model")
    parser.add_argument("-D", "--data_path", type=str, required=True, help="Original dataset path")
    parser.add_argument("-d", "--dest_file", type=str, default=None, help="Dest file path to save the processed dataset npz")
    parser.add_argument("-t", "--deg_thresh", type=int, default=75, help="Thresh value in degree, [0, 180]")
    parser.add_argument("-L", "--limit", type=int, default=0, help="Test parameter, limit converting only the first [NUM] ones")
    args = parser.parse_known_args(sys.argv[1:])[0]

    print(">>>> Output:", data_drop_top_k(args.model_file, args.data_path, args.dest_file, args.deg_thresh, args.limit))
elif __name__ == "__test__":
    model_file = "TT_mobilenet_topk_bs256.h5"
    data_path = "/datasets/faces_casia"
    dest_file = None
    deg_thresh = 75
    data_drop_top_k(model_file, dataset_pickle_file_src, dest_file, deg_thresh)

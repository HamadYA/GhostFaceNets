from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from easydict import EasyDict as edict
import time
import sys
import numpy as np
import argparse
import struct
import cv2
import sklearn
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow import keras
import os



def read_img(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(type(img))
    img = (img / 255 - 0.5) / 0.5
    return img


def get_feature(imgs, model):
    count = len(imgs)
    data = np.zeros(shape=(count, *imgs[0].shape))

    for idx, img in enumerate(imgs):
        img = img[:, :, ::-1]  #to rgb
        data[idx] = img
    
    F = model.predict(data)
    F = sklearn.preprocessing.normalize(F)
    #print('F', F.shape)
    return F

def write_bin(path, feature):
    feature = list(feature)
    with open(path, 'wb') as f:
        f.write(struct.pack('4i', len(feature), 1, 4, 5))
        f.write(struct.pack("%df" % len(feature), *feature))


def get_and_write(buffer, model):
    imgs = []
    for k in buffer:
        imgs.append(k[0])
    features = get_feature(imgs, model)
    #print(np.linalg.norm(feature))
    assert features.shape[0] == len(buffer)
    for ik, k in enumerate(buffer):
        out_path = k[1]
        feature = features[ik].flatten()
        write_bin(out_path, feature)


# batch_size = 8
# image_size = '3,112,112'
# # gpu = -1
# nomf = False
# algo = 'insightface'
# megaface_data = 'data'
# facescrub_lst = 'data/facescrub_lst'
# megaface_lst = 'data/megaface_lst'
# facescrub_root ='data/facescrub_images'
# megaface_root ='data/megaface_images'
# output = 'feature_out'
# facescrub_out = os.path.join(output, 'facescrub')
# megaface_out = os.path.join(output, 'megaface')

# model = 'GhostFaceNets/GhostFaceNet_W1.3_S1_ArcFace.h5'


def main(args):

  print(args)
  
  nets = []
  image_shape = [int(x) for x in args.image_size.split(',')]

  for model_path in model.split('|'):
    model = keras.models.load_model(model, compile=False)
    # model = backbone.to(ctx)
    # model.eval()
    print(f"LOADED MODEL FROM {model_path}")
    nets.append(model)
    
  model = nets[0]


  facescrub_out = os.path.join(args.output, 'facescrub')
  megaface_out = os.path.join(args.output, 'megaface')

  i = 0
  succ = 0
  buffer = []
  for line in open(args.facescrub_lst, 'r'):
    if i % 1000 == 0:
        print("writing fs", i, succ)
    i += 1
    image_path = line.strip()
    _path = image_path.split('/')
    a, b = _path[-2], _path[-1]
    out_dir = os.path.join(facescrub_out, a)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    image_path = os.path.join(args.facescrub_root, image_path)
    print(image_path)
    img = read_img(image_path)
    if img is None:
        print('read error:', image_path)
        continue
    out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer) == args.batch_size:
        get_and_write(buffer, model)
        buffer = []
    succ += 1
  if len(buffer) > 0:
      get_and_write(buffer, model)
      buffer = []
  print('fs stat', i, succ)

  i = 0
  succ = 0
  buffer = []
  for line in open(args.megaface_lst, 'r'):
    if i % 1000 == 0:
        print("writing mf", i, succ)
    i += 1
    image_path = line.strip()
    _path = image_path.split('/')
    a1, a2, b = _path[-3], _path[-2], _path[-1]
    out_dir = os.path.join(megaface_out, a1, a2)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        #continue
    #print(landmark)
    image_path = os.path.join(args.megaface_root, image_path)
    img = read_img(image_path)
    if img is None:
        print('read error:', image_path)
        continue
    out_path = os.path.join(out_dir, b + "_%s.bin" % (args.algo))
    item = (img, out_path)
    buffer.append(item)
    if len(buffer) == args.batch_size:
        get_and_write(buffer, model)
        buffer = []
    succ += 1
  if len(buffer) > 0:
      get_and_write(buffer, model)
      buffer = []
  print('mf stat', i, succ)


def parse_arguments(argv):
  parser = argparse.ArgumentParser()
  
  parser.add_argument('--batch_size', type=int, help='', default=8)
  parser.add_argument('--image_size', type=str, help='', default='3,112,112')
  parser.add_argument('--gpu', type=int, help='', default=0)
  parser.add_argument('--algo', type=str, help='', default='insightface')
  parser.add_argument('--facescrub-lst', type=str, help='', default='./data/facescrub_lst')
  parser.add_argument('--megaface-lst', type=str, help='', default='./data/megaface_lst')
  parser.add_argument('--facescrub-root', type=str, help='', default='./data/facescrub_images')
  parser.add_argument('--megaface-root', type=str, help='', default='./data/megaface_images')
  parser.add_argument('--output', type=str, help='', default='./feature_out')
  parser.add_argument('--model', type=str, help='', default='GhostFaceNets/GhostFaceNet_W1.3_S1_ArcFace.h5')
  parser.add_argument('--nomf', default=False, action="store_true", help='')
  return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))


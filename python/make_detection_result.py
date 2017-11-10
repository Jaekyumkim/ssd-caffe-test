import caffe
import matplotlib.pyplot as plt
import time as timelib
import pdb
import numpy as np
import os
import sys
import cv2
import time
import glob
import re
import scipy.io as si
import copy
from math import *

s_idx = 6000
f_idx = 7481

base_dir = '/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/'
txt_dir = (base_dir + 'SSD/caffe/data/KITTI/depth_gain_test/4channel_sparse_fusion_scaling/result_00002_2028_360000_big_stand_aug/Car/')
caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/4channel_sparse_fusion_scaling/deploy.prototxt', '/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/4channel_sparse_fusion_scaling/VGG_KITTI_SSD_600x150_00002_2028_360000_big_stan_aug_iter_360000.caffemodel', caffe.TEST)
labels = ["background", "Car", "Pedestrian", "Cyclist", "Van", "Truck", "Person_sitting", "Tram", "Misc"]
if not(os.path.exists(txt_dir)):
  os.makedirs(txt_dir)

def load_labels():
  total_gt_label = dict()
  for data_idx in range(s_idx,f_idx,1):
    gt_label = []
    label_path = (base_dir + 'SSD/caffe/data/KITTI/label_2/test/' + '%06d' % (data_idx) + '.txt')
    filename = open(label_path,'r')
    patten = '(\w*) ([-\d]*.\d*) ([-\d]*.\d*) .* ([-\d]*.\d*) ([-\d]*.\d*) ([-\d]*.\d*) ([-\d]*.\d*) .* .* .* .* .* .* .*'
    r = re.compile(patten)
    while True:
      line = filename.readline()
      line_split = r.findall(line)
      pdb.set_trace()
      if not line: break
      if line_split[0][0] == 'Car':
        gt_label.append(line_split)
    total_gt_label[data_idx] = gt_label
    filename.close()
  return total_gt_label

def load_predictions_3channel():
  total_test_box = dict()
  for data_idx in range(s_idx,f_idx,1):
    test_box_set = []
    test_input_xsize = 1242
    test_input_ysize = 375
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([104,117,123]))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,3,test_input_ysize,test_input_xsize)
#    dataDir = (base_dir + 'SSD/caffe/data/KITTI/image_2_ori/' + '%06d' % (data_idx) + '.png')
    dataDir = (base_dir + 'SSD/caffe/data/KITTI/depth_image/python_sparse/sparse_depthheightintensity/' + '%06d.png'%(data_idx))
    frame = caffe.io.load_image(dataDir)
#    dhi_img = caffe.io.load_image('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_image/python_sparse/sparse_depthheightintensity/'+'%06d.png'%(data_idx))
#    new_img = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.float32)
#    new_img[:,:,0] = frame[:,:,0]
#    new_img[:,:,1] = frame[:,:,1]
#    new_img[:,:,2] = frame[:,:,2]
#    new_img[:,:,3] = dhi_img[:,:,2]
#    new_img[:,:,4] = dhi_img[:,:,1]
#    new_img[:,:,5] = dhi_img[:,:,2]
    transformed_image = transformer.preprocess('data',frame)
    net.blobs['data'].data[...] = transformed_image
    out = net.forward()
    result_file = open(txt_dir+'%06d.txt'%(data_idx-s_idx),'w')
    for test_idx in range(out['detection_out'].shape[2]):
      if out['detection_out'][0][0][test_idx][1] == 1:
        test_box = np.zeros((5,),np.float32)
        test_value = copy.deepcopy(out['detection_out'][0][0][test_idx])
        test_box[0] = copy.deepcopy(test_value[2])
        test_box[1] = max(0,copy.deepcopy(test_value[3] * frame.shape[1]).astype(np.int32))
        test_box[2] = max(0,copy.deepcopy(test_value[4] * frame.shape[0]).astype(np.int32))
        test_box[3] = copy.deepcopy(test_value[5] * frame.shape[1]).astype(np.int32)
        test_box[4] = copy.deepcopy(test_value[6] * frame.shape[0]).astype(np.int32)
        result_file.write("Car -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(test_box[1],test_box[2],test_box[3],test_box[4],test_box[0]))
#    total_test_box[data_idx] = test_box_set
    print data_idx
  return total_test_box

# gt = load_labels()
#pr_4 = load_predictions_4channel()
pr_3 = load_predictions_3channel()


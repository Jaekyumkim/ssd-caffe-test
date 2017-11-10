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

net1 = caffe.Net('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/4channel_sparse_fusion/deploy_standard.prototxt', '/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/4channel_sparse_fusion/VGG_KITTI_SSD_600x150_000025_2028_360000_big_stan_aug_iter_325000.caffemodel', caffe.TEST)
net2 = caffe.Net('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/3channel_scaling/deploy.prototxt', '/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_gain_test/3channel_scaling/VGG_KITTI_SSD_600x150_00003_2028_360000_dhi_pre_iter_30000.caffemodel', caffe.TEST)

item1 = ['conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'conv6_1', 'conv6_2', 'conv7_1', 'conv7_2', 'conv8_1', 'conv8_2', 'conv9_1', 'conv9_2']
#pdb.set_trace()
conv1_1 = np.zeros([64,1,3,3])
conv1_1[:,0,:,:] = net2.params['conv1_1'][0].data[:,0,:,:]
net1.params['conv1_1_d'][0].data[...] = conv1_1
for item in item1:
	net1.params[item+'_d'][0].data[...] = net2.params[item][0].data
	net1.params[item+'_d'][1].data[...] = net2.params[item][1].data

net1.save('./4channel_fusion_weight.caffemodel')
'''
item1 = [ 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
item2 = ['fc6', 'fc7', 'conv6_1', 'conv6_2', 'conv7_1', 'conv7_2', 'conv8_1', 'conv8_2']
net.params['deconv1_1'][0].data[...] = net.params['conv1_1'][0].data
#for item in item1:
#  net.params[item + '_d'][0].data[...] = net.params[item][0].data
#  net.params[item + '_d'][1].data[...] = net.params[item][1].data
#for item in item2:
#  net.params[item + '_d'][0].data[...] = net.params[item][0].data
#  net.params[item + '_d'][1].data[...] = net.params[item][1].data
#pdb.set_trace()
net.save('./copy_vgg_4channel_deconv_test.caffemodel')
'''

'''
########## weight
net.params['conv1_2_d'][0].data[...] = net.params['conv1_2'][0].data
net.params['conv2_1_d'][0].data[...] = net.params['conv2_1'][0].data
net.params['conv2_2_d'][0].data[...] = net.params['conv2_2'][0].data
net.params['conv3_1_d'][0].data[...] = net.params['conv3_1'][0].data
net.params['conv3_2_d'][0].data[...] = net.params['conv3_2'][0].data
net.params['conv3_3_d'][0].data[...] = net.params['conv3_3'][0].data
net.params['conv4_1_d'][0].data[...] = net.params['conv4_1'][0].data
net.params['conv4_2_d'][0].data[...] = net.params['conv4_2'][0].data
net.params['conv4_3_d'][0].data[...] = net.params['conv4_3'][0].data
net.params['conv5_1_d'][0].data[...] = net.params['conv5_1'][0].data
net.params['conv5_2_d'][0].data[...] = net.params['conv5_2'][0].data
net.params['conv5_3_d'][0].data[...] = net.params['conv5_3'][0].data
net.params['fc6_d'][0].data[...] = net.params['fc6'][0].data
net.params['fc7_d'][0].data[...] = net.params['fc7'][0].data
net.params['conv6_1_d'][0].data[...] = net.params['conv6_1'][0].data
net.params['conv6_2_d'][0].data[...] = net.params['conv6_2'][0].data
net.params['conv7_1_d'][0].data[...] = net.params['conv7_1'][0].data
net.params['conv7_2_d'][0].data[...] = net.params['conv7_2'][0].data
net.params['conv8_1_d'][0].data[...] = net.params['conv8_1'][0].data
net.params['conv8_2_d'][0].data[...] = net.params['conv8_2'][0].data

########## bias
net.params['conv1_2_d'][0].data[...] = net.params['conv1_2'][1].data
net.params['conv2_1_d'][0].data[...] = net.params['conv2_1'][1].data
net.params['conv2_2_d'][0].data[...] = net.params['conv2_2'][1].data
net.params['conv3_1_d'][0].data[...] = net.params['conv3_1'][1].data
net.params['conv3_2_d'][0].data[...] = net.params['conv3_2'][1].data
net.params['conv3_3_d'][0].data[...] = net.params['conv3_3'][1].data
net.params['conv4_1_d'][0].data[...] = net.params['conv4_1'][1].data
net.params['conv4_2_d'][0].data[...] = net.params['conv4_2'][1].data
net.params['conv4_3_d'][0].data[...] = net.params['conv4_3'][1].data
net.params['conv5_1_d'][0].data[...] = net.params['conv5_1'][1].data
net.params['conv5_2_d'][0].data[...] = net.params['conv5_2'][1].data
net.params['conv5_3_d'][0].data[...] = net.params['conv5_3'][1].data
net.params['fc6_d'][0].data[...] = net.params['fc6'][1].data
net.params['fc7_d'][0].data[...] = net.params['fc7'][1].data
net.params['conv6_1_d'][0].data[...] = net.params['conv6_1'][1].data
net.params['conv6_2_d'][0].data[...] = net.params['conv6_2'][1].data
net.params['conv7_1_d'][0].data[...] = net.params['conv7_1'][1].data
net.params['conv7_2_d'][0].data[...] = net.params['conv7_2'][1].data
net.params['conv8_1_d'][0].data[...] = net.params['conv8_1'][1].data
net.params['conv8_2_d'][0].data[...] = net.params['conv8_2'][1].data
'''

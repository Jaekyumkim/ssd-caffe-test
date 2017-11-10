from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe
import numpy as np
import yaml
import pdb

class broadcasting(caffe.Layer):
  def setup(self, bottom, top):
    params = eval(self.param_str)
    self.pad_num = params['mean']

  def reshape(self, bottom, top):
#    pdb.set_trace()
    top[0].reshape(bottom[0].data.shape[0],bottom[0].data.shape[1],3*bottom[0].data.shape[2],3*bottom[0].data.shape[3])
#    top[0].reshape(bottom[0].data.shape[0],1,bottom[0].data.shape[2]/3,bottom[0].data.shape[3]/3)
  def forward(self, bottom, top):
    bottom0_data = bottom[0].data
    bottom0_data_padded = -1 * self.pad_num * np.ones((bottom0_data.shape[0],bottom0_data.shape[1],bottom0_data.shape[2]+2,bottom0_data.shape[3]+2))
    bottom0_data_padded[:,:,1:bottom0_data.shape[2]+1,1:bottom0_data.shape[3]+1] = bottom0_data
    num_data = np.zeros((bottom0_data.shape[0],bottom0_data.shape[1],3*bottom0_data.shape[2],3*bottom0_data.shape[3]))
#    for batch_idx in range(bottom0_data.shape[0]):
#      for chan_idx in range(bottom0_data.shape[1]):
    for w_idx in range(0,bottom0_data_padded.shape[2]-2,1):
      for h_idx in range(0,bottom0_data_padded.shape[3]-2,1):
        num_data[:,:,3*w_idx:3*w_idx+3,3*h_idx:3*h_idx+3] = bottom0_data_padded[:,:,w_idx:w_idx+3,h_idx:h_idx+3]
    top[0].data[...] = num_data
  def backward(self, bottom, top, propagate_down):
    pass
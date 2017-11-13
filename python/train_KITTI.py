import caffe
import matplotlib.pyplot as plt
import time as timelib
import pdb
import numpy as np
import cv2

caffe.set_mode_gpu()
caffe.set_device(1)
#pdb.set_trace()
save_path = ('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/ssd-caffe-test/python/test_img/')
solver = caffe.get_solver('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/ssd-caffe-test/data/KITTI/test/solver.prototxt')
#solver.net.copy_from('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel')
#solver.net.copy_from('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/4channel_scatter_fusion/VGG_KITTI_SSD_600x150_00003_2430_360000_iter_360000.caffemodel')
max_iter = 120000
#fig, axes = plt.subplots()
#fig.show()
#loss_list = []
iter0 = solver.iter

while solver.iter < max_iter:
#  pdb.set_trace()
  solver.step(1)
#  label1=solver.net.blobs['label1'].data
#  label2=solver.net.blobs['label2'].data
  data=solver.net.blobs['data'].data
  label=solver.net.blobs['label'].data
#  data1=solver.net.blobs['data1'].data
#  data2=solver.net.blobs['data_dhi'].data
#  pdb.set_trace()
  for num_idx in range(2):
    img = data[num_idx,:,:,:].transpose((1,2,0)).astype(np.uint8)
    rgb = img[:,:,0:3]
    dhi = img[:,:,3:6]
#    cv2.imwrite(save_path+'%06d.png'%(num_idx),img[:,:,0:3])
#    cv2.imwrite(save_path+'depth_%06d.png'%(num_idx),img[:,:,3:6])
    x_min = (label[0,0,0,:][3]*data.shape[3]).astype(np.int32)
    y_min = (label[0,0,0,:][4]*data.shape[2]).astype(np.int32)
    x_max = (label[0,0,0,:][5]*data.shape[3]).astype(np.int32)
    y_max = (label[0,0,0,:][6]*data.shape[2]).astype(np.int32)
    cv2.imwrite(save_path+'%06d.png'%(num_idx),rgb)
    pdb.set_trace()
    #cv2.rectangle(rgb, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    #cv2.rectangle(rgb, (x_min, y_min), (x_max,y_max), (0, 0, 255), 2)
    cv2.imwrite(save_path+'%06d.png'%(num_idx),rgb)

#  for num_idx in range(4):
#    img = data1[num_idx,:,:,:].transpose((1,2,0)).astype(np.uint8)
#    cv2.imwrite(save_path+'%06d.png'%(num_idx+4),img)
#  cv2.imshow('image', img)
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()  
#  if solver.iter == 5:
  pdb.set_trace()
#  loss = solver.net.blobs['mbox_loss'].data.flatten()
#  loss_list.append(loss)
#  if solver.iter % 100 == 0:
#    axes.clear()
#    axes.plot(range(iter0, iter0+len(loss_list)), loss_list)
#    axes.grid(True)
#    fig.canvas.draw()
#    plt.pause(0.01)

#fig.savefig('fig_iter_%d.png' % solver.iter)
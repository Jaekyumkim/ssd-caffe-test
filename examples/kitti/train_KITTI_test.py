import caffe
import matplotlib.pyplot as plt
import time as timelib
import pdb
import numpy as np
from PIL import Image
#import cv2

caffe.set_mode_gpu()
caffe.set_device(1)
#pdb.set_trace()
save_path = ('/media/user/c45eb821-d419-451d-b171-3152a8436ba2/jkkim/caffe/data/KITTI/test/test_img/')
solver = caffe.get_solver('/media/user/c45eb821-d419-451d-b171-3152a8436ba2/jkkim/caffe/data/KITTI/test/solver.prototxt')
solver.net.copy_from('/media/user/c45eb821-d419-451d-b171-3152a8436ba2/jkkim/caffe/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel')
#solver.net.copy_from('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/4channel_scatter_fusion/VGG_KITTI_SSD_600x150_00003_2430_360000_iter_360000.caffemodel')
max_iter = 120000
#fig, axes = plt.subplots()
#fig.show()
#loss_list = []
iter0 = solver.iter

while solver.iter < max_iter:
#  pdb.set_trace()
  solver.step(1)
#  pdb.set_trace()
#  label1=solver.net.blobs['label1'].data
#  label2=solver.net.blobs['label2'].data
  data=solver.net.blobs['data'].data
#  label=solver.net.blobs['label'].data
#  data1=solver.net.blobs['data1'].data
#  data2=solver.net.blobs['data_dhi'].data
  for num_idx in range(4):
#    pdb.set_trace()
    img = data[num_idx,:,:,:].transpose((1,2,0)).astype(np.uint8)
    rgb_img = img[:,:,0:3]
    d_img = img[:,:,3]
    rgb_im = Image.fromarray(rgb_img)
    d_im = Image.fromarray(d_img)
    rgb_im.save(save_path+'rgb_%06d.png'%(num_idx))
    d_im.save(save_path+'d_%06d.png'%(num_idx))
  pdb.set_trace()
#    im = Image.fromarray(img)
#    im.save(save_path+'%06d.png'%(num_idx))
#    cv2.imwrite(save_path+'%06d.png'%(num_idx),img[:,:,0:3])
#    cv2.imwrite(save_path+'depth_%06d.png'%(num_idx),img[:,:,3])
#  for num_idx in range(4):
#    img = data1[num_idx,:,:,:].transpose((1,2,0)).astype(np.uint8)
#    cv2.imwrite(save_path+'%06d.png'%(num_idx+4),img)
#  cv2.imshow('image', img)
#  cv2.waitKey(0)
#  cv2.destroyAllWindows()  
#  if solver.iter == 5:
#  loss = solver.net.blobs['mbox_loss'].data.flatten()
#  loss_list.append(loss)
#  if solver.iter % 100 == 0:
#    axes.clear()
#    axes.plot(range(iter0, iter0+len(loss_list)), loss_list)
#    axes.grid(True)
#    fig.canvas.draw()
#    plt.pause(0.01)

#fig.savefig('fig_iter_%d.png' % solver.iter)
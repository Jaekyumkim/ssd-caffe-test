import lmdb
import caffe
import scipy.io as si
import hickle as hkl
import numpy as np
import cv2
import hickle as hkl
import pdb
from PIL import Image
import re 
import time
import random

def write(images, image_dhi, labels = []):
    """
    Write a single image or multiple images and the corresponding label(s).
    The imags are expected to be two-dimensional NumPy arrays with
    multiple channels (if applicable).
        
    :param images: input images as list of numpy.ndarray with height x width x channels
    :type images: [numpy.ndarray]
    :param labels: corresponding labels (if applicable) as list
    :type labels: [float]
    :return: list of keys corresponding to the written images
    :rtype: [string]
    """
    
#    if len(labels) > 0:
#        assert len(images) == len(labels)
    keys = []
    env = lmdb.open('/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/depth_image/depth_image_code/', map_size = max(1099511627776, len(images)*images[0].nbytes))
    with env.begin(write = True) as transaction:
        for i in range(len(images)):
            datum = caffe.proto.caffe_pb2.AnnotatedDatum()
            datum.datum2.channels = images[i].shape[2]
            datum.datum2.height = images[i].shape[0]
            datum.datum2.width = images[i].shape[1]
            datum.datum.channels = images[i].shape[2]
            datum.datum.height = images[i].shape[0]
            datum.datum.width = images[i].shape[1]
            datum.datum.encoded = True
            datum.datum2.encoded = True
            datum.type = caffe.proto.caffe_pb2.AnnotatedDatum.BBOX
            veh_group = caffe.proto.caffe_pb2.AnnotationGroup()
            ped_group = caffe.proto.caffe_pb2.AnnotationGroup()
            cyc_group = caffe.proto.caffe_pb2.AnnotationGroup()
            datum.annotation_group.extend([veh_group,ped_group,cyc_group])
            veh_group.group_label = 1
            ped_group.group_label = 2
            cyc_group.group_label = 3
            for label_idx in range(len(labels[i])):
                anno = caffe.proto.caffe_pb2.Annotation()
                anno.instance_id = label_idx
                anno.bbox.xmin = float(labels[i][label_idx][0][1])/images[i].shape[1]
                anno.bbox.ymin = float(labels[i][label_idx][0][2])/images[i].shape[0]
                anno.bbox.xmax = float(labels[i][label_idx][0][3])/images[i].shape[1]
                anno.bbox.ymax = float(labels[i][label_idx][0][4])/images[i].shape[0]
                if labels[i][label_idx][0][0] == 'Car':
                    veh_group.annotation.extend([anno])
                elif labels[i][label_idx][0][0] == 'Pedestrian':
                    ped_group.annotation.extend([anno])
                else:
                    cyc_group.annotation.extend([anno])
#            pdb.set_trace()
            assert images[i].dtype == np.uint8 or images[i].dtype == np.float, "currently only numpy.uint8 and numpy.float images are supported"
                    
            if images[i].dtype == np.uint8:
                # For NumPy 1.9 or higher, use tobytes() instead!
                flag, image_encoded = cv2.imencode('.png', images[i])
                assert(flag == True)
                flag, image_dhi_encoded = cv2.imencode('.png', image_dhi[i])
                datum.datum.data = image_encoded.tobytes()
                datum.datum2.data = image_dhi_encoded.tobytes()
            else:
                datum.datum.float_data.extend(images[i].transpose(2, 0, 1).flat)

            keystr = '{:08}'.format(i)
            transaction.put(keystr, datum.SerializeToString())

#            key = to_key(_write_pointer)
#            keys.append(key)
                    
#            transaction.put(key.encode('ascii'), datum.SerializeToString());
#            _write_pointer += 1
            
    return

data_path = '/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI/image_2_ori_train_test/train/'
img_total = []
img_dhi_total = []
start = time.time()
idx = range(10); random.shuffle(idx)
for i in range(10):
    i = idx[i]
    rgb_img = (cv2.imread(data_path + '%06d' % (i) + '.png')).astype(np.uint8)
    dhi_img = Image.open(data_path + '../../sparse_dhi/' + '%06d.png'%(i))
    dhi_img = np.array(dhi_img, dtype=np.uint8)
    new_img = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3),dtype=np.uint8)
    new_img2 = np.zeros((rgb_img.shape[0],rgb_img.shape[1],3),dtype=np.uint8)    
    new_img[:,:,0:3] = rgb_img
    new_img2[:,:,0:3] = dhi_img
    img_total.append(new_img)
    img_dhi_total.append(new_img2)

label_total = []
patten = '(\w*) .* .* .* ([-\d]*.\d*) ([-\d]*.\d*) ([-\d]*.\d*) ([-\d]*.\d*) .* .* .* .* .* .* .*'
r = re.compile(patten)
for i in range(10):
    label_group = []
    fi = open(data_path + '../../label_2_3cls/train/' + '%06d.txt'%(i))
    while True:
        line = fi.readline()
        line_split = r.findall(line)
        if not line: break
        label_group.append(line_split)
    label_total.append(label_group)
end = time.time()
print '%s'%(end-start)
write(img_total, img_dhi_total, label_total)
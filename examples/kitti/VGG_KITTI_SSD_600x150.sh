cd /media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/ssd-caffe-test
./build/tools/caffe train \
--solver="/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/ssd-caffe-test/data/KITTI/test/solver.prototxt" \
--weights="/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/ssd-caffe-test/models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel" \
--gpu 0 2>&1 | tee 
cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../../../../

cd $root_dir

redo=1
data_root_dir="/media/user/4b3dfae8-c6b6-4a03-936d-d8fee7ff0b89/SSD/caffe/data/KITTI"
dataset_name="KITTI"
mapfile="$root_dir/data/$dataset_name/3cls_new/datascript_3classes/labelmap_KITTI.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=png --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/3cls_new/datascript_3classes/$subset.txt $data_root_dir/../$dataset_name/3cls_new/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done

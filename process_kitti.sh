cd $(readlink -f `dirname $0`)

export PYTHONPATH="."
export OMP_NUM_THREADS=8

kitti_root=./data/occupancy/semanticKITTI/RGB/
kitti_preprocess_root=./data/occupancy/semanticKITTI/lss-semantic_kitti_voxel_label
data_info_path=./tools/data_converter/kitti_process/semantic-kitti.yaml

python ./tools/data_converter/kitti_process/semantic_kitti_preprocess.py \
--kitti_root $kitti_root \
--kitti_preprocess_root $kitti_preprocess_root \
--data_info_path $data_info_path

_base_ = [
    '../../datasets/custom_nus-3d.py',
    '../../_base_/default_runtime.py'
]

# kitti lidar coordinate: x front, y left, z up

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
# to be checked, from https://github.com/open-mmlab/mmclassification/blob/master/configs/efficientnet/efficientnet-b7_8xb32-01norm_in1k.py
# fix: https://github.com/open-mmlab/mmdetection/blob/master/configs/_base_/datasets/coco_detection.py
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left', 'right']

# 20 classes with unlabeled
class_names = [
    'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
    'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
    'pole', 'traffic-sign'
]

point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
occ_size = [256, 256, 32]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [2, 2, 2]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config={
    'input_size': (384, 1280),
    # 'resize': (-0.06, 0.11),
    # 'rot': (-5.4, 5.4),
    # 'flip': True,
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

numC_Trans = 128
voxel_channels = [128, 256, 512]
voxel_out_indices = (0, 1, 2)
voxel_out_channels = [128, 128, 128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='BEVDepthOccupancy',
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(type='Pretrained', prefix='backbone', 
        checkpoint='./pretrain/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4], 
        out_channels=[128, 128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        downsample=8, 
        numC_input=640,
        cam_channels=30,
        semkitti=False,
        loss_depth_weight=1.0,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False,
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        num_stage=3,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=norm_cfg,
    ),
    img_bev_encoder_neck=dict(
        type='SECONDFPN3D',
        norm_cfg=norm_cfg,
        in_channels=voxel_channels,
        upsample_strides=[1, 2, 4],
        out_channels=voxel_out_channels,
    ),
    pts_bbox_head=dict(
        type='OccHead',
        num_level=1,
        in_channels=[sum(voxel_out_channels), ],
        out_channel=20,
        semantic_kitti=True,
        point_cloud_range=point_cloud_range,
        # for point-level 
        supervise_points=False,
        sampling_img_feats=True,
        in_img_channels=640,
        soft_weights=True,
        semkitti_loss_weight_cfg={
            "voxel_ce": 1.0,
            "voxel_sem_scal": 1.0,
            "voxel_geo_scal": 1.0,
            "voxel_ohem": 0.0,
            "voxel_lovasz": 0.0,
            "frustum_dist": 0.0,
        },
    ),
    train_cfg=dict(pts=None),
    test_cfg=dict(pts=None),
)

dataset_type = 'CustomSemanticKITTILssDataset'
data_root = './data/occupancy/semanticKITTI/RGB/'
ann_file = './data/occupancy/semanticKITTI/lss-semantic_kitti_voxel_label/labels/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=True, colorjitter=False, #############
         data_config=data_config, load_depth=False, img_norm_cfg=img_norm_cfg),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='CreateDepthFromLiDAR', point_cloud_range=point_cloud_range, grid_size=occ_size),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points_occ', "points_uv"], 
            meta_keys=['pc_range', 'occ_size', 'sequence', ' frame_id', 'img_filename' ]),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_SemanticKitti', is_train=False, 
         data_config=data_config, load_depth=False, img_norm_cfg=img_norm_cfg),
    dict(type='LoadSemKittiAnnotation', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False), 
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size' ,'sequence', 'frame_id', 'img_filename' ]),
]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    queue_length=1,
    split='test',
    camera_used=camera_used,
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=ann_file,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        queue_length=1,
        split='train',
        camera_used=camera_used,
    ),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    step=[20, 25],
)

checkpoint_config = dict(max_keep_ckpts=2, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=30)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='semkitti_combined_IoU',
    rule='greater',
)
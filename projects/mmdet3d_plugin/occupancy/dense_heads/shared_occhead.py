import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv

from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from .lovasz_softmax import lovasz_softmax
from projects.mmdet3d_plugin.utils import cm_to_ious, query_points_from_voxels, per_class_iu, fast_hist_crop

import pdb

'''
TODO:
[1] points with image features
[2] shared decoder for voxel and points, which contradicts with [1] potentially
'''

@HEADS.register_module()
class SharedOccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        # for voxel-level supervison
        supervise_voxel=True,
        num_level=1,
        # for point-level supervision
        supervise_points=False,
        num_img_level=1,
        in_img_channels=512,
        sampling_img_feats=False,
        soft_weights=False,
        # loss weights
        loss_weight_cfg=None,
        loss_voxel_prototype='cylinder3d',
        # network settings
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        train_cfg=None,
        test_cfg=None,
    ):
        super(SharedOccHead, self).__init__()
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.supervise_voxel = supervise_voxel
        self.supervise_points = supervise_points
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range))
        self.loss_voxel_prototype = loss_voxel_prototype
        
        assert self.supervise_voxel and self.supervise_points
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 0.0,
                "loss_voxel_lovasz_weight": 1.0,
                "loss_point_ce_weight": 0.0,
                "loss_point_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        
        # point losses
        self.loss_point_ce_weight = self.loss_weight_cfg.get('loss_point_ce_weight', 1.0)
        self.loss_point_lovasz_weight = self.loss_weight_cfg.get('loss_point_lovasz_weight', 1.0)
        
        # voxel-level prediction
        hidden_dims = in_channels
        self.point_occ_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.Linear(hidden_dims, out_channel),
        )
        
        # voxel-level & image-level fusion
        self.soft_weights = soft_weights
        self.num_img_level = num_img_level
        self.in_img_channels = in_img_channels
        self.sample_img_feats = sampling_img_feats
            
        if self.sample_img_feats:
            self.img_feat_reduce = nn.Conv2d(self.in_img_channels, self.in_channels, kernel_size=1)

        # loss functions
        if self.loss_voxel_prototype == 'cylinder3d':
            self.ignore_label = 0
            self.voxel_ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        
        elif self.loss_voxel_prototype == 'tpv':
            # supervise all 18 classes
            self.ignore_label = None
            self.voxel_ce_criterion = torch.nn.CrossEntropyLoss()
            
        else:
            pass
        
        self.point_ce_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    def sample_point_feats(self, points, voxel_feats):
        # normalize points coordinates
        pc_range_min = self.point_cloud_range[:3]
        pc_range = self.point_cloud_range[3:] - pc_range_min
        
        # using the last few levels
        voxel_feats = voxel_feats[:self.num_level]
        
        point_feats = []
        for batch_index, points_i in enumerate(points):
            points_i = points_i[:, :3].float()            
            points_i = (points_i - pc_range_min.type_as(points_i)) / pc_range.type_as(points_i)
            # [0, 1] ==> [-1, 1]
            points_i = (points_i * 2) - 1
            
            # grid_sample
            points_i = points_i.view(1, 1, 1, -1, 3)
            point_feats_i = 0
            for voxel_level_index, voxel_feat in enumerate(voxel_feats):
                point_feats_i += F.grid_sample(voxel_feat[batch_index].unsqueeze(0), points_i, mode='bilinear', align_corners=False)
            
            point_feats_i = point_feats_i.squeeze()
            point_feats.append(point_feats_i)
        
        return point_feats
    
    def sampling_img_feats(self, img_feats, points_uv):
        # reduce the channel of img_feats to align with voxel_feats
        num_batch, num_cam = img_feats.shape[:2]
        img_feats = self.img_feat_reduce(img_feats.flatten(0, 1))
        img_feats = img_feats.view(num_batch, num_cam, *img_feats.shape[1:])
        
        sample_img_feats_list = []
        sample_indices_list = []
        for batch_index in range(img_feats.shape[0]):
            points_img_feats, points_indices = feature_sampling([img_feats[batch_index]], points_uv[batch_index])
            
            sample_img_feats_list.append(points_img_feats)
            sample_indices_list.append(points_indices)
        
        return sample_img_feats_list, sample_indices_list
        
    
    def forward(self, voxel_feats, points=None, img_metas=None, img_feats=None, points_uv=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        # sampling point features
        point_feats_list = self.sample_point_feats(points=points, voxel_feats=voxel_feats)
        
        if self.sample_img_feats:
            joint_img_feats_list, sample_img_indices_list = \
                self.sampling_img_feats(img_feats=img_feats, points_uv=points_uv)
        
        # concatenate voxel features and point features
        batch_voxel_feats = voxel_feats[-1]
        output_voxels = []
        output_points = []
        
        for batch_index in range(batch_voxel_feats.shape[0]):
            # [C, x, Y, Z] == [C, num_voxel]
            voxel_shape = batch_voxel_feats[batch_index].shape[1:]
            flatten_voxel_feats = batch_voxel_feats[batch_index].flatten(1)
            num_voxel = flatten_voxel_feats.shape[1]
            # [C, num_point]
            flatten_point_feats = point_feats_list[batch_index]
            # [num_voxel + num_point, C]
            joint_feats = torch.cat((flatten_voxel_feats, flatten_point_feats), dim=1).t()
            
            if self.sample_img_feats:
                joint_img_feats_i = joint_img_feats_list[batch_index]
                sample_img_indices_i = sample_img_indices_list[batch_index]
                
                for cam_index in range(len(joint_img_feats_i)):
                    joint_feats[sample_img_indices_i[cam_index]] = joint_feats[sample_img_indices_i[cam_index]] + joint_img_feats_i[cam_index]
            
            joint_output = self.point_occ_mlp(joint_feats)
            # [C, X, Y, Z]
            output_voxels_i = joint_output[:num_voxel].t().contiguous().view(-1, *voxel_shape)
            output_voxels.append(output_voxels_i)
            output_points.append(joint_output[num_voxel:])
        
        output_voxels = [torch.stack(output_voxels, dim=0)]
        res = {
            'output_voxels': output_voxels,
            'output_points': output_points,
        }
        
        return res

    def loss_voxel_single(self, output_voxels, target_voxels, tag):
        # upsample to the target-size
        output_voxels = F.interpolate(output_voxels, size=target_voxels.shape[-3:], mode='trilinear', align_corners=False).contiguous()
        target_voxels = target_voxels.long()
        
        # compute voxel mIoU
        loss_dict = {}
        # cross-entropy
        if self.loss_voxel_ce_weight > 0:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.voxel_ce_criterion(output_voxels, 
                    target_voxels) * self.loss_voxel_ce_weight
        
        # lovasz_softmax
        if self.loss_voxel_lovasz_weight > 0:
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = lovasz_softmax(torch.softmax(output_voxels, dim=1), 
                    target_voxels, ignore=self.ignore_label) * self.loss_voxel_lovasz_weight
        
        return loss_dict

    def loss_point_single(self, output_points, target_points, tag):
        target_points = target_points[:, -1].long()        
        loss_dict = {}
        
        # compute mIoU
        with torch.no_grad():
            output_clses = torch.argmax(output_points, dim=1)
            target_points_np = target_points.cpu().numpy()
            output_clses_np = output_clses.cpu().numpy()
            unique_label = np.arange(16)
            
            hist = fast_hist_crop(output_clses_np, target_points_np, unique_label)
            
            iou = per_class_iu(hist)
            mean_iou = np.nanmean(iou)
            loss_dict['point_mean_iou'] = torch.tensor(mean_iou).cuda()
        
        # cross entropy
        if self.loss_point_ce_weight > 0:
            loss_dict['loss_point_ce_{}'.format(tag)] = self.point_ce_criterion(output_points, 
                    target_points) * self.loss_point_ce_weight
        
        # lovasz softmax
        if self.loss_point_lovasz_weight > 0:
            num_points, num_class = output_points.shape
            output_points = output_points.permute(1, 0).contiguous().view(1, num_class, num_points, 1, 1)
            loss_dict['loss_point_lovasz_{}'.format(tag)] = lovasz_softmax(torch.softmax(output_points, dim=1), 
                    target_points, ignore=0) * self.loss_point_lovasz_weight
        
        return loss_dict

    def loss(self, output_voxels=None, target_voxels=None, output_points=None, target_points=None, img_metas=None):
        loss_dict = {}
        # 1. compute the losses for voxel-level semantic occupancy
        if self.supervise_voxel:
            for index, output_voxel in enumerate(output_voxels):
                loss_dict.update(self.loss_voxel_single(output_voxel, target_voxels, tag='{}'.format(index)))
        
        # 2. compute the losses for point-level semantic segmentation
        if self.supervise_points:
            for index, output_point in enumerate(output_points):
                loss_dict.update(self.loss_point_single(output_point, target_points[index], tag='{}'.format(index)))
        
        return loss_dict
    
def feature_sampling(mlvl_feats, reference_points_2d):
    '''
    mlvl_feats: list of image features [num_cam, num_channel, h', w']
    reference_points_2d: [num_points, num_cam, 3] in [u, v, d]
    '''
    
    assert len(mlvl_feats) == 1 # only support num_level = 1
    num_points, num_cam = reference_points_2d.shape[:2]
    eps = 1e-5
    mask = (reference_points_2d[..., 2] > eps) \
        & (reference_points_2d[..., 0] > -1.0) \
        & (reference_points_2d[..., 0] < 1.0) \
        & (reference_points_2d[..., 1] > -1.0) \
        & (reference_points_2d[..., 1] < 1.0)

    sampled_feats_list = []
    sampled_indices_list = []
    for lvl, feat in enumerate(mlvl_feats):
        N, C, H, W = feat.size()
        # [num_cam, num_points, 1, 2]
        reference_points_cam_lvl = reference_points_2d[..., :2].permute(1, 0, 2).unsqueeze(2)
        for cam_index in range(N):
            sampled_indices = mask[:, cam_index].nonzero()[:, 0]
            valid_reference_points = reference_points_cam_lvl[cam_index : cam_index + 1, sampled_indices]
            sampled_feat = F.grid_sample(feat[cam_index : cam_index + 1], valid_reference_points, mode='bilinear', align_corners=False)
            
            sampled_feats_list.append(sampled_feat.squeeze().t())
            sampled_indices_list.append(sampled_indices)
    
    '''
    return:
        sampled_feats_list: list of tensor (num_visible_points, num_channel)
        sampled_indices_list: list of tensor (num_visible_points)
    '''
    
    return sampled_feats_list, sampled_indices_list
        

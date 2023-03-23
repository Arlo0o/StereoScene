# Copyright (c) Phigent Robotics. All rights reserved.
import math
import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.ops.bev_pool import bev_pool
from mmdet3d.ops.voxel_pooling import voxel_pooling
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F
import numpy as np

from .ViewTransformerLSSBEVDepth import *
from mmdet3d.ops import Voxelization
from mmdet3d.models import builder

import pdb

@NECKS.register_module()
class ViewTransformerLiftSplatShootSpconv(ViewTransformerLSSBEVDepth):
    def __init__(self, loss_depth_weight, voxel_layer, voxel_encoder, middle_encoder, **kwargs):
        super(ViewTransformerLiftSplatShootSpconv, self).__init__(loss_depth_weight=loss_depth_weight, **kwargs)
        
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
    
    def shuffle_points(self, points):
        return points[torch.randperm(points.shape[0])]
    
    def range_filter(self, points):
        points_xyz = ((points[:, :3] - (self.bx - self.dx / 2.)) / self.dx).long()

        # filter out points that are outside box
        kept = (points_xyz[:, 0] >= 0) & (points_xyz[:, 0] < self.nx[0]) \
               & (points_xyz[:, 1] >= 0) & (points_xyz[:, 1] < self.nx[1]) \
               & (points_xyz[:, 2] >= 0) & (points_xyz[:, 2] < self.nx[2])
        
        return kept
    
    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res = res[self.range_filter(res)]
            res = self.shuffle_points(res)
            # res_voxels: tensors of shape [num_max_voxel, num_points, num_features]
            # res_coors: tensors of shape [num_max_voxel, 3], format in [z, y, x]
            # res_num_points: tensors of shape [num_max_voxel, ], number of points in the voxel
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            
            # print(res_voxels.shape, res_num_points.float().mean(), res_num_points.float().max())
        
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        
        # coors: [N, 4] in the format (batch_idx, z, y, x)
        coors_batch = torch.cat(coors_batch, dim=0)
        
        return voxels, num_points, coors_batch
    
    def voxel_spconv(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        num_points = N * D * H * W
        
        x = x.reshape(B, num_points, C)
        geom_feats = geom_feats.reshape(B, num_points, 3)
        # [batch, num_points, (3 + C)]
        points = torch.cat((geom_feats, x), dim=2)
        
        # flatten x
        voxels, num_points, coors = self.voxelize(points)
        return voxels, num_points, coors

    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input) = input[:8]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        img_feat = x[:, self.D:self.D + self.numC_Trans, ...]
        depth_prob = self.get_depth_dist(depth_digit)

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat with voxel-generator from spconv
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans, bda)
        
        # voxelize
        voxels, num_points, coors = self.voxel_spconv(geom, volume)
        
        # avg-pool inside each voxel to extract voxel features
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        
        # spconv encoding ==> convert to 3D volumess
        x = self.middle_encoder(voxel_features, coors, B)
        
        ''' How voxelnet extracts voxel features:
        
        voxels, num_points, coors = self.voxelize(points)
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        x = self.middle_encoder(voxel_features, coors, batch_size)
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x
        
        '''
        # multi-scale features in [b, c, x, y, z]
        x = [voxel_feat.permute(0, 1, 4, 3, 2) for voxel_feat in x]
        # for voxel_feat in x:
        #     tmp = voxel_feat.reshape(voxel_feat.shape[1], -1)
        #     occupied_ratio = (tmp.abs().sum(dim=0) > 0).sum() / tmp.shape[1]
        #     print(voxel_feat.shape, occupied_ratio)

        '''
        torch.Size([1, 128, 128, 128, 4]) tensor(0.9311, device='cuda:0')
        torch.Size([1, 128, 256, 256, 8]) tensor(0.4718, device='cuda:0')
        torch.Size([1, 64, 512, 512, 16]) tensor(0.0358, device='cuda:0')
        '''
        
        return x, depth_prob

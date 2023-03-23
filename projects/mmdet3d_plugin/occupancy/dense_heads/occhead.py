import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv

from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from .lovasz_softmax import lovasz_softmax
from .mlp import Mlp
from sklearn.metrics import confusion_matrix
from projects.mmdet3d_plugin.utils import cm_to_ious, query_points_from_voxels, \
    per_class_iu, fast_hist_crop, SoftDiceLossWithProb, PositionAwareLoss
from .bevocc_head_kitti import CE_ssc_loss, sem_scal_loss, geo_scal_loss
from projects.mmdet3d_plugin.utils.semkitti import semantic_kitti_class_frequencies, kitti_class_names, \
    geo_scal_loss, sem_scal_loss, CE_ssc_loss, KL_sep, OHEM_CE_ssc_loss, OHEM_CE_sc_loss, compute_frustum_dist_loss
from projects.mmdet3d_plugin.utils.ssc_metric import SSCMetrics

import pdb

'''
TODO:
[1] points with image features
[2] shared decoder for voxel and points, which contradicts with [1] potentially
'''

@HEADS.register_module()
class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        out_point_channel=None,
        semantic_kitti=False,
        # for voxel-level supervison
        supervise_voxel=True,
        num_level=1,
        # for point-level supervision
        num_img_level=1,
        in_img_channels=512,
        sampling_img_feats=False,
        soft_weights=False,
        supervise_points=False,
        # loss weights
        loss_weight_cfg=None,
        semkitti_loss_weight_cfg=None,
        loss_voxel_prototype='cylinder3d',
        use_ohem_loss=False,
        use_sc_ohem_loss=False,
        ohem_topk=0.25,
        # network settings
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        with_cp=False,
        train_cfg=None,
        test_cfg=None,
    ):
        super(OccHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.semantic_kitti = semantic_kitti
        self.supervise_voxel = supervise_voxel
        self.supervise_points = supervise_points
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range))
        
        self.with_cp = with_cp
        self.loss_voxel_prototype = loss_voxel_prototype
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
                "loss_point_ce_weight": 1.0,
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
        if self.supervise_voxel:
            self.occ_convs = nn.ModuleList()
            for i in range(self.num_level):
                mid_channel = self.in_channels[i] // 2
                occ_conv = nn.Sequential(
                    build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                            out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                    build_norm_layer(norm_cfg, mid_channel)[1],
                    nn.ReLU(inplace=True),
                    build_conv_layer(conv_cfg, in_channels=mid_channel, 
                            out_channels=out_channel, kernel_size=1, stride=1, padding=0),
                )
                self.occ_convs.append(occ_conv)
        
        # point-level prediction
        if self.supervise_points:
            self.soft_weights = soft_weights
            self.num_img_level = num_img_level
            self.in_img_channels = in_img_channels
            self.sampling_img_feats = sampling_img_feats
            
            if self.sampling_img_feats:
                self.img_feat_reduce = nn.Linear(self.in_img_channels, self.in_channels[0])
            
            # for each query point, sampling voxel feats & (optional) image feats
            self.num_point_sampling_feat = self.num_level + int(self.sampling_img_feats) * self.num_img_level
            if self.soft_weights:
                soft_in_channel = self.in_channels[0]
                self.point_soft_weights = nn.Sequential(
                    nn.Linear(soft_in_channel, soft_in_channel // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(soft_in_channel // 2, self.num_point_sampling_feat),
                )
            
            point_in_channel = self.in_channels[-1]
            mid_channel = point_in_channel
            out_point_channel = out_point_channel or out_channel
            self.point_occ_mlp = Mlp(point_in_channel, mid_channel, out_point_channel)

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
        
        # build semantic kitti losses and metrics
        if self.semantic_kitti:
            self.class_names = kitti_class_names    
            assert self.out_channel == len(self.class_names)
            
            self.class_weights = torch.from_numpy(
                1 / np.log(semantic_kitti_class_frequencies + 0.001)
            )
            
            binary_class_frequencies = np.array([semantic_kitti_class_frequencies[0], semantic_kitti_class_frequencies[1:].sum()])
            self.binary_class_weights = torch.from_numpy(1 / np.log(binary_class_frequencies))
            
            # build train metric
            self.ssc_metric = SSCMetrics(self.class_names)
            
            self.semkitti_loss_weight_cfg = semkitti_loss_weight_cfg or {}
            
            self.use_ohem_loss = use_ohem_loss
            self.use_sc_ohem_loss = use_sc_ohem_loss
            self.ohem_topk = ohem_topk
        
    def sample_point_feats(self, points, voxel_feats, img_feats=None, points_uv=None):
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
            point_feats_i = []
            for voxel_level_index, voxel_feat in enumerate(voxel_feats):
                sampled_point_feats_level = F.grid_sample(voxel_feat[batch_index].unsqueeze(0), points_i, mode='bilinear', align_corners=False)
                point_feats_i.append(sampled_point_feats_level.squeeze().t())
                
            if self.sampling_img_feats:
                points_img_feats, points_img_mask = feature_sampling([img_feats[batch_index]], points_uv[batch_index])
                # [num_points, num_cam, num_level, C]
                points_img_feats = torch.nan_to_num(points_img_feats)
                # [num_points, num_cam]
                points_img_mask = torch.nan_to_num(points_img_mask)
                
                # sum along camera & num_level
                points_img_feats = points_img_feats * points_img_mask.unsqueeze(-1).unsqueeze(-1)
                points_img_feats = points_img_feats.sum(dim=1).sum(dim=1)
                points_img_feats = self.img_feat_reduce(points_img_feats)
                point_feats_i.append(points_img_feats)
            
            if self.soft_weights:
                point_soft_weights = self.point_soft_weights(point_feats_i[0])
                point_soft_weights = torch.softmax(point_soft_weights, dim=1)
                
                out_point_feats_i = 0
                for feats, weights in zip(point_feats_i, torch.unbind(point_soft_weights, dim=1)):
                    out_point_feats_i += feats * weights.unsqueeze(1)
            else:
                out_point_feats_i = sum(point_feats_i)
            
            point_feats.append(out_point_feats_i)
        
        return point_feats

    def forward_voxel(self, voxel_feats):
        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
                output_occs.append(torch.utils.checkpoint.checkpoint(occ_conv, feats))
            else:
                output_occs.append(occ_conv(feats))
        
        return output_occs
    
    def forward_points(self, points, voxel_feats, img_feats=None, points_uv=None):
        point_feats_list = self.sample_point_feats(points, voxel_feats, img_feats=img_feats, points_uv=points_uv)
        output_points = []
        for points_feats in point_feats_list:
            output_points.append(self.point_occ_mlp(points_feats))

        return output_points        
    
    def forward(self, voxel_feats, points=None, img_metas=None, 
                img_feats=None, points_uv=None, **kwargs):
        
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        # forward voxel 
        if self.supervise_voxel:
            output_voxels = self.forward_voxel(voxel_feats)
        else:
            output_voxels = None
        
        # forward points
        if points is None:
            output_points = None
        
        elif self.supervise_points:
            # sampling voxel features and apply MLP
            output_points = self.forward_points(points=points, voxel_feats=voxel_feats, img_feats=img_feats, points_uv=points_uv)
        
        else:
            # propagate voxel predictions to query points
            output_points = []
            for batch_index, points_i in enumerate(points):
                output_points_i = query_points_from_voxels(output_voxels[-1][batch_index], 
                                        points_i, img_metas[batch_index])

                output_points.append(output_points_i)
        
        res = {
            'output_voxels': output_voxels,
            'output_points': output_points,
        }
        
        return res

    def loss_voxel_single(self, output_voxels, target_voxels, tag):
        # upsample to the target-size
        output_voxels = F.interpolate(output_voxels, size=target_voxels.shape[-3:], mode='trilinear', align_corners=False).contiguous()
        target_voxels = target_voxels.long()
        
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

    def loss_voxel_single_semkitti(self, output_voxels, target_voxels, tag, compute_metric=False, **kwargs):
        # upsample to the target-size
        if output_voxels.shape[-3:] != target_voxels.shape[-3:]:
            output_voxels = F.interpolate(output_voxels, size=target_voxels.shape[-3:], mode='trilinear', align_corners=False).contiguous()
        
        target_voxels = target_voxels.long()
        
        loss_dict = {}
        # voxel cross entropy
        voxel_ce_weight = self.semkitti_loss_weight_cfg.get('voxel_ce', 0.0)
        if voxel_ce_weight > 0:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels)) * \
                voxel_ce_weight
        
        # voxel sem_scal
        voxel_sem_scal = self.semkitti_loss_weight_cfg.get('voxel_sem_scal', 0.0)
        if voxel_sem_scal > 0:
            loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = sem_scal_loss(output_voxels, target_voxels) * voxel_sem_scal
        
        # voxel geo_scal
        voxel_geo_scal = self.semkitti_loss_weight_cfg.get('voxel_geo_scal', 0.0)
        if voxel_geo_scal > 0:
            loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = geo_scal_loss(output_voxels, target_voxels) * voxel_geo_scal
        
        # voxel ce ohem
        voxel_ohem = self.semkitti_loss_weight_cfg.get('voxel_ohem', 0.0)
        if voxel_ohem > 0:
            loss_dict['loss_voxel_sem_ohem_{}'.format(tag)] = OHEM_CE_ssc_loss(output_voxels, target_voxels, 
                            self.class_weights.type_as(output_voxels), top_k=self.ohem_topk) * voxel_ohem
        
        voxel_lovasz = self.semkitti_loss_weight_cfg.get('voxel_lovasz', 0.0)
        if voxel_lovasz > 0:
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = lovasz_softmax(torch.softmax(output_voxels, dim=1), 
                                            target_voxels, ignore=255) * voxel_lovasz
        
        frustum_dist = self.semkitti_loss_weight_cfg.get('frustum_dist', 0.0)
        if frustum_dist > 0:
            loss_dict['loss_voxel_fp_{}'.format(tag)] = compute_frustum_dist_loss(
                output_voxels, kwargs['frustums_masks'], kwargs['frustums_class_dists']) * frustum_dist
        
        # voxel_ce_dice
        voxel_dice = self.semkitti_loss_weight_cfg.get('voxel_dice', 0.0)
        if voxel_dice > 0:
            dice_criterion = SoftDiceLossWithProb()
            occu_probs = torch.softmax(output_voxels, dim=1)[:, 1:].sum(dim=1)
            loss_dict['loss_voxel_dice_{}'.format(tag)] = dice_criterion(occu_probs, target_voxels) * voxel_dice
        
        # LGA weighted ce loss
        voxel_lga_weight = self.semkitti_loss_weight_cfg.get('voxel_lga', 0.0)
        if voxel_lga_weight > 0:
            PAL_criterion = PositionAwareLoss(num_class=self.out_channel)
            loss_dict['loss_voxel_lga_{}'.format(tag)] = PAL_criterion(output_voxels, target_voxels, self.class_weights.type_as(output_voxels)) * \
                voxel_lga_weight
        
        if compute_metric:
            with torch.no_grad():
                output_voxels_tmp = output_voxels.clone().detach()
                target_voxels_tmp = target_voxels.clone().detach()
                
                output_voxels_tmp = torch.argmax(output_voxels_tmp, dim=1)
                mask = target_voxels_tmp != 255
                tp, fp, fn = self.ssc_metric.get_score_completion(output_voxels_tmp, target_voxels_tmp, mask)
                tp_sum, fp_sum, fn_sum = self.ssc_metric.get_score_semantic_and_completion(output_voxels_tmp, target_voxels_tmp, mask)
                sc_iou = tp / (tp + fp + fn)
                ssc_iou = tp_sum / (tp_sum + fp_sum + fn_sum + 1e-5)
                ssc_miou = ssc_iou[1:].mean()
                
                loss_dict['sc_iou_{}'.format(tag)] = sc_iou
                loss_dict['ssc_miou_{}'.format(tag)] = ssc_miou
        
        return loss_dict

    def loss_point_single(self, output_points, target_points, tag):
        target_points = target_points[:, -1].long()
        # print('point labels: ', torch.unique(target_points, return_counts=True))
        
        loss_dict = {}
        
        # compute mIoU
        if not self.semantic_kitti:
            with torch.no_grad():
                output_clses = torch.argmax(output_points, dim=1)
                target_points_np = target_points.cpu().numpy()
                output_clses_np = output_clses.cpu().numpy()
                
                if self.semantic_kitti:
                    unique_label = np.arange(19)
                else:
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

    def loss(self, output_voxels=None, target_voxels=None, 
                output_points=None, target_points=None, 
                img_metas=None, **kwargs):

        if type(target_voxels) is list:
            assert len(target_voxels) >= self.num_level
            target_voxels = target_voxels[:self.num_level]
        else:
            target_voxels = [target_voxels] * self.num_level

        loss_dict = {}
        # 1. compute the losses for voxel-level semantic occupancy
        if self.supervise_voxel:
            for index, output_voxel in enumerate(output_voxels):
                if self.semantic_kitti:
                    loss_dict.update(self.loss_voxel_single_semkitti(output_voxel, target_voxels[index], tag='{}'.format(index), 
                                                compute_metric=(index == 0), **kwargs))
                else:
                    loss_dict.update(self.loss_voxel_single(output_voxel, target_voxels[index], tag='{}'.format(index)))
        
        # 2. compute the losses for point-level semantic segmentation
        if self.supervise_points:
            output_points = torch.cat(output_points, dim=0)
            target_points = torch.cat(target_points, dim=0)
            loss_dict.update(self.loss_point_single(output_points, target_points, tag=''))
        
        return loss_dict
    
def feature_sampling(mlvl_feats, reference_points_2d):
    '''
    mlvl_feats: list of image features [num_cam, num_channel, h', w']
    reference_points_2d: [num_points, num_cam, 3] in [u, v, d]
    '''
    
    num_points, num_cam = reference_points_2d.shape[:2]
    eps = 1e-5
    mask = (reference_points_2d[..., 2] > eps) \
        & (reference_points_2d[..., 0] > -1.0) \
        & (reference_points_2d[..., 0] < 1.0) \
        & (reference_points_2d[..., 1] > -1.0) \
        & (reference_points_2d[..., 1] < 1.0)
    
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        N, C, H, W = feat.size()
        reference_points_cam_lvl = reference_points_2d[..., :2].permute(1, 0, 2).unsqueeze(2)
        sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, mode='bilinear', align_corners=False)
        sampled_feat = sampled_feat.squeeze(-1).permute(2, 0, 1)
        sampled_feats.append(sampled_feat)
    
    # list of sampled features: [num_points, num_cam, C, num_level]
    sampled_feats = torch.stack(sampled_feats, dim=2)
    
    return sampled_feats, mask
        

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
import mcubes
import pdb, os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
import trimesh

@HEADS.register_module()
class BEVOccHead_kitti(nn.Module): #DETRHead
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 use_fpn=False,
                 fpn_channels=None,
                 in_channels=None,
                 out_channels=None,
                 img_channels=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 pc_range=None,
                 upsample_strides=[1, 2, 1, 2],
                 conv_input=None,
                 conv_output=None,
                 use_3d_conv=False,
                 bev_z=None,
                 direct_proj=False,
                 use_semantic=True,
                 ignore_ground=False,
                 ignore_tree=False,
                 iou_loss=False,
                 balance_weight=False,
                 pred_ground=False,
                 ground_class=False,
                 no_multiscale_loss=False,
                 no_decay=False,
                 no_norm=False,
                 large_weight=False,
                 lovesz=False,
                 upsample_loss=False,
                 **kwargs):
        super(BEVOccHead_kitti, self).__init__()
        self.fpn = use_fpn
        self.conv_input = conv_input
        self.conv_output = conv_output
        self.use_3d_conv = use_3d_conv
        self.bev_z = bev_z
        self.direct_proj = direct_proj
        self.use_semantic = use_semantic
        self.ignore_ground=ignore_ground
        self.ignore_tree = ignore_tree
        self.balance_weight = balance_weight
        self.pred_ground = pred_ground
        self.fpn_channels = fpn_channels
        self.no_multiscale_loss = no_multiscale_loss
        self.no_decay = no_decay
        self.no_norm = no_norm
        self.large_weight = large_weight
        self.ground_class = ground_class
        self.lovesz = lovesz
        if self.fpn_channels is not None:
            self.fpn_level = len(self.fpn_channels)
        self.img_channels = img_channels
        self.upsample_strides = upsample_strides
        self.iou_loss = iou_loss
        self.upsample_loss = upsample_loss
        if self.fpn:
            self.transformer = nn.ModuleList()
            self.embed_dims = []
            self.positional_encoding = nn.ModuleList()
            embed_dims_ori = in_channels
            for i in range(self.fpn_level):
                #embed_dims_i = embed_dims_ori // (2 ** (self.fpn_level - 1 - i))
                embed_dims_i = self.fpn_channels[i]
                transformer.embed_dims = embed_dims_i
                transformer.encoder.num_points_in_pillar = 4 * 2**(self.fpn_level - 1 - i)
                #transformer.encoder.transformerlayers.attn_cfgs[1].deformable_attention.num_points = 8 * 2**(self.fpn_level - 1 - i)
                transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = 8 * 2**(self.fpn_level - 1 - i)


                transformer.encoder.transformerlayers.feedforward_channels = 2 * embed_dims_i
                # transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = embed_dims_i
                # transformer.encoder.transformerlayers.attn_cfgs[1].embed_dims = embed_dims_i
                # transformer.encoder.transformerlayers.attn_cfgs[1].deformable_attention.embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.ffn_cfgs.embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.ffn_cfgs.feedforward_channels = 4 * embed_dims_i
                positional_encoding.num_feats = embed_dims_i // 2
                positional_encoding.row_num_embed = bev_h * (2**(self.fpn_level - 1 - i))
                positional_encoding.col_num_embed = bev_w * (2**(self.fpn_level - 1 - i))
                transformer.encoder.num_layers = i + 1

                if self.use_3d_conv:
                    transformer.encoder.num_points_in_pillar = 1
                    transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = 2 **(i + 1)
                    positional_encoding.z_num_embed = bev_z * (2**(self.fpn_level - 1 - i))
                    positional_encoding.num_feats = embed_dims_i // 3
                    transformer.encoder.num_layers = i + 1

                transformer_i = build_transformer(transformer)
                positional_encoding_i = build_positional_encoding(positional_encoding)

                self.transformer.append(transformer_i)
                self.positional_encoding.append(positional_encoding_i)
                self.embed_dims.append(embed_dims_i)


        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.in_channels = in_channels
        self.out_channels = out_channels


        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        deblocks = []
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        if self.use_3d_conv:
            if self.no_norm:
                norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
            else:
                norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)
            upsample_cfg=dict(type='deconv3d', bias=False)
            conv_cfg=dict(type='Conv3d', bias=False)

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)


            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))

            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if self.use_3d_conv:
            if self.use_semantic:
                if self.ignore_ground:
                    if self.ignore_tree:
                        if self.fpn:
                            self.occ = nn.ModuleList()
                            for i in range(self.fpn_level + 1):
                                occ = build_conv_layer(
                                    conv_cfg,
                                    in_channels=out_channels[i*2],
                                    out_channels=12,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                                self.occ.append(occ)
                        else:
                            self.occ = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[-1],
                                out_channels=12,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                    else:
                        if self.fpn:
                            self.occ = nn.ModuleList()
                            for i in range(self.fpn_level + 1):
                                occ = build_conv_layer(
                                    conv_cfg,
                                    in_channels=out_channels[i*2],
                                    out_channels=13,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                                self.occ.append(occ)

                    if self.pred_ground:
                        if self.fpn:
                            self.ground = nn.ModuleList()
                            for i in range(self.fpn_level + 1):
                                ground = build_conv_layer(
                                    conv_cfg,
                                    in_channels=out_channels[i*2],
                                    out_channels=5,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
                                self.ground.append(ground)
                else:
                    if self.fpn:
                        self.occ = nn.ModuleList()
                        for i in range(self.fpn_level + 1):
                            occ = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[i*2],
                                out_channels=21,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                            self.occ.append(occ)
           

        else:

            if self.fpn:
                self.occ = nn.ModuleList()
                for i in range(self.fpn_level + 1):
                    occ = build_conv_layer(
                        conv_cfg,
                        in_channels=out_channels[i*2],
                        out_channels=self.out_channels // 2**(self.fpn_level -i),
                        kernel_size=3,
                        stride=1,
                        padding=1)
                    self.occ.append(occ)

        if self.fpn:
            out_channels = self.fpn_channels
            in_channels = self.img_channels
            self.bev_embedding = nn.ModuleList()
            self.transfer_conv = nn.ModuleList()
            for i in range(self.fpn_level):
                if self.use_3d_conv:
                    self.bev_embedding.append(nn.Embedding(
                        self.bev_h * (2**(self.fpn_level - i - 1)) * self.bev_w * (2**(self.fpn_level - i - 1)) * self.bev_z * (2**(self.fpn_level - i - 1)), self.embed_dims[i]))
                else:
                    self.bev_embedding.append(nn.Embedding(
                        self.bev_h * (2**(self.fpn_level - i - 1)) * self.bev_w * (2**(self.fpn_level - i - 1)), self.embed_dims[i]))


                if self.no_norm:
                    norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
                else:
                    norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)
                conv_cfg=dict(type='Conv2d', bias=True)

                transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=1,
                    stride=1)
                transfer_block = nn.Sequential(transfer_layer,
                        nn.ReLU(inplace=True))
                #build_norm_layer(norm_cfg, out_channels[i])[1],

                self.transfer_conv.append(transfer_block)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        if self.fpn:
            for i in range(self.fpn_level):
                self.transformer[i].init_weights()
        else:
            self.transformer.init_weights()

        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        #object_query_embeds = self.query_embedding.weight.to(dtype)
        if self.fpn:
            bev_embed = []
            for i in range(self.fpn_level):
                bev_queries = self.bev_embedding[i].weight.to(dtype)
                bev_h = self.bev_h * (2**(self.fpn_level - i - 1))
                bev_w = self.bev_w * (2**(self.fpn_level - i - 1))

                if self.use_3d_conv:
                    bev_z = self.bev_z * (2**(self.fpn_level - i - 1))
                    bev_mask = torch.zeros((bs, bev_z, bev_h, bev_w),
                               device=bev_queries.device).to(dtype)
                else:
                    bev_mask = torch.zeros((bs, bev_h, bev_w),
                               device=bev_queries.device).to(dtype)
                bev_pos = self.positional_encoding[i](bev_mask).to(dtype)

                _, _, C, H, W = mlvl_feats[i].shape
                view_features = self.transfer_conv[i](mlvl_feats[i].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)

                if self.use_3d_conv:
                    bev_z = self.bev_z * (2**(self.fpn_level - i - 1))
                    bev_embed_i = self.transformer[i].get_bev_features(
                        [view_features],
                        bev_queries,
                        bev_h,
                        bev_w,
                        grid_length=(self.real_h / bev_h,
                                     self.real_w / bev_w),
                        bev_pos=bev_pos,
                        img_metas=img_metas,
                        fpn_index=i,
                        bev_z=bev_z,
                        use_3d_conv=self.use_3d_conv,
                        direct_proj=(self.direct_proj)
                    )
                bev_embed.append(bev_embed_i)


        if self.fpn:
            bev_embed_reshape = []
            for i in range(self.fpn_level):
                bev_h = self.bev_h * (2**(self.fpn_level - i - 1))
                bev_w = self.bev_w * (2**(self.fpn_level - i - 1))

                if self.use_3d_conv:
                    bev_z = self.bev_z * (2**(self.fpn_level - i - 1))
                    bev_embed_reshape_i = bev_embed[i].reshape(bs, bev_z, bev_h, bev_w, -1).permute(0, 4, 2, 3, 1)
                else:
                    bev_embed_reshape_i = bev_embed[i].reshape(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2)
                bev_embed_reshape.append(bev_embed_reshape_i)

            outputs = []
            result = bev_embed_reshape[-1]
            for i in range(len(self.deblocks)):
                result = self.deblocks[i](result)
                if i == 0 or i == 2 or i == 4 or i == 6:
                    outputs.append(result)
                elif i == 1:
                    result = result + bev_embed_reshape[1]
                elif i == 3:
                    result = result + bev_embed_reshape[0]

        if self.fpn:
            occ_preds = []
            for i in range(len(outputs)):
                occ_pred = self.occ[i](outputs[i])
                if not self.use_3d_conv:
                    occ_pred = occ_pred.permute(0, 2, 3, 1)
                occ_preds.append(occ_pred)

        if self.pred_ground:
            if self.fpn:
                ground_preds = []
                for i in range(len(outputs)):
                    ground_pred = self.ground[i](outputs[i])
                    if not self.use_3d_conv:
                        ground_pred = ground_pred.permute(0, 2, 3, 1)
                    ground_preds.append(ground_pred)
            else:
                ground_preds = self.ground(outputs)

                if self.use_3d_conv:
                    ground_preds = ground_preds#[:, 0]
                else:
                    ground_preds = ground_preds.permute(0, 2, 3, 1)

            outs = {
                'bev_embed': bev_embed,
                'occ_preds': occ_preds,
                'ground_preds': ground_preds,
             #   'speed_preds': speed_preds,
            }
        else:
            outs = {
                'bev_embed': bev_embed,
                'occ_preds': occ_preds,
             #   'speed_preds': speed_preds,
            }

        return outs


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ,
             preds_dicts,
             img_metas=None):
        gt_occ[gt_occ==255] = 20
        semantic_kitti_class_frequencies = np.array(
            [
                5.41773033e09,
                1.57835390e07,
                1.25136000e05,
                1.18809000e05,
                6.46799000e05,
                8.21951000e05,
                2.62978000e05,
                2.83696000e05,
                2.04750000e05,
                6.16887030e07,
                4.50296100e06,
                4.48836500e07,
                2.26992300e06,
                5.68402180e07,
                1.57196520e07,
                1.58442623e08,
                2.06162300e06,
                3.69705220e07,
                1.15198800e06,
                3.34146000e05,
                1e15,  # pseudo class for ignored points
            ]
        )
        pred = preds_dicts['occ_preds']
        class_weights = torch.from_numpy(
            1 / np.log(semantic_kitti_class_frequencies + 0.001)
        ).type_as(pred[0])


        if self.fpn:
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):
                pred = preds_dicts['occ_preds'][i]
                b_, h_, w_, z_ = gt_occ.shape
                up_ratio = 2**(len(preds_dicts['occ_preds'])  - 1 - i)

                if not self.upsample_loss:
                    gt = gt_occ.clone().reshape(b_, h_ // up_ratio, up_ratio, w_ // up_ratio, up_ratio, z_ // up_ratio,
                        up_ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, h_ // up_ratio, w_ // up_ratio, z_ // up_ratio, -1)
                    gt = torch.mode(gt, dim=-1)[0].float()
                else:
                    gt = gt_occ.clone()
                    pred = F.interpolate(pred, size=(h_, w_, z_), mode='trilinear', align_corners=True)

                loss_ce = CE_ssc_loss(pred, gt, class_weights)
                loss_dict['loss_ce_{}'.format(i)] = loss_ce

                loss_sem_scal = sem_scal_loss(pred, gt)
                loss_dict['loss_sem_{}'.format(i)] = loss_sem_scal

                loss_geo_scal = geo_scal_loss(pred, gt)
                loss_dict['loss_geo_{}'.format(i)] = loss_geo_scal

        return loss_dict

from mmcv.runner import BaseModule
class LearnedPositionalEncoding(BaseModule):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super(LearnedPositionalEncoding, self).__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str



def CE_ssc_loss(pred, target, class_weights):
    """
    :param: prediction: the predicted tensor, must be [BS, C, H, W, D]
    """
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=20, reduction="mean"
    )
    loss = criterion(pred, target.long())

    return loss


def KL_sep(p, target):
    """
    KL divergence on nonzeros classes
    """
    nonzeros = target != 0
    nonzero_p = p[nonzeros]
    kl_term = F.kl_div(torch.log(nonzero_p), target[nonzeros], reduction="sum")
    return kl_term


def geo_scal_loss(pred, ssc_target):

    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)

    # Compute empty and nonempty probabilities
    empty_probs = pred[:, 0, :, :, :]
    nonempty_probs = 1 - empty_probs

    # Remove unknown voxels
    mask = ssc_target != 255
    nonempty_target = ssc_target != 0
    nonempty_target = nonempty_target[mask].float()
    nonempty_probs = nonempty_probs[mask]
    empty_probs = empty_probs[mask]

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (
        F.binary_cross_entropy(precision, torch.ones_like(precision))
        + F.binary_cross_entropy(recall, torch.ones_like(recall))
        + F.binary_cross_entropy(spec, torch.ones_like(spec))
    )


def sem_scal_loss(pred, ssc_target):
    # Get softmax probabilities
    pred = F.softmax(pred, dim=1)
    loss = 0
    count = 0
    mask = ssc_target != 255
    n_classes = pred.shape[1]
    for i in range(0, n_classes):

        # Get probability of class i
        p = pred[:, i, :, :, :]

        # Remove unknown voxels
        target_ori = ssc_target
        p = p[mask]
        target = ssc_target[mask]

        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0
        completion_target_ori = torch.ones_like(target_ori).float()
        completion_target_ori[target_ori != i] = 0
        if torch.sum(completion_target) > 0:
            count += 1.0
            nominator = torch.sum(p * completion_target)
            loss_class = 0
            if torch.sum(p) > 0:
                precision = nominator / (torch.sum(p))
                loss_precision = F.binary_cross_entropy(
                    precision, torch.ones_like(precision)
                )
                loss_class += loss_precision
            if torch.sum(completion_target) > 0:
                recall = nominator / (torch.sum(completion_target))
                loss_recall = F.binary_cross_entropy(recall, torch.ones_like(recall))
                loss_class += loss_recall
            if torch.sum(1 - completion_target) > 0:
                specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                    torch.sum(1 - completion_target)
                )
                loss_specificity = F.binary_cross_entropy(
                    specificity, torch.ones_like(specificity)
                )
                loss_class += loss_specificity
            loss += loss_class
    return loss / count


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
class BEVOccHead(nn.Module): #DETRHead
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
                 bbox_coder=None,
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
                 use_semantic=False,
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
                 **kwargs):
        super(BEVOccHead, self).__init__()
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
        
        # img_channels=[512, 1024, 2048]
        # fpn_channels=[96, 192, 384]
        
        if self.fpn:
            self.transformer = nn.ModuleList()
            self.embed_dims = []
            self.positional_encoding = nn.ModuleList()
            embed_dims_ori = in_channels
            for i in range(self.fpn_level):
                #embed_dims_i = embed_dims_ori // (2 ** (self.fpn_level - 1 - i))
                embed_dims_i = self.fpn_channels[i]
                transformer.embed_dims = embed_dims_i
                # [16, 8, 4]
                transformer.encoder.num_points_in_pillar = 4 * 2**(self.fpn_level - 1 - i)
                # [32, 16, 8]
                transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = 8 * 2**(self.fpn_level - 1 - i)
                
                transformer.encoder.transformerlayers.feedforward_channels = 2 * embed_dims_i
                transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.ffn_cfgs.embed_dims = embed_dims_i
                transformer.encoder.transformerlayers.ffn_cfgs.feedforward_channels = 4 * embed_dims_i
                
                positional_encoding.num_feats = embed_dims_i // 2
                # (bev_h, bev_w) * [4, 2, 1]
                positional_encoding.row_num_embed = bev_h * (2 ** (self.fpn_level - 1 - i))
                positional_encoding.col_num_embed = bev_w * (2 ** (self.fpn_level - 1 - i))
                
                # num_layer = [1, 2, 3]
                transformer.encoder.num_layers = i + 1

                if self.use_3d_conv:
                    transformer.encoder.num_points_in_pillar = 1
                    # [2, 4, 8]
                    transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = 2 ** (i + 1)
                    # bev_z_ = 6, reso_z becomes 1/2 for each level
                    positional_encoding.z_num_embed = bev_z * (2 ** (self.fpn_level - 1 - i))
                    positional_encoding.num_feats = embed_dims_i // 3
                    transformer.encoder.num_layers = i + 1

                transformer_i = build_transformer(transformer)
                positional_encoding_i = build_positional_encoding(positional_encoding)

                self.transformer.append(transformer_i)
                self.positional_encoding.append(positional_encoding_i)
                self.embed_dims.append(embed_dims_i)

        else:
            self.transformer = build_transformer(transformer)
            self.embed_dims = self.transformer.embed_dims
            self.positional_encoding = build_positional_encoding(
                positional_encoding)

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
        #super(BEVOccHead, self).__init__(
        #    *args, transformer=transformer, in_channels=in_channels, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        deblocks = []
        upsample_strides = self.upsample_strides

        #out_channels = [self.in_channels // 2, self.in_channels // 2, self.in_channels // 4, self.in_channels //4]
        #in_channels = [self.in_channels, self.in_channels // 2, self.in_channels //2, self.in_channels // 4]

        out_channels = self.conv_output
        in_channels = self.conv_input

        if self.use_3d_conv:
            # no_norm = True, using GroupNorm
            if self.no_norm:
                norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
            else:
                norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01)
            
            upsample_cfg=dict(type='deconv3d', bias=False)
            conv_cfg=dict(type='Conv3d', bias=False)
        else:
            if self.no_norm:
                norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
            else:
                norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)
            
            upsample_cfg=dict(type='deconv', bias=False)
            conv_cfg=dict(type='Conv2d', bias=False)
        
        # conv_input: [384, 256, 192, 128, 96, 64, 32]
        # conv_output: [256, 192, 128, 96, 64, 32, 16]

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
                    # here
                        if self.fpn:
                            # here
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
                        else:
                            self.occ = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[-1],
                                out_channels=13,
                                kernel_size=3,
                                stride=1,
                                padding=1)

                    if self.pred_ground:
                        if self.fpn:
                            # here
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
                            self.ground = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[-1],
                                out_channels=5,
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
                                out_channels=17,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                            self.occ.append(occ)
                    else:
                        self.occ = build_conv_layer(
                            conv_cfg,
                            in_channels=out_channels[-1],
                            out_channels=17,
                            kernel_size=3,
                            stride=1,
                            padding=1)
            else:
                if self.fpn:
                    self.occ = nn.ModuleList()
                    if self.ground_class:
                        for i in range(self.fpn_level + 1):
                            occ = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[i*2],
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                            self.occ.append(occ)
                    else:
                        for i in range(self.fpn_level + 1):
                            occ = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[i*2],
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                            self.occ.append(occ)
                else:
                    self.occ = build_conv_layer(
                        conv_cfg,
                        in_channels=out_channels[-1],
                        out_channels=1,
                        kernel_size=3,
                        stride=1,
                        padding=1)

                if self.pred_ground:
                    if self.fpn:
                        self.ground = nn.ModuleList()
                        for i in range(self.fpn_level + 1):
                            ground = build_conv_layer(
                                conv_cfg,
                                in_channels=out_channels[i*2],
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1)
                            self.ground.append(ground)
                    else:
                        self.ground = build_conv_layer(
                            conv_cfg,
                            in_channels=out_channels[-1],
                            out_channels=1,
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
                        out_channels=self.out_channels // 2**(self.fpn_level -i),
                        kernel_size=3,
                        stride=1,
                        padding=1)
                    self.occ.append(occ)
            else:
                self.occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[-1],
                    out_channels=self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1)

        # self.speed = build_conv_layer(
        #             conv_cfg,
        #             in_channels=out_channels[-1],
        #             out_channels=self.out_channels,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1)


        #self.query_embedding = nn.Embedding(self.num_query,
        #                                    self.embed_dims * 2)

        if self.fpn:
            out_channels = self.fpn_channels
            in_channels = self.img_channels
            self.bev_embedding = nn.ModuleList()
            self.transfer_conv = nn.ModuleList()
            for i in range(self.fpn_level):
                if self.use_3d_conv:
                    # num_query = [bev_h * bev_w * bev_z]
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
        else:
            out_channels = self.fpn_channels
            in_channels = self.img_channels
            conv_cfg=dict(type='Conv2d', bias=True)
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                        nn.ReLU(inplace=True))
            self.transfer_conv = transfer_block

            if self.use_3d_conv:
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w * self.bev_z, self.embed_dims)
            else:
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)



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
    def forward(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

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

                if prev_bev is None:
                    prev_bev_i = None
                else:
                    prev_bev_i = prev_bev[i]

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
                        prev_bev=prev_bev_i,
                        fpn_index=i,
                        bev_z=bev_z,
                        use_3d_conv=self.use_3d_conv,
                        direct_proj=(self.direct_proj)
                    )
                else:
                    bev_embed_i = self.transformer[i].get_bev_features(
                        [view_features],
                        bev_queries,
                        bev_h,
                        bev_w,
                        grid_length=(self.real_h / bev_h,
                                     self.real_w / bev_w),
                        bev_pos=bev_pos,
                        img_metas=img_metas,
                        prev_bev=prev_bev_i,
                        fpn_index=i
                    )
                bev_embed.append(bev_embed_i)
        else:
            _, _, C, H, W = mlvl_feats[0].shape
            view_features = self.transfer_conv(mlvl_feats[0].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)
            mlvl_feats = [view_features]

            if self.use_3d_conv:
                bev_queries = self.bev_embedding.weight.to(dtype)
                bev_mask = torch.zeros((bs, self.bev_z, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
                bev_pos = self.positional_encoding(bev_mask).to(dtype)

                bev_embed = self.transformer.get_bev_features(
                    mlvl_feats,
                    bev_queries,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                 self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    img_metas=img_metas,
                    prev_bev=prev_bev,
                    bev_z=self.bev_z,
                    use_3d_conv=self.use_3d_conv,
                    direct_proj=self.direct_proj
                )
            else:
                bev_queries = self.bev_embedding.weight.to(dtype)
                bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                                   device=bev_queries.device).to(dtype)
                bev_pos = self.positional_encoding(bev_mask).to(dtype)

                bev_embed = self.transformer.get_bev_features(
                    mlvl_feats,
                    bev_queries,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                 self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    img_metas=img_metas,
                    prev_bev=prev_bev,
                    direct_proj=self.direct_proj
                )

        if only_bev:
            return bev_embed

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
        else:
            if self.use_3d_conv:
                outputs = bev_embed.reshape(bev_embed.shape[0], self.bev_z, self.bev_h, self.bev_w, -1).permute(0, 4, 2, 3, 1)
                for i in range(len(self.deblocks)):
                    outputs = self.deblocks[i](outputs)
            else:
                outputs = bev_embed.reshape(bev_embed.shape[0], self.bev_h, self.bev_w, -1).permute(0, 3, 1, 2)
                for i in range(len(self.deblocks)):
                    outputs = self.deblocks[i](outputs)

        if self.fpn:
            occ_preds = []
            for i in range(len(outputs)):
                occ_pred = self.occ[i](outputs[i])
                if not self.use_3d_conv:
                    occ_pred = occ_pred.permute(0, 2, 3, 1)
                occ_preds.append(occ_pred)
        else:
            occ_preds = self.occ(outputs)

            if self.use_3d_conv:
                occ_preds = occ_preds#[:, 0]
            else:
                occ_preds = occ_preds.permute(0, 2, 3, 1)

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


    def dice_loss(self, prob, gt, ep=1e-7):
        loss_iou = 1 - (2 * (prob * gt).sum() + ep) / ((prob * prob).sum() + (gt * gt).sum() +ep)
        return loss_iou

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ,
             preds_dicts,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        '''
        self.alpha = 0.25
        self.gamma = 2.0
        input = preds_dicts['occ_preds']
        target = gt_occ.float()
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        weight_masks = []
        bs = gt_occ.shape[0]
        for i in range(bs):
            weight_mask = gt_occ[i] > 10
            negative_mask = gt_occ[i] == 0
            positive_mask = gt_occ[i] == 1
            ignore_mask = gt_occ[i] == -1

            #unif = torch.ones(negative_mask.sum().long()).cuda()
            #drop_indices = unif.multinomial(positive_mask.sum().long(), replacement = False)

            mask = negative_mask[negative_mask]
            sample_ratio = (negative_mask.sum() / (positive_mask.sum().float())).long()
            random_start = torch.multinomial(torch.ones(sample_ratio), 1)
            mask[random_start::sample_ratio] = False
            weight_mask[gt_occ[i] == 0] = mask
            weight_mask = 1 - weight_mask.float()
            weight_mask[ignore_mask] = 0.0
            #print(weight_mask.sum(), positive_mask.sum(), ignore_mask.sum())
            weight_masks.append(weight_mask)

        weight_masks = torch.stack(weight_masks, dim=0)
        pred = preds_dicts['occ_preds'][weight_masks == 1]
        gt = gt_occ.float()[weight_masks == 1]
        #loss_occ =  F.binary_cross_entropy_with_logits(preds_dicts['occ_preds'], gt_occ.float(), weight=weight_masks) #* focal_weight
        '''

        if not self.use_semantic:
            gt_occ_ori = gt_occ.clone()

            if self.pred_ground:
                gt_ground = ((gt_occ >= 11) * (gt_occ <= 14)).float()
                #gt_ground = gt_occ == -1
                #gt_occ[gt_occ == -1] = 0

            if self.ignore_ground:
                gt_occ[(gt_occ >= 11) * (gt_occ <= 14)] = 0
            elif self.ground_class:
                gt_occ[(gt_occ >= 11) * (gt_occ <= 14)] = 2
                class_weights = torch.from_numpy(np.array([0.1, 1, 1])).to(gt_occ.device).type(torch.float)
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights, ignore_index=255, reduction="mean"
                )
            else:
                gt_occ[(gt_occ >= 11) * (gt_occ <= 14)] = 1

            if self.ignore_tree:
                gt_occ[gt_occ == 16] = 0

            gt_occ[(gt_occ >= 1) * (gt_occ <= 10)] = 1
            gt_occ[gt_occ >= 15] = 1

            if not self.fpn:
                if len(preds_dicts['occ_preds'].shape) == 5:
                    pred = preds_dicts['occ_preds'][:,0]
                else:
                    pred = preds_dicts['occ_preds']
            gt = gt_occ.float()
            #pred = preds_dicts['occ_preds'][gt_occ > -1]
            #gt = gt_occ.float()[gt_occ > -1]

            if self.fpn:
                loss_dict = {}
                loss_iou = 0
                for i in range(len(preds_dicts['occ_preds'])):
                    if self.no_multiscale_loss and i != len(preds_dicts['occ_preds']) - 1:
                        continue

                    if self.use_3d_conv and not self.ground_class:
                        pred = preds_dicts['occ_preds'][i][:, 0]
                    else:
                        pred = preds_dicts['occ_preds'][i]
                    b_, h_, w_, z_ = gt_occ.shape
                    up_ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)
                    gt = gt_occ.clone().reshape(b_, h_ // up_ratio, up_ratio, w_ // up_ratio, up_ratio, z_ // up_ratio,
                          up_ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, h_ // up_ratio, w_ // up_ratio, z_ // up_ratio, -1)
                    gt = torch.mode(gt, dim=-1)[0].float()

                    if self.ground_class:
                        loss_occ_i = criterion(pred, gt.long()) * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))
                        loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

                    else:
                        if self.no_decay:
                            loss_occ_i =  F.binary_cross_entropy_with_logits(pred, gt, pos_weight=(torch.tensor(10).to(pred.device).float()))
                        else:
                            loss_occ_i =  F.binary_cross_entropy_with_logits(pred, gt, pos_weight=(torch.tensor(10).to(pred.device).float())) * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i)) #* focal_weight
                        loss_dict['loss_occ_{}'.format(i)] = loss_occ_i


                        if self.iou_loss:
                            prob = torch.sigmoid(pred)
                            loss_iou = loss_iou + self.dice_loss(prob, gt)

                        if self.pred_ground:
                            if self.use_3d_conv:
                                pred = preds_dicts['ground_preds'][i][:, 0]
                            else:
                                pred = preds_dicts['ground_preds'][i]
                            b_, h_, w_, z_ = gt_ground.shape
                            up_ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)
                            gt = gt_ground.clone().reshape(b_, h_ // up_ratio, up_ratio, w_ // up_ratio, up_ratio, z_ // up_ratio,
                                  up_ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, h_ // up_ratio, w_ // up_ratio, z_ // up_ratio, -1)
                            gt = torch.mode(gt, dim=-1)[0].float()

                            if self.no_decay:
                                loss_ground_i =  0.1 * F.binary_cross_entropy_with_logits(pred, gt, pos_weight=(torch.tensor(10).to(pred.device).float()))
                            else:
                                loss_ground_i =  0.1 * F.binary_cross_entropy_with_logits(pred, gt, pos_weight=(torch.tensor(10).to(pred.device).float())) * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i)) #* focal_weight

                            loss_dict['loss_ground_{}'.format(i)] = loss_ground_i


            else:
                loss_occ =  F.binary_cross_entropy_with_logits(pred, gt, pos_weight=(torch.tensor(10).to(pred.device).float())) #* focal_weight

                if self.iou_loss:
                    prob = torch.sigmoid(pred)
                    loss_iou = self.dice_loss(prob, gt)

                if self.pred_ground:
                    if len(preds_dicts['ground_preds'].shape) == 5:
                        pred_ground = preds_dicts['ground_preds'][:,0]
                    else:
                        pred_ground = preds_dicts['ground_preds']

                    loss_ground = 0.1 *  F.binary_cross_entropy_with_logits(pred_ground, gt_ground, pos_weight=(torch.tensor(10).to(pred.device).float())) #* focal_weight



                if self.iou_loss:
                    loss_iou = loss_iou * 3
                    loss_dict = {'loss_occ': loss_occ, 'loss_iou': loss_iou}
                else:
                    loss_dict = {'loss_occ': loss_occ}

                if self.pred_ground:
                    loss_dict = {'loss_occ': loss_occ, 'loss_ground': loss_ground}


        else:
            pred = preds_dicts['occ_preds']
            gt_occ_ori = gt_occ.clone()

            if self.pred_ground:
                '''
                background and 4 ground classes:
                11: 'driveable_surface'
                12: 'other_flat'
                13: 'sidewalk'
                14: 'terrain'
                '''
                class_weights_ground = torch.from_numpy(np.array([0.1, 1, 1, 1, 1])).to(gt_occ.device).type(torch.float)
                gt_ground = gt_occ.clone()
                gt_ground[gt_occ == 11] = 1
                gt_ground[gt_occ == 12] = 2
                gt_ground[gt_occ == 13] = 3
                gt_ground[gt_occ == 14] = 4
                
                # all other classes as the background
                gt_ground[(gt_occ >= 1) * (gt_occ <= 10)] = 0
                gt_ground[gt_occ >= 15] = 0
                criterion_ground = nn.CrossEntropyLoss(
                    weight=class_weights_ground, ignore_index=255, reduction="mean"
                )

            if self.ignore_ground:
                if self.ignore_tree:
                    class_weights = np.array([1.01989629e+11, 9.11177700e+06, 5.25340000e+05, 4.72792200e+06, 4.28557810e+07, 4.52313600e+06, 6.58822000e+05, 5.13073500e+06, \
                                          1.91219300e+06, 5.62978800e+06, 1.71617210e+07, 4.24780924e+08]).astype(np.double)
                    gt_occ[gt_occ == 11] = 0
                    gt_occ[gt_occ == 12] = 0
                    gt_occ[gt_occ == 13] = 0
                    gt_occ[gt_occ == 14] = 0
                    gt_occ[gt_occ == 15] = 11
                    gt_occ[gt_occ == 16] = 0
                else:
                    # ???????
                    class_weights = np.array([1.01989629e+11, 9.11177700e+06, 5.25340000e+05, 4.72792200e+06, 4.28557810e+07, 4.52313600e+06, 6.58822000e+05, 5.13073500e+06, \
                                          1.91219300e+06, 5.62978800e+06, 1.71617210e+07, 4.24780924e+08, 5.44355938e+08]).astype(np.double)
                    gt_occ[gt_occ == 11] = 0
                    gt_occ[gt_occ == 12] = 0
                    gt_occ[gt_occ == 13] = 0
                    gt_occ[gt_occ == 14] = 0
                    gt_occ[gt_occ == 15] = 11
                    gt_occ[gt_occ == 16] = 12

            else:
                class_weights = np.array([1.01989629e+11, 9.11177700e+06, 5.25340000e+05, 4.72792200e+06, 4.28557810e+07, 4.52313600e+06, 6.58822000e+05, 5.13073500e+06, \
                                          1.91219300e+06, 5.62978800e+06, 1.71617210e+07, 5.33409882e+08, 2.25120310e+07, 1.65556889e+08, 2.20547765e+08, 4.24780924e+08, 5.44355938e+08]).astype(np.double)
            #class_weights = 1.0 / ((class_weights)**(1/3.0))
            #class_weights = class_weights / class_weights.sum() * 10
            #class_weights = 1.0 / np.log(class_weights)
            class_weights[:] = 1
            class_weights[0] = 0.1
            if self.balance_weight:
                class_weights[:] = 2
                class_weights[0] = 0.1
                class_weights[-1] = 0.5
                class_weights[-2] = 0.5

            if self.large_weight:
                class_weights[:] = 2
                class_weights[0] = 0.05
                class_weights[-1] = 0.5
                class_weights[-2] = 0.5

            if not self.ignore_ground:
                 class_weights[:] = 1

            if self.ignore_tree:
                class_weights[:] = 3
                class_weights[0] = 0.1
                class_weights[-1] = 1

            class_weights = torch.from_numpy(class_weights).to(gt_occ.device).type(torch.float)
            criterion = nn.CrossEntropyLoss(
                weight=class_weights, ignore_index=255, reduction="mean"
            )

            if self.fpn:
                loss_dict = {}
                loss_iou = 0

                for i in range(len(preds_dicts['occ_preds'])):

                    if self.no_multiscale_loss and i != len(preds_dicts['occ_preds']) - 1:
                        continue

                    pred = preds_dicts['occ_preds'][i]
                    b_, h_, w_, z_ = gt_occ.shape
                    # upsample ratios = [8, 4, 2, 1]
                    up_ratio = 2**(len(preds_dicts['occ_preds'])  - 1 - i)
                    gt = gt_occ.clone().reshape(b_, h_ // up_ratio, up_ratio, w_ // up_ratio, up_ratio, z_ // up_ratio,
                          up_ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, h_ // up_ratio, w_ // up_ratio, z_ // up_ratio, -1)
                    gt = torch.mode(gt, dim=-1)[0].float()

                    if self.no_decay:
                        loss_occ_i = criterion(pred, gt.long())
                    else:
                        loss_occ_i = criterion(pred, gt.long()) * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))

                    if self.lovesz:
                        loss_occ_i = (criterion(pred, gt.long()) + lovasz_softmax(torch.nn.functional.softmax(pred, dim=1), gt, ignore=255))* ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))

                    loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

                    if self.iou_loss:
                        loss_iou_i = 0
                        prob = torch.softmax(pred, dim=1)
                        for j in range(1, pred.shape[1]):
                            gt_j = (gt == j).float()
                            loss_iou_i = loss_iou_i + self.dice_loss(prob[:, j], gt_j)
                        loss_iou_i = loss_iou_i / (pred.shape[1] - 1)
                        loss_iou = loss_iou + loss_iou_i

                    if self.pred_ground:
                        pred = preds_dicts['ground_preds'][i]
                        b_, h_, w_, z_ = gt_ground.shape
                        up_ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)
                        gt = gt_ground.clone().reshape(b_, h_ // up_ratio, up_ratio, w_ // up_ratio, up_ratio, z_ // up_ratio,
                              up_ratio).permute(0, 1, 3, 5, 2, 4, 6).reshape(b_, h_ // up_ratio, w_ // up_ratio, z_ // up_ratio, -1)
                        gt = torch.mode(gt, dim=-1)[0].float()

                        if self.no_decay:
                            loss_ground_i = 0.1 * criterion_ground(pred, gt.long())
                        else:
                            loss_ground_i = 0.1 * criterion_ground(pred, gt.long()) * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))
                        loss_dict['loss_ground_{}'.format(i)] = loss_ground_i



            else:
                loss_occ = criterion(pred, gt_occ.long())

                if self.iou_loss:
                    loss_iou = 0
                    prob = torch.softmax(pred, dim=1)
                    for j in range(1, pred.shape[1]):
                        gt_j = (gt_occ == j).float()
                        loss_iou = loss_iou + self.dice_loss(prob[:, j], gt_j)
                    loss_iou = loss_iou / (pred.shape[1] - 1)

                if self.pred_ground:
                    pred_ground = preds_dicts['ground_preds']
                    loss_ground = 0.1 * criterion_ground(pred_ground, gt_ground.long())


                if self.iou_loss:
                    loss_iou = loss_iou * 3
                    loss_dict = {'loss_occ': loss_occ, 'loss_iou': loss_iou}
                else:
                    loss_dict = {'loss_occ': loss_occ}

                if self.pred_ground:
                    loss_dict = {'loss_occ': loss_occ, 'loss_ground': loss_ground}

     

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

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)

def jaccard_loss(probas, labels,ignore=None, smooth = 100, bk_class = None):
    """
    Something wrong with this loss
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)


    true_1_hot = torch.eye(vprobas.shape[1])[vlabels]

    if bk_class:
        one_hot_assignment = torch.ones_like(vlabels)
        one_hot_assignment[vlabels == bk_class] = 0
        one_hot_assignment = one_hot_assignment.float().unsqueeze(1)
        true_1_hot = true_1_hot*one_hot_assignment

    true_1_hot = true_1_hot.to(vprobas.device)
    intersection = torch.sum(vprobas * true_1_hot)
    cardinality = torch.sum(vprobas + true_1_hot)
    loss = (intersection + smooth / (cardinality - intersection + smooth)).mean()
    return (1-loss)*smooth

def hinge_jaccard_loss(probas, labels,ignore=None, classes = 'present', hinge = 0.1, smooth =100):
    """
    Multi-class Hinge Jaccard loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      ignore: void class labels
    """
    vprobas, vlabels = flatten_probas(probas, labels, ignore)
    C = vprobas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        if c in vlabels:
            c_sample_ind = vlabels == c
            cprobas = vprobas[c_sample_ind,:]
            non_c_ind =np.array([a for a in class_to_sum if a != c])
            class_pred = cprobas[:,c]
            max_non_class_pred = torch.max(cprobas[:,non_c_ind],dim = 1)[0]
            TP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max = hinge)+1.) + smooth
            FN = torch.sum(torch.clamp(max_non_class_pred - class_pred, min = -hinge)+hinge)

            if (~c_sample_ind).sum() == 0:
                FP = 0
            else:
                nonc_probas = vprobas[~c_sample_ind,:]
                class_pred = nonc_probas[:,c]
                max_non_class_pred = torch.max(nonc_probas[:,non_c_ind],dim = 1)[0]
                FP = torch.sum(torch.clamp(class_pred - max_non_class_pred, max = hinge)+1.)

            losses.append(1 - TP/(TP+FP+FN))

    if len(losses) == 0: return 0
    return mean(losses)

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

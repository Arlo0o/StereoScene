# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn
import torch.utils.checkpoint as cp
import pdb

from mmdet.models import NECKS

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

@NECKS.register_module()
class SimpleFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
        in_channels=768,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        out_channels=256,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        init_cfg=None, 
    ):
        super(SimpleFPN, self).__init__(init_cfg=init_cfg)
        self.fp16_enabled = False

        self.in_channels = in_channels
        self.scale_factors = scale_factors

        # using self-defined LayerNorm
        self.norm_cfg = None

        # generating multi-scale features
        self.stages = []
        dim = self.in_channels
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    LayerNorm(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            layers.extend([
                build_conv_layer(dict(type='Conv2d', bias=False, kernel_size=1),
                    in_channels=out_dim, out_channels=out_channels),
                LayerNorm(out_channels),
                build_conv_layer(dict(type='Conv2d', bias=False, kernel_size=3, padding=1),
                    in_channels=out_channels, out_channels=out_channels),
                LayerNorm(out_channels),
            ])
            layers = nn.Sequential(*layers)
            self.add_module(f"simfp_{idx}", layers)
            self.stages.append(layers)
        
        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """

        x = x[0]
        out_h, out_w = x.shape[-2:]
        out = []

        for stage in self.stages:
            y = stage(x)
            out.append(y)
        
        return out

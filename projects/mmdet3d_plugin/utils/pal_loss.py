import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
import numpy as np

import pdb

class PositionAwareLoss(nn.Module):
    def __init__(self, num_class=20, alpha=1.0, beta=1.0):
        super(PositionAwareLoss, self).__init__()
        
        # loss_weight = alpha + beta * LGA
        self.alpha = alpha
        self.beta = beta
        
        # number of classes
        self.num_class = num_class
    
    def get_lga(self, targets):
        res = 0
        for index in range(self.num_class):
            cls_binary = torch.zeros_like(targets)
            cls_binary[targets == index] = 1
            # tuple of the gradients along (x, y, z)
            cls_gradients = torch.gradient(cls_binary, dim=(1, 2, 3))
            cls_gradients = torch.stack(cls_gradients, dim=1)
            # [X, Y, Z] and only cls-related areas can have positive values
            cls_gradients = torch.norm(cls_gradients, p=1, dim=1)
            res += cls_gradients
        
        return res

    def forward(self, predictions, targets, class_weights):
        criterion = nn.CrossEntropyLoss(weight=class_weights, 
                        ignore_index=255, reduction="none")
        targets = targets.long()
        
        loss = criterion(predictions, targets)
        
        # non-ignored areas
        flatten_targets = targets.flatten()
        valid_mask = flatten_targets != 255
        norm_weights = class_weights[flatten_targets[valid_mask]]
        
        # for debug, compute the original weighted CE loss
        # loss_tmp = loss.flatten()[valid_mask].sum() / norm_weights.sum()
        
        # with LGA scaling
        lga = self.get_lga(targets)
        lga_weights = self.alpha + self.beta * lga
        loss = loss * lga_weights
        
        loss = loss.flatten()[valid_mask].sum() / norm_weights.sum()
        
        return loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import pdb

## Soft Dice Loss for binary segmentation
##
# v1: pytorch autograd
class SoftDiceLossV1(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss
    
class SoftDiceLossWithProb(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossWithProb, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, probs, labels, ignore_index=255):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        valid_mask = (labels != ignore_index)
        
        probs = probs[valid_mask]
        labels = labels[valid_mask]
        # convert to [0, 1]
        labels = (labels > 0).float()
        
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        
        return loss

##
# v2: self-derived grad formula
class SoftDiceLossV2(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    '''
    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV2, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        logits = logits.view(1, -1)
        labels = labels.view(1, -1)
        loss = SoftDiceLossV2Func.apply(logits, labels, self.p, self.smooth)
        return loss

class SoftDiceLossV2Func(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, labels, p, smooth):
        '''
        inputs:
            logits: (N, L)
            labels: (N, L)
        outpus:
            loss: (N,)
        '''
        #  logits = logits.float()

        probs = torch.sigmoid(logits)
        numer = 2 * (probs * labels).sum(dim=1) + smooth
        denor = (probs.pow(p) + labels.pow(p)).sum(dim=1) + smooth
        loss = 1. - numer / denor

        ctx.vars = probs, labels, numer, denor, p, smooth
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of soft-dice loss
        '''
        probs, labels, numer, denor, p, smooth = ctx.vars

        numer, denor = numer.view(-1, 1), denor.view(-1, 1)

        term1 = (1. - probs).mul_(2).mul_(labels).mul_(probs).div_(denor)

        term2 = probs.pow(p).mul_(1. - probs).mul_(numer).mul_(p).div_(denor.pow_(2))

        grads = term2.sub_(term1).mul_(grad_output)

        return grads, None, None, None
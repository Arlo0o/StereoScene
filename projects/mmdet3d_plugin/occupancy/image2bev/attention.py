

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import gc
import time
from einops import rearrange, reduce
from torch import nn, einsum


class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Disp(nn.Module):
    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x


        
class attention(nn.Module):
 
    def __init__(self, in_dim):
        super(attention, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim  , kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))   

        self.softmax = nn.Softmax(dim=-1)   

    def forward(self, q, kv):
 
        x=kv
        m_batchsize, C, D, height, width = x.size()

        confidence  = F.softmax(q, dim=2)  ## B 1 D H W
        confidence = torch.max(confidence, dim=2)[0] ## B 1 H W
        confidence =confidence.view(m_batchsize, -1, width * height)## B 1 HW

     
        proj_query = self.query_conv(q ).view(m_batchsize, -1, width * height).permute(0, 2, 1)
 
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
 
        energy = torch.bmm(proj_query, proj_key)
 
        attention = self.softmax(energy)

        attention = confidence*attention 

 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
   
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))   
     
        out = out.view(m_batchsize, C, D, height, width)

        out = self.gamma * out + x
        return out



class CA3D(nn.Module):
    def __init__(self, channel):
        super(CA3D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
            )
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channel, channel//8, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
            nn.Conv3d(channel//8, channel, kernel_size=1, stride=1, dilation=1, padding=0),
            nn.GELU(),
  
        )

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, dilation=1, padding=1, groups=1),
            nn.GELU(),
            nn.GroupNorm(1, channel),
        )
    def forward(self, x):
        data = self.conv1(x)
        pool = self.avg_pool(data)
        squeeze = self.conv2(pool)
        weight = self.sigmoid(squeeze)
        out = weight*data
        out = self.conv(out)
        return out


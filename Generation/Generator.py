# encoding=utf-8

import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

# add for shape-preserving Loss
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
cudnn.benchnark=True
from Generation.modules import *
from torch.nn import AvgPool2d, Conv1d, Conv2d, Embedding, LeakyReLU, Module

neg = 0.01
neg_2 = 0.2
class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim, use_eql=False):
        super().__init__()
        Conv = EqualConv1d if use_eql else nn.Conv1d

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = Conv(style_dim, in_channel * 2, 1)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EdgeBlock(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, attn=True):
        super(EdgeBlock, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv_w = nn.Sequential(
            nn.Conv2d(Fin, Fout//2, 1),
            nn.BatchNorm2d(Fout//2),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv2d(Fout//2, Fout, 1),
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_x = nn.Sequential(
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(neg, inplace=True)
        )

        self.conv_out = nn.Conv2d(Fout, Fout, [1, k], [1, 1])  # Fin, Fout, kernel_size, stride



    def forward(self, x):
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        x = self.conv_x(x)  # Bx2CxNxk
        x = x * w  # Bx2CxNxk

        x = self.conv_out(x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts
        self.np = opts.np
        self.nk = opts.nk//2
        self.nz = opts.nz
        softmax = opts.softmax
        self.off = opts.off
        self.use_attn = opts.attn
        self.use_head = opts.use_head

        Conv = EqualConv1d if self.opts.eql else nn.Conv1d
        Linear = EqualLinear if self.opts.eql else nn.Linear

        dim = 128
        self.head = nn.Sequential(
            Conv(3 + self.nz, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Conv(dim, dim, 1),
            #nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
        )

        if self.use_attn:
            self.attn = Attention(dim + 512)

        self.global_conv = nn.Sequential(
            Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True),
            Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(neg, inplace=True),
        )


        self.tail = nn.Sequential(
            Conv1d(512+dim, 256, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(256, 64, 1),
            nn.LeakyReLU(neg, inplace=True),
            Conv1d(64, 3, 1),
            nn.Tanh()
        )

        if self.use_head:
            self.pc_head = nn.Sequential(
                Conv(3, dim // 2, 1),
                nn.LeakyReLU(inplace=True),
                Conv(dim // 2, dim, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.EdgeConv1 = EdgeBlock(dim, dim, self.nk)
            self.adain1 = AdaptivePointNorm(dim, dim)
            self.EdgeConv2 = EdgeBlock(dim, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)
        else:
            self.EdgeConv1 = EdgeBlock(3, 64, self.nk)
            self.adain1 = AdaptivePointNorm(64, dim)
            self.EdgeConv2 = EdgeBlock(64, dim, self.nk)
            self.adain2 = AdaptivePointNorm(dim, dim)

        self.lrelu1 = nn.LeakyReLU(neg_2)
        self.lrelu2 = nn.LeakyReLU(neg_2)



    def forward(self, x, z):

        B,N,_ = x.size()
        if self.opts.z_norm:
            z = z / (z.norm(p=2, dim=-1, keepdim=True)+1e-8)

        style = torch.cat([x, z], dim=-1)
        style = style.transpose(2, 1).contiguous()
        style = self.head(style)  # B,C,N

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)


        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_o = self.tail(feat_cat)                   # Bx3x256

        x1_p = pc + x1_o if self.off else x1_o

        return x1_p

    def interpolate(self, x, z1, z2, selection, alpha, use_latent = False):

        if not use_latent:

            ## interpolation
            z = z1
            z[:, selection == 1] = z1[:, selection == 1] * (1 - alpha) + z2[:, selection == 1] * (alpha)

            B, N, _ = x.size()
            if self.opts.z_norm:
                z = z / (z.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style = torch.cat([x, z], dim=-1)
            style = style.transpose(2, 1).contiguous()
            style = self.head(style)  # B,C,N

        else:
            # interplolation
            B, N, _ = x.size()
            if self.opts.z_norm:
                z1 = z1 / (z1.norm(p=2, dim=-1, keepdim=True) + 1e-8)
                z2 = z2 / (z2.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            style_1 = torch.cat([x, z1], dim=-1)
            style_1 = style_1.transpose(2, 1).contiguous()
            style_1 = self.head(style_1)  # B,C,N

            style_2 = torch.cat([x, z2], dim=-1)
            style_2 = style_2.transpose(2, 1).contiguous()
            style_2 = self.head(style_2)  # B,C,N

            style = style_1
            style[:, :, selection == 1] = style_1[:, :, selection == 1] * (1 - alpha) + style_2[:, :, selection == 1] * alpha

        pc = x.transpose(2, 1).contiguous()
        if self.use_head:
            pc = self.pc_head(pc)

        x1 = self.EdgeConv1(pc)
        x1 = self.lrelu1(x1)
        x1 = self.adain1(x1, style)

        x2 = self.EdgeConv2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.adain2(x2, style)

        feat_global = torch.max(x2, 2, keepdim=True)[0]
        feat_global = feat_global.view(B, -1)
        feat_global = self.global_conv(feat_global)
        feat_global = feat_global.view(B, -1, 1)
        feat_global = feat_global.repeat(1, 1, N)

        feat_cat = torch.cat((feat_global, x2), dim=1)

        if self.use_attn:
            feat_cat = self.attn(feat_cat)

        x1_o = self.tail(feat_cat)  # Bx3x256

        x1_p = pc + x1_o if self.off else x1_o

        return x1_p




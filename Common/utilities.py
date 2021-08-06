#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: utilities.py
@time: 2019/09/27
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from Common import ops
from typing import List, Tuple

class DenseModule1D(nn.Module):
    def __init__(self, in_dim=64, levels=3, growth_rate=64):
        super(DenseModule1D, self).__init__()
        self.dense_modules = nn.ModuleList()
        self.input_channel = in_dim
        for i in range(levels):
            if i == levels-1:
                growth_rate = in_dim
            self.dense_modules.append(
                nn.Sequential(nn.Conv1d(self.input_channel, growth_rate, kernel_size=1, bias=True),
                              nn.BatchNorm1d(growth_rate),
                              nn.LeakyReLU(negative_slope=0.2))
            )
            self.input_channel = self.input_channel + growth_rate

    def forward(self, x):
        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](x)
            x = torch.cat([x,y],1)

        return y


class DenseModule2D(nn.Module):
    def __init__(self, in_dim=64, levels=3, growth_rate=64, mlps=[]):
        super(DenseModule2D, self).__init__()
        self.dense_modules = nn.ModuleList()
        self.input_channel = in_dim
        for i in range(1, len(mlps)):
            growth_rate = mlps[i]
            self.dense_modules.append(
                nn.Sequential(nn.Conv2d(self.input_channel, growth_rate, kernel_size=1, bias=True),
                              nn.BatchNorm2d(growth_rate),
                              nn.LeakyReLU(negative_slope=0.2))
            )
            self.input_channel = self.input_channel + growth_rate

    def forward(self, x):

        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](x)
            x = torch.cat([x,y],1)

        return y

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, conv=nn.Conv2d, dropRate=0.2):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv(c_in, c_out, kernel_size=1, bias=True)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block=BasicBlock, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class MultiDenseMLP(nn.Module):
    def __init__(self,
                 mlps: List[int], #for output
                 mlps2: List[int], # for inputs):
                 ):

        super(MultiDenseMLP, self).__init__()
        assert len(mlps) == len(mlps2)
        self.dense_modules = nn.ModuleList()
        c_in = mlps2[0]
        for i in range(len(mlps)):
            c_out = mlps[i]
            self.dense_modules.append(
                nn.Sequential(nn.Conv2d(c_in, c_out, kernel_size=1, bias=True),
                              nn.BatchNorm2d(c_out),
                              nn.ReLU(inplace=True))
            )
            if i < len(mlps) - 1:
                c_in = c_in + c_out + mlps2[i + 1]

    def forward(self, x:List):

        pc = x[0]
        size = len(self.dense_modules)
        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](pc)
            if i < size - 1:
                pc = torch.cat([pc, y, x[i + 1]], 1)

        return y

class DenseEdgeModule(nn.Module):
    def __init__(self, input_channel=64, levels=4, growth_rate=64, k=20):
        super(DenseEdgeModule, self).__init__()
        self.k = k
        self.dense_modules = nn.ModuleList()
        for i in range(levels):
            self.dense_modules.append(
                nn.Sequential(nn.Conv2d(input_channel, growth_rate, kernel_size=1, bias=True),
                              nn.BatchNorm2d(growth_rate),
                              nn.LeakyReLU(negative_slope=0.2))
            )
            input_channel = input_channel + growth_rate

    def forward(self, x):

        x = ops.get_graph_feature(x,k=self.k)
        for i in range(len(self.dense_modules)):
            y = self.dense_modules[i](x)
            x = torch.cat([x,y],1)
        y = y.max(dim=-1, keepdim=False)[0]

        return y

class Mish(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    torch.nn.init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    # https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    # https://github.com/rosinality/sagan-pytorch/blob/master/model_resnet.py

    def __init__(self, in_dim=128, activation='relu'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        size = x.size()
        flatten = x.view(size[0], size[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*size)
        out = self.gamma * attn + x

        return out

class Self_Attn2(nn.Module):
    """ Self attention Layer"""
    # https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    # https://github.com/rosinality/sagan-pytorch/blob/master/model_resnet.py

    def __init__(self, in_dim=128, activation='relu'):
        super(Self_Attn2, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=True),
                      nn.BatchNorm1d(in_dim // 8),
                      nn.LeakyReLU(negative_slope=0.2))

        self.key = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=True),
                                   nn.BatchNorm1d(in_dim // 8),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.value = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=True),
                                 nn.BatchNorm1d(in_dim),
                                 nn.LeakyReLU(negative_slope=0.2))
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        size = x.size()
        flatten = x.view(size[0], size[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*size)
        out = self.gamma * attn + x

        return out

class Dense_Attn(nn.Module):
    """ Self attention Layer"""
    # https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    # https://github.com/rosinality/sagan-pytorch/blob/master/model_resnet.py

    def __init__(self, in_dim=128, activation='relu'):
        super(Dense_Attn, self).__init__()

        #self.atten = Self_Attn(in_dim=in_dim)
        self.atten = Self_Attn(in_dim=in_dim)


        self.dense = DenseModule1D(in_dim=in_dim, levels=3, growth_rate=in_dim)


    def forward(self, x, res=True):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        pt = x
        x = self.atten(x)
        x = self.dense(x)
        if res:
            return pt + x
        else:
            return x

class Dense_Attn_2D(nn.Module):
    """ Self attention Layer"""
    # https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    # https://github.com/rosinality/sagan-pytorch/blob/master/model_resnet.py

    def __init__(self, in_dim=128, activation='relu',mlps=[],atten="gc"):
        super(Dense_Attn_2D, self).__init__()

        if atten=='gc':
            self.atten = GC_attn(in_dim=mlps[-1])
        else:
            self.atten = Self_Attn(in_dim=mlps[-1])

        self.dense = DenseModule2D(in_dim=in_dim, mlps=mlps)


    def forward(self, x, res=True):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        pt = x
        # import ipdb
        # ipdb.set_trace()
        x = self.dense(x)
        x = self.atten(x)

        return x



class GC_attn(nn.Module):

    def __init__(self, in_dim, out_dim=None, pool='att', fusions=['channel_add','channel_mul']):
        super(GC_attn, self).__init__()
        #assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        #assert len(fusions) > 0, 'at least one fusion should be used'
        self.in_dim = in_dim
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv1d(in_dim, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1),
                nn.LayerNorm([self.out_dim,1]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.out_dim, self.in_dim, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1),
                nn.LayerNorm([self.out_dim, 1]),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.out_dim, self.in_dim, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        B, C, N = x.size()
        if self.pool == 'att':
            input_x = x
            # [B, C, N]
            input_x = input_x.view(B, C, N)
            # [B, 1, N]
            context_mask = self.conv_mask(x)
            # [B, N, 1]
            context_mask = self.softmax(context_mask).permute(0, 2, 1)
            context = torch.bmm(input_x, context_mask)
        else:
            # [B, C, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        size = x.size()
        flatten = x.view(size[0], size[1], -1)
        context = self.spatial_pool(flatten)
        # import ipdb
        # ipdb.set_trace()
        if self.channel_mul_conv is not None:
            # [B, C, 1]
            channel_mul_term = self.channel_mul_conv(context)
            out = flatten * channel_mul_term
        else:
            out = flatten
        if self.channel_add_conv is not None:
            # [B, C, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out.view(*size)






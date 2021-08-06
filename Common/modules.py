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
from torch.nn import Parameter as P
from torch.nn import Parameter
# add for shape-preserving Loss
from metrics.pointops import pointops
from torch import Tensor
from collections import OrderedDict
from torch import einsum
from einops import repeat
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
cudnn.benchnark=True

################################################################################################
# -------------------------------- class of nework structure -----------------------------------
################################################################################################

# ---------------------------------------G---------------------------------------

def conv(nin, nout, kernel_size=3, stride=1, padding=1, layer=nn.Conv2d,
         ws=False, bn=False, pn=False, activ=None, gainWS=2):
    conv = layer(nin, nout, kernel_size, stride=stride, padding=padding, bias=False if bn else True)
    layers = OrderedDict()

    if ws:
        layers['ws'] = WScaleLayer(conv, gain=gainWS)

    layers['conv'] = conv

    if bn:
        layers['bn'] = nn.BatchNorm2d(nout)
    if activ:
        if activ == nn.PReLU:
            # to avoid sharing the same parameter, activ must be set to nn.PReLU (without '()') and initialized here
            layers['activ'] = activ(num_parameters=1)
        else:
            layers['activ'] = activ
    if pn:
        layers['pn'] = PixelNorm()
    return nn.Sequential(layers)

class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        B, C, N = x.shape
        group_size = min(self.group_size, B)
        y = x.reshape([group_size, -1, self.num_new_features, C // self.num_new_features, N])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, N).clone().reshape(B, self.num_new_features, N)
        z = torch.cat([x, y], dim=1)

        # out_std = torch.sqrt(x.var(0, unbiased=False) + 1e-8)
        # mean_std = out_std.mean()
        # mean_std = mean_std.expand(x.size(0), 1, N)
        # z = torch.cat([x, mean_std], 1)

        return z

class MinibatchStdDev(torch.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
    """

    def __init__(self, group_size  = 4):
        """

        Args:
            group_size: Size of each group into which the batch is split
        """
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"

    def forward(self, x, alpha = 1e-8):
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        B, C, N = x.shape
        if B > self.group_size:
            assert B % self.group_size == 0, (
                f"batch_size {B} should be "
                f"perfectly divisible by group_size {self.group_size}"
            )
            group_size = self.group_size
        else:
            group_size = B

        # reshape x into a more amenable sized tensor
        y = torch.reshape(x, [group_size, -1, C, N])

        # indicated shapes are after performing the operation
        # [G x M x C x H x W] Subtract mean over groups
        y = y - y.mean(dim=0, keepdim=True)

        # [M x C x H x W] Calc standard deviation over the groups
        y = torch.sqrt(y.pow(2).mean(dim=0, keepdim=False) + alpha)

        # [M x 1 x 1 x 1]  Take average over feature_maps and pixels.
        y = y.mean(dim=[1, 2], keepdim=True)

        # [B x 1 x H x W]  Replicate over group and pixels
        y = y.repeat(group_size, 1, N)

        # [B x (C + 1) x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


class WScaleLayer(nn.Module):
    def __init__(self, incoming, gain=2):
        super(WScaleLayer, self).__init__()

        self.gain = gain
        self.scale = (self.gain / incoming.weight[0].numel()) ** 0.5

    def forward(self, input):
        return input * self.scale

    def __repr__(self):
        return '{}(gain={})'.format(self.__class__.__name__, self.gain)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(self, channels, dlatent_size, use_wscale,
                 use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()

        layers = []
        if use_noise:
            layers.append(('noise', NoiseLayer(channels)))
        layers.append(('activation', activation_layer))
        if use_pixel_norm:
            layers.append(('pixel_norm', PixelNorm()))
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))


    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        return x

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class PixelwiseNorm(torch.nn.Module):
    """
    ------------------------------------------------------------------------------------
    Pixelwise feature vector normalization.
    reference:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    ------------------------------------------------------------------------------------
    """

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    @staticmethod
    def forward(x, alpha=1e-8):
        y = x.pow(2.0).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y

class EqualConv1d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv1d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class EqualizedLR(nn.Module):
    def __init__(self, layer):
        super().__init__()

        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        layer.bias.data.fill_(0)

        self.wscale = layer.weight.data.detach().pow(2.).mean().sqrt()
        layer.weight.data /= self.wscale

        self.layer = layer

    def forward(self, x):
        return self.layer(x * self.wscale)


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))


def snlinear(in_features, out_features):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features))


def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim))



class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)


# class NoiseLayer(nn.Module):
#     def __init__(self, loss_type):
#         super().__init__()
#         self.magnitude = 0
#         self.loss_type = loss_type
#
#     def forward(self, x):
#         if self.loss_type != "lsgan":
#             return x
#         return x + (self.magnitude * torch.randn_like(x))
#


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))
        self.noise = None

    def forward(self, x, noise=None):
        if noise is None and self.noise is None:
            noise = 0.2*torch.randn(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
        elif noise is None:
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1) * noise
        return x

class Self_Attn2(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels):
        super(Self_Attn2, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.snconv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_attn(attn_g)
        # Out
        out = x + self.sigma*attn_g
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, N = x.size()
        proj_query = self.query_conv(x)  # B X C X N
        proj_key = self.key_conv(x)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X N X N
        proj_value = self.value_conv(x)   # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))

        out = self.gamma * out + x
        return out#, attention


class Attention(nn.Module):
  def __init__(self, ch, name='attention'):
    super(Attention, self).__init__()
    # Channel multiplier
    self.ch = ch
    self.which_conv = nn.Conv2d

    self.theta = nn.Conv1d(self.ch, self.ch // 8, 1, bias=False)
    self.phi = nn.Conv1d(self.ch, self.ch // 8, 1, bias=False)
    self.g = nn.Conv1d(self.ch, self.ch // 2, 1, bias=False)
    self.o = nn.Conv1d(self.ch // 2, self.ch, 1, bias=False)
    # Learnable gain parameter
    self.gamma = P(torch.tensor(0.), requires_grad=True)

  def forward(self, x, y=None):
    # Apply convs
    theta = self.theta(x)
    phi = self.phi(x)
    g = self.g(x)
    # Perform reshapes
    # Matmul and softmax to get attention maps
    beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
    # Attention map times g path
    o = self.o(torch.bmm(g, beta.transpose(1,2)))
    return self.gamma * o + x



class fcbr(nn.Module):
    """ fc-bn-relu
    [B, Fin] -> [B, Fout]
    """
    def __init__(self, Fin, Fout):
        super(fcbr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class fcdbr(nn.Module):
    """ fc-dp-bn-relu
    [B, Fin] -> [B, Fout]
    """
    def __init__(self, Fin, Fout, dp=0.5):
        super(fcdbr, self).__init__()
        self.fc = nn.Linear(Fin, Fout)
        self.dp = nn.Dropout(dp)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.fc(x)
        x = self.dp(x)
        x = self.bn(x)
        x = self.ac(x)
        return x

class conv1dbr(nn.Module):
    """ Conv1d-bn-relu
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, kernel_size):
        super(conv1dbr, self).__init__()
        self.conv = nn.Conv1d(Fin, Fout, kernel_size)
        self.bn = nn.BatchNorm1d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, N]
        x = self.bn(x)
        x = self.ac(x)
        return x

class conv2dbr(nn.Module):
    """ Conv2d-bn-relu
    [B, Fin, H, W] -> [B, Fout, H, W]
    """
    def __init__(self, Fin, Fout, kernel_size, stride=[1,1]):
        super(conv2dbr, self).__init__()
        self.conv = nn.Conv2d(Fin, Fout, kernel_size, stride)
        self.bn = nn.BatchNorm2d(Fout)
        self.ac = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x) # [B, Fout, H, W]
        x = self.bn(x)
        x = self.ac(x)
        return x


def pairwise_dist(x,y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    # xx = torch.bmm(x, x.transpose(2,1))
    xx = torch.sum(x ** 2, dim=2, keepdim=True)
    yy = torch.sum(y ** 2, dim=2, keepdim=True)
    xy = -2 * torch.bmm(x, y.permute(0, 2, 1))
    dist = xy + xx + yy.permute(0, 2, 1)  # [B, N, N]
    return dist


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx




def get_graph_feature(x, k=20, idx=None):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
    """
    B, dims, N = x.shape
    x = x.view(B, -1, N)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, num_dims)
    x = x.view(B, N, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)

    if return_idx:
        return ee, idx
    return ee

def get_edge_features_xyz(x, pc, k, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
        idx
    """
    B, dims, N = x.shape

    # ----------------------------------------------------------------
    # batched pair-wise distance in feature space maybe is can be changed to coordinate space
    # ----------------------------------------------------------------
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]

    # get k NN id    
    _, idx_o = torch.sort(dist, dim=2)
    idx = idx_o[: ,: ,1:k+1] # [B, N, k]
    idx = idx.contiguous().view(B, N*k)

    
    # gather
    neighbors = []
    xyz =[]
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

        tp = torch.index_select(pc[b], 1, idx[b])
        tp = tp.view(3, N, k)
        xyz.append(tp)

    neighbors = torch.stack(neighbors)  # [B, d, N, k]
    xyz = torch.stack(xyz)              # [B, 3, N, k]
    
    # centralize
    central = x.unsqueeze(3).repeat(1, 1, 1, k)         # [B, d, N, 1] -> [B, d, N, k]
    central_xyz = pc.unsqueeze(3).repeat(1, 1, 1, k)    # [B, 3, N, 1] -> [B, 3, N, k]
    
    e_fea = torch.cat([central, neighbors-central], dim=1)
    e_xyz = torch.cat([central_xyz, xyz-central_xyz], dim=1)
    
    assert e_fea.size() == (B, 2*dims, N, k) and e_xyz.size() == (B, 2*3, N, k)
    return e_fea, e_xyz


class edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k):
        super(edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.conv = conv2dbr(2*Fin, Fout, 1)

    def forward(self, x):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        x = self.conv(x) # [B, Fout, N, k]
        x, _ = torch.max(x, 3) # [B, Fout, N]
        assert x.shape == (B, self.Fout, N)
        return x


class upsample_edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, num):
        super(upsample_edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.num = num
        
        #self.conv1 = conv2dbr(2*Fin, 2*Fin, 1, 1)
        #self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k+2], [1, 1])
        self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k], [1, 1])
        
        self.inte_conv_hk = nn.Sequential(
            #nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, 4*Fin, [1, k//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4*Fin),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k, self.num) # [B, 2Fin, N, k]


        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        #x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)                                   # Bx2CCxNxKK/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2CCxKK/2
        #inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2+1)       # BxNxCx2x(k//2+1)
        #inte_x = inte_x.contiguous().view(BB, NN, CC, KK+2)             # BxNxCx(k+2)
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)         # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)               # BxNxCx(k+2)
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNxk
        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k

        x = self.conv2(merge_x) # [B, 2*Fout, N, 1]

        x = x.unsqueeze(3)                    # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2*N)

        assert x.shape == (B, self.Fout, 2*N)
        return x

class bilateral_upsample_edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """
    def __init__(self, Fin, Fout, k, num, softmax=True):
        super(bilateral_upsample_edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.softmax = softmax
        self.num = num

        # self.conv = conv2dbr(2*Fin, Fout, [1, 20], [1, 20])
        #self.conv1 = conv2dbr(2*Fin, 2*Fin, 1 ,1)
        self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k], [1, 1])

        self.conv_xyz = nn.Sequential(
            nn.Conv2d(6, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_fea = nn.Sequential(
            nn.Conv2d(2*Fin, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_all = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 2*Fin, 1),
            nn.BatchNorm2d(2*Fin),
            nn.LeakyReLU(inplace=True)
        )

        self.inte_conv_hk = nn.Sequential(
            #nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, 4*Fin, [1, k//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(4*Fin),
            nn.LeakyReLU(inplace = True)
        )

    def forward(self, x, pc):
        B, Fin, N = x.size()
        
        #x = get_edge_features(x, self.k, self.num); # [B, 2Fin, N, k]
        x, y = get_edge_features_xyz(x, pc, self.k, self.num); # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]
        
        
        w_fea = self.conv_fea(x)
        w_xyz = self.conv_xyz(y)
        w = w_fea * w_xyz
        w = self.conv_all(w)
        if self.softmax == True:
            w = F.softmax(w, dim=-1)    # [B, Fout, N, k] -> [B, Fout, N, k]
        
        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        #x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)                                   # Bx2CxNxk/2
        inte_x = inte_x.transpose(2, 1)                                 # BxNx2Cxk/2
        inte_x = inte_x.contiguous().view(BB, NN, CC, 2, KK//2)       # BxNxCx2x(k//2+1)
        inte_x = inte_x.contiguous().view(BB, NN, CC, KK)             # BxNxCx(k+2)
      
        inte_x = inte_x.permute(0, 2, 1, 3)                             # BxCxNx(k+2)
        inte_x = inte_x * w
        
        # Here we concatenate the interpolated feature with the original feature.
        merge_x = torch.cat((x, inte_x), 3)                             # BxCxNx2k
        
        # Since conv2 uses a wide kernel size, the process of sorting by distance can be omitted.
        x = self.conv2(merge_x) # [B, 2*Fout, N, 1]

        x = x.unsqueeze(3)                    # BxkcxN
        x = x.contiguous().view(B, self.Fout, 2, N)
        x = x.contiguous().view(B, self.Fout, 2*N)

        assert x.shape == (B, self.Fout, 2*N)
        return x


class bilateral_block_l1(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20):
        super(bilateral_block_l1, self).__init__()
        self.maxpool = nn.MaxPool2d((1, maxpool), (1, 1))

        self.upsample_cov = nn.Sequential(
            upsample_edgeConv(Fin, Fout, num_k // 2, 1),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin, 2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2 * point_num)

        xs = xs.view(batchsize, -1, 1)
        xs = xs.repeat(1, 1, 2 * point_num)
        x_ec = self.upsample_cov(x)
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)

        return x_out, g_out


class bilateral_block_l2(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l2, self).__init__()
        self.maxpool = nn.MaxPool2d((1, maxpool), (1, 1))

        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k // 2, 1, softmax=softmax)  # (256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin, 2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        batchsize, _, point_num = x.size()
        xs = self.maxpool(x)
        xs = xs.view(batchsize, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2 * point_num)

        xs = xs.view(batchsize, -1, 1)
        xs = xs.repeat(1, 1, 2 * point_num)

        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)

        return x_out, g_out


class bilateral_block_l3(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l3, self).__init__()

        self.maxpool = nn.MaxPool2d((1, maxpool), (1, 1))

        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k // 2, 1, softmax=softmax)  # (256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(batchsize, -1, 1)
        g = g.repeat(1, 1, 2 * point_num)

        xs = xs.view(batchsize, -1, 1)
        xs = xs.repeat(1, 1, 2 * point_num)
        # x_ec = self.upsample_cov(x)
        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)

        return x_out, g_out


class bilateral_block_l4(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20, softmax=True):
        super(bilateral_block_l4, self).__init__()

        self.maxpool = nn.MaxPool2d((1, maxpool), (1, 1))

        self.upsample_cov = bilateral_upsample_edgeConv(Fin, Fout, num_k // 2, 1, softmax=softmax)  # (256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        batchsize = x.size()[0]
        point_num = x.size()[2]
        xs = self.maxpool(x)
        xs = xs.view(batchsize, -1)
        xs = self.fc(xs)
        xs = xs.view(batchsize, -1, 1)
        xs = xs.repeat(1, 1, 2 * point_num)
        # x_ec = self.upsample_cov(x)
        x_ec = self.relu_uc(self.bn_uc(self.upsample_cov(x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        return x_out


class bilateral_block(nn.Module):
    def __init__(self, Fin, Fout, maxpool, stride=1, num_k=20):
        super(bilateral_block, self).__init__()

        self.upsample_cov = nn.Sequential(
            upsample_edgeConv(Fin, Fout, num_k // 2, 1),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin, 2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        B,C,N = x.size()
        xs = torch.max(x, 2, keepdim=True)[0]
        # xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)


        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, 2 * N)
        x_ec = self.upsample_cov(x)
        x_out = torch.cat((xs, x_ec), 1)


        return x_out

class deform_block_head(nn.Module):
    def __init__(self, Fin, Fout, stride=1, num_k=20):
        super(deform_block_head, self).__init__()

        self.deform_cov = nn.Sequential(
            deform_edgeConv_first(Fin, Fout, num_k),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        B = x.size()[0]
        N = x.size()[2]
        xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(B, -1, 1)
        g = g.repeat(1, 1, 2 * N)

        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, 2 * N)
        # x_ec = self.upsample_cov(x)
        x_ec = self.deform_cov(x)
        x_out = torch.cat((xs, x_ec), 1)

        g_out = torch.cat((g, x_ec), dim=1)

        return x_out, g_out


class deform_block_middle(nn.Module):
    def __init__(self, Fin, Fout, stride=1, num_k=20, softmax=True):
        super(deform_block_middle, self).__init__()

        self.deform_cov = deform_edgeConv(Fin, Fout, num_k)  # (256->512)
        self.add_cov = nn.Sequential(
            nn.Conv1d(2*Fout, Fout, 1),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)
        #self.attn = Attention(Fout+512)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, pc):
        B = x.size()[0]
        N = x.size()[2]
        xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(B, -1, 1)
        g = g.repeat(1, 1, N)

        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, N)
        # x_ec = self.upsample_cov(x)

        x_ec = self.relu_uc(self.bn_uc(self.deform_cov(x, pc)))


        x_out = torch.cat((xs, x_ec), 1)
        x_out = self.add_cov(x_out)

        g_out = torch.cat((g, x_ec), dim=1)

        # if False:
        #     g_out = self.attn(g_out)


        return x_out, g_out


class deform_block_tail(nn.Module):
    def __init__(self, Fin, Fout, stride=1, num_k=20, softmax=True):
        super(deform_block_tail, self).__init__()

        self.deform_cov = deform_edgeConv(Fin, Fout, num_k)  # (256->512)
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, x, pc):
        B = x.size()[0]
        N = x.size()[2]
        xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)

        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, N)
        # x_ec = self.upsample_cov(x)
        x_ec = self.relu_uc(self.bn_uc(self.deform_cov(x, pc)))
        x_out = torch.cat((xs, x_ec), 1)

        return x_out


class deform_block_feat_middle(nn.Module):
    def __init__(self, Fin, Fout, stride=1, num_k=20, softmax=True):
        super(deform_block_feat_middle, self).__init__()

        self.deform_cov = deform_edgeConv_feat(Fin, Fout, num_k,softmax=softmax)  # (256->512)
        self.add_cov = nn.Sequential(
            nn.Conv1d(2*Fout, Fout, 1),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, x):
        B = x.size()[0]
        N = x.size()[2]
        xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)


        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, N)
        # x_ec = self.upsample_cov(x)

        x_ec = self.relu_uc(self.bn_uc(self.deform_cov(x)))

        x_out = torch.cat((xs, x_ec), 1)
        x_out = self.add_cov(x_out)


        return x_out

class deform_block_feat(nn.Module):
    def __init__(self, Fin, Fout, stride=1, num_k=20, softmax=True):
        super(deform_block_feat, self).__init__()

        self.deform_cov = deform_edgeConv_feat(Fin, Fout, num_k,softmax=softmax)  # (256->512)
        self.add_cov = nn.Sequential(
            nn.Conv1d(2*Fout, Fout, 1),  # (128->256)
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True)
        )
        self.bn_uc = nn.BatchNorm1d(Fout)
        self.relu_uc = nn.LeakyReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(Fin, Fin),
            nn.BatchNorm1d(Fin),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(Fin,2*Fin),
            # nn.BatchNorm1d(2*Fin),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(Fin, Fout),
            nn.BatchNorm1d(Fout),
            nn.LeakyReLU(inplace=True),
        )
        self.g_fc = nn.Sequential(
            nn.Linear(Fout, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        B = x.size()[0]
        N = x.size()[2]
        xs = torch.max(x, 2, keepdim=True)[0]
        xs = xs.view(B, -1)
        xs = self.fc(xs)

        g = self.g_fc(xs)
        g = g.view(B, -1, 1)
        g = g.repeat(1, 1, N)

        xs = xs.view(B, -1, 1)
        xs = xs.repeat(1, 1, N)
        # x_ec = self.upsample_cov(x)

        x_ec = self.relu_uc(self.bn_uc(self.deform_cov(x)))

        x_out = torch.cat((xs, x_ec), 1)
        x_out = self.add_cov(x_out)

        g_out = torch.cat((g, x_ec), dim=1)

        return x_out, g_out


class deform_edgeConv_first(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """

    def __init__(self, Fin, Fout, k):
        super(deform_edgeConv_first, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout

        # self.conv1 = conv2dbr(2*Fin, 2*Fin, 1, 1)
        # self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k+2], [1, 1])
        self.conv2 = conv2dbr(Fin, Fout, [1, k], [1, 1])

        self.inte_conv_hk = nn.Sequential(
            # nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2 * Fin, Fin, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k)  # [B, 2Fin, N, k]

        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        # x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)  # Bx2CCxNxKK/2
        x = self.conv2(inte_x)  # [B, Fout, N, 1]

        x = x.unsqueeze(3)  # BxkcxN

        return x



class deform_edgeConv_simple(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """

    def __init__(self, Fin, Fout, k):
        super(deform_edgeConv_simple, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout

        # self.conv1 = conv2dbr(2*Fin, 2*Fin, 1, 1)
        # self.conv2 = conv2dbr(2*Fin, 2*Fout, [1, 2*k+2], [1, 1])
        self.conv2 = conv2dbr(Fout, Fout, [1, k], [1, 1])

        self.inte_conv_hk = nn.Sequential(
            # nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2 * Fin, Fout, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fout),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, pc):
        B, Fin, N = x.shape
        x = get_edge_features(x, self.k)  # [B, 2Fin, N, k]

        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        # x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)  # Bx2CCxNxKK/2
        x = self.conv2(inte_x)  # [B, Fout, N, 1]

        x = x.squeeze(3)  # BxkcxN

        return x

class deform_edgeConv(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """

    def __init__(self, Fin, Fout, k, softmax=True):
        super(deform_edgeConv, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.softmax = softmax

        # self.conv = conv2dbr(2*Fin, Fout, [1, 20], [1, 20])
        # self.conv1 = conv2dbr(2*Fin, 2*Fin, 1 ,1)
        self.conv2 =  nn.Sequential(
            # nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(Fin, Fout, [1, k], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )
         #   conv2dbr(Fin, Fout, [1,  k], [1, 1])

        self.conv_xyz = nn.Sequential(
            nn.Conv2d(6, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_fea = nn.Sequential(
            nn.Conv2d(2*Fin, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv_all = nn.Sequential(
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, Fin, 1),
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )

        self.inte_conv_hk = nn.Sequential(
            # nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, Fin, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, pc):
        B, Fin, N = x.size()

        # x = get_edge_features(x, self.k, self.num); # [B, 2Fin, N, k]
        x, y = get_edge_features_xyz(x, pc, self.k)  # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]

        w_fea = self.conv_fea(x)
        w_xyz = self.conv_xyz(y)
        w = w_fea * w_xyz
        w = self.conv_all(w)
        if self.softmax == True:
            w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        # x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)  # Bx2CxNxk
        inte_x = inte_x * w  # Bx2CxNxk

        # Since conv2 uses a wide kernel size, the process of sorting by distance can be omitted.
        x = self.conv2(inte_x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x


class deform_edgeConv_feat(nn.Module):
    """ Edge Convolution using 1x1 Conv h
    [B, Fin, N] -> [B, Fout, N]
    """

    def __init__(self, Fin, Fout, k, softmax=True):
        super(deform_edgeConv_feat, self).__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        self.softmax = softmax

        # self.conv = conv2dbr(2*Fin, Fout, [1, 20], [1, 20])
        # self.conv1 = conv2dbr(2*Fin, 2*Fin, 1 ,1)
        self.conv2 = conv2dbr(Fin, Fout, [1,  k], [1, 1])

        self.conv_fea = nn.Sequential(
            nn.Conv2d(2*Fin, 16, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, Fin, 1),
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )

        self.inte_conv_hk = nn.Sequential(
            # nn.Conv2d(2*Fin, 4*Fin, [1, k//2], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.Conv2d(2*Fin, Fin, [1, 1], [1, 1]),  # Fin, Fout, kernel_size, stride
            nn.BatchNorm2d(Fin),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        B, Fin, N = x.size()

        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        #x, y = get_edge_features_xyz(x, pc, self.k)  # feature x: [B, 2Fin, N, k] coordinate y: [B, 6, N, k]

        w = self.conv_fea(x)
        if self.softmax == True:
            w = F.softmax(w, dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]

        # -------------learn_v2----------------------
        BB, CC, NN, KK = x.size()
        # x = self.conv1(x)
        inte_x = self.inte_conv_hk(x)  # Bx2CxNxk
        inte_x = inte_x * w  # Bx2CxNxk

        # Since conv2 uses a wide kernel size, the process of sorting by distance can be omitted.
        x = self.conv2(inte_x)  # [B, 2*Fout, N, 1]

        x = x.squeeze(3)  # BxCxN

        return x


class PointTransformerLayer(nn.Module):
    def __init__( self, dim, pos_mlp_hidden_dim = 64, attn_mlp_hidden_mult = 4):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, 1),
        )

    def forward(self, x, pos):
        n = x.shape[1]

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None] - pos[:, None, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None] - k[:, None, :]

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb).squeeze(dim = -1)

        # expand transformed features and add relative positional embeddings
        v = repeat(v, 'b j d -> b i j d', i = n)
        v = v + rel_pos_emb

        # attention
        attn = sim.softmax(dim = -1)

        # aggregate
        agg = einsum('b i j, b i j d -> b i d', attn, v)
        return agg


attn = PointTransformerLayer(
    dim = 128,
    pos_mlp_hidden_dim = 64,
    attn_mlp_hidden_mult = 4
)

x = torch.randn(1, 16, 128)
pos = torch.randn(1, 16, 3)

attn(x, pos) # (1, 16, 128)
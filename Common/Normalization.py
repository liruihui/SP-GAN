#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: Normalization.py 
@time: 2021/01/09
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import torch
from torch import nn

x = torch.rand(10, 3, 5, 5) * 10000
#####################################################BN

bn = nn.BatchNorm2d(num_features=3, eps=0, affine=False, track_running_stats=False)

# 乘 10000 为了扩大数值，如果出现不一致，差别更明显
official_bn = bn(x)

# 把 channel 维度单独提出来，而把其它需要求均值和标准差的维度融合到一起
print(x.permute(1,0,2,3).shape)
x1 = x.permute(1,0,2,3).contiguous().view(3, -1)

mu = x1.mean(dim=1).view(1,3,1,1)

# unbiased=False, 求方差时不做无偏估计（除以 N-1 而不是 N），和原始论文一致
# 个人感觉无偏估计仅仅是数学上好看，实际应用中差别不大
std = x1.std(dim=1, unbiased=False).view(1,3,1,1)

my_bn = (x-mu)/std

diff = (official_bn - my_bn).sum()
print('diff={}'.format(diff)) # 差别是 10-5 级的，证明和官方版本基本一致

#####################################################LN
# normalization_shape 相当于告诉程序这本书有多少页，每页多少行多少列
# eps=0 排除干扰
# elementwise_affine=False 不作映射
# 这里的映射和 BN 以及下文的 IN 有区别，它是 elementwise 的 affine，
# 即 gamma 和 beta 不是 channel 维的向量，而是维度等于 normalized_shape 的矩阵
ln = nn.LayerNorm(normalized_shape=[3,5,5], eps=0, elementwise_affine=False)

official_ln = ln(x)

x1 = x.view(10,-1)
mu = x1.mean(dim=1).view(10,1,1,1)
std = x1.std(dim=1, unbiased=False).view(10,1,1,1)

my_ln = (x-mu)/std

diff = (official_ln - my_ln).sum()

print('Diff={}'.format(diff)) # 差别和官方版本数量级在 1e-5


#####################################################IN
# track_running_stats=False，求当前 batch 真实平均值和标准差，
# 而不是更新全局平均值和标准差
# affine=False, 只做归一化，不乘以 gamma 加 beta（通过训练才能确定）
# num_features 为 feature map 的 channel 数目
# eps 设为 0，让官方代码和我们自己的代码结果尽量接近

In = nn.InstanceNorm2d(num_features=3, eps=0, track_running_stats=False)

official_in = In(x)

x1 = x.view(30,-1)
mu = x1.mean(dim=1).view(10,3,1,1)
std = x1.std(dim=1, unbiased=False).view(10,3,1,1)
my_in = (x - mu)/std

diff = (my_in-official_in).sum()
print('Diff={}'.format(diff)) # 误差量级在 1e-5

#####################################################GN
# 分成 4 个 group
# 其余设定和之前相同
gn = nn.GroupNorm(num_groups=4, num_channels=20, eps=0, affine=False)
official_gn = gn(x)

# 把同一 group 的元素融合到一起
x1 = x.view(10,4,-1)
mu = x1.mean(dim=-1).reshape(10,4,-1)
std = x1.std(dim=-1,unbiased=False).reshape(10,4,-1)

x1_norm = (x1-mu)/std
my_gn = x1_norm.reshape(10,20,5,5)

diff = (official_gn - my_gn).sum()
print('diff={}'.format(diff)) # 误差在 1e-4 级

#####https://zhuanlan.zhihu.com/p/56244285
# Tricks for training ProGAN
class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding, bias=True):
        super(EqualizedConv2d, self).__init__()
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.weight_param = nn.Parameter(torch.FloatTensor(out_features, in_features, kernel_size, kernel_size).normal_(0.0, 1.0))
        if self.bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(out_features).fill_(0))
        fan_in = kernel_size * kernel_size * in_features
        self.scale = math.sqrt(2. / fan_in)
    def forward(self, x):
        return F.conv2d(input=x,
                        weight=self.weight_param.mul(self.scale),  # scale the weight on runtime
                        bias=self.bias_param if self.bias else None,
                        stride=self.stride, padding=self.padding)

class PixelwiseNorm(nn.Module):
    def __init__(self, sigma=1e-8):
        super(PixelwiseNorm, self).__init__()
        self.sigma = sigma # small number for numerical stability
    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.sigma).sqrt() # [N1HW]
        return x.div(y)


class MinibatchStddev(nn.Module):
    def __init__(self):
        super(MinibatchStddev, self).__init__()
    def forward(self, x):
        y = x - torch.mean(x, dim=0, keepdim=True)       # [NCHW] Subtract mean over batch.
        y = torch.mean(y.pow(2.), dim=0, keepdim=False)  # [CHW]  Calc variance over batch.
        y = torch.sqrt(y + 1e-8)                         # [CHW]  Calc stddev over batch.
        y = torch.mean(y).view(1,1,1,1)                  # [1111] Take average over fmaps and pixels.
        y = y.repeat(x.size(0),1,x.size(2),x.size(3))    # [N1HW] Replicate over batch and pixels.
        return torch.cat([x, y], 1)                      # [N(C+1)HW] Append as new fmap.


def update_moving_average(self, decay=0.999):
    # update exponential running average (EMA) for the weights of the generator
    # W_EMA_t = decay * W_EMA_{t-1} + (1-decay) * W_G
    with torch.no_grad():
        param_dict_G = dict(self.G.module.named_parameters())
        for name, param_EMA in self.G_EMA.named_parameters():
            param_G = param_dict_G[name]
            assert (param_G is not param_EMA)
            param_EMA.copy_(decay * param_EMA + (1. - decay) * param_G.detach().cpu())


opt_G_state_dict = self.opt_G.state_dict()
old_opt_G_state = opt_G_state_dict['state']
self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
new_opt_G_param_id =  self.opt_G.state_dict()['param_groups'][0]['params']
opt_G_state = copy.deepcopy(old_opt_G_state)
for key in old_opt_G_state.keys():
    if key not in new_opt_G_param_id:
        del opt_G_state[key]
opt_G_state_dict['param_groups'] = self.opt_G.state_dict()['param_groups']
opt_G_state_dict['state'] = opt_G_state
self.opt_G.load_state_dict(opt_G_state_dict)
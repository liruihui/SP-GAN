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

from collections import namedtuple
cudnn.benchnark=True
from torch.nn import AvgPool2d, Conv1d, Conv2d, ConvTranspose2d, Embedding, LeakyReLU, Module

neg = [1e-2, 0.2][0]

class BasicConv1D(nn.Module):
    def __init__(self, Fin, Fout, act=True, norm="BN", kernal=1):
        super(BasicConv1D, self).__init__()

        self.conv = nn.Conv1d(Fin,Fout,kernal)
        if act:
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = nn.BatchNorm1d(Fout) if norm=="BN" else nn.InstanceNorm1d(Fout)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)  # Bx2CxNxk

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, opts, num_point=2048):
        super(Discriminator, self).__init__()
        self.num_point = num_point
        BN = True
        self.small_d = opts.small_d

        self.mlps = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.mode = ["max","max_avg"][0]

        if self.mode == "max":
            dim = 1024
        else:
            dim = 512

        if self.small_d:
            dim = dim//2

        self.fc2 = nn.Sequential(
            nn.Conv1d(256,dim,1),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True)
        )


        self.mlp = nn.Sequential(
            nn.Linear(dim, 512),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(64, 1)
            )

    def forward(self, x):
        B = x.size()[0]


        x = self.mlps(x)
        x = self.fc2(x)

        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)

        if self.mode == "max":
            x = x1
        else:
            x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
            x = torch.cat((x1, x2), 1)
        #x2 = x2.view(batchsize,1024)
        x3 = self.mlp(x)

        return x3




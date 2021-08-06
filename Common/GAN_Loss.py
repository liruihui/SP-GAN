#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: GAN_Loss.py 
@time: 2021/01/11
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""


import torch
import numpy as np
import warnings
from scipy.stats import entropy
import torch.nn as nn
from numpy.linalg import norm
import sys,os
import torch.nn.functional as F
from torch.autograd import Variable, grad



def dis_loss_fake(discriminator,real_image,fake_image,loss):
    fake_predict = discriminator(fake_image)
    if loss == 'wgan-gp':
        fake_predict = fake_predict.mean()
        #fake_predict.backward()

        eps = torch.rand(real_image.size(0), 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
        )[0]
        grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
        ).mean()
        grad_penalty = 10 * grad_penalty
        #grad_penalty.backward()

        fake_loss = fake_predict + grad_penalty
        #grad_loss_val = grad_penalty.item()
        #disc_loss_val = (-real_predict + fake_predict).item()

    elif loss == 'r1':
        fake_predict = F.softplus(fake_predict).mean()
        #fake_predict.backward()
        fake_loss = fake_predict
        #disc_loss_val = (real_predict + fake_predict).item()

def dis_loss_real(discriminator,real_image,fake_image,loss):
    if loss == 'wgan-gp':
        real_predict = discriminator(real_image)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        (-real_predict).backward()

    elif loss == 'r1':
        real_image.requires_grad = True
        real_scores = discriminator(real_image)
        real_predict = F.softplus(-real_scores).mean()
        real_predict.backward(retain_graph=True)

        grad_real = grad(
            outputs=real_scores.sum(), inputs=real_image, create_graph=True
        )[0]
        grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty
        grad_penalty.backward()


def gen_loss(discriminator,real_image,fake_image,loss):
    predict = discriminator(fake_image)

    if loss == 'wgan-gp':
        loss = -predict.mean()

    elif loss == 'r1':
        loss = F.softplus(-predict).mean()


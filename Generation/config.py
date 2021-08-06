#!/usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:liruihui
@file: config_cls.py
@time: 2019/09/17
@contact: ruihuili.lee@gmail.com
@github: https://liruihui.github.io/
@description: 
"""

import argparse
import os

def str2bool(x):
    return x.lower() in ('true')


def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def check_args(args):
    if args.model_dir is None:
        print('please create model dir')
        exit()
    if args.network is None:
        print('please select model!!!')
        exit()
    check_folder(args.checkpoint_dir)                                   # --checkpoint_dir
    check_folder(os.path.join(args.checkpoint_dir, args.model_dir))     # --chekcpoint_dir + model_dir

    try: # --epoch
        assert args.max_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try: # --batch_size
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

root = os.getcwd()[:5]
# define the H5 data folder
data_root_part = root+"/lirh/pointcloud2/dataset/Generation/shapenetcore_partanno_segmentation_benchmark_v0/"
data_root_h5 = root+"/lirh/pointcloud2/dataset/Generation/H5/"

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train', help='train or test ?')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=24, help='input batch size [default: 30]')
parser.add_argument('--np', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--nk',type=int, default=20,help = 'number of the knn graph point')
parser.add_argument('--lr_g', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--lr_d', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--lr_decay_rate', type=float, default=0.7, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--scale', type=float, default=1.0, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--lr_decay_feq', type=int, default=40, help='use offset')
parser.add_argument('--neg', type=float, default=0.01, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--neg2', type=float, default=0.01, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--save', action='store_true', help='use offset')
parser.add_argument('--augment', type=str2bool, default=False, help='use offset')
parser.add_argument('--off',action='store_true', help='use offset')
parser.add_argument('--part', action='store_true', help='use offset')
parser.add_argument('--part_more', action='store_true', help='use offset')
parser.add_argument('--moving', action='store_true', help='use offset')
parser.add_argument('--max_epoch', type=int, default=6000, help='number of epochs to train for')
parser.add_argument('--nz', type=int, default=128, help='dimensional of noise')
parser.add_argument('--nv', type=float, default=0.2, help='value of noise')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--gan', default='ls', help='[ls,wgan,hinge]')
parser.add_argument('--debug', type=bool, default = True,  help='print log')
parser.add_argument('--data_root', type=str,default=data_root_h5, help='data root [default: xxx]')
parser.add_argument('--data_root_part', type=str,default=data_root_part, help='data root [default: xxx]')
#parser.add_argument('--data_root', default='/test/shapenetcore_partanno_segmentation_benchmark_v0/', help='data root [default: xxx]')
parser.add_argument('--log_dir', default='log', help='log_dir')
parser.add_argument('--log_info', default='log_info.txt', help='log_info txt')
parser.add_argument('--model_dir', default='PDGN_v1', help='model dir [default: None, must input]')
parser.add_argument('--checkpoint_dir', default='checkpoint', help='Checkpoint dir [default: checkpoint]')
parser.add_argument('--snapshot', type=int, default=50, help='how many epochs to save model')
parser.add_argument('--choice', default='Chair', help='choice class')
parser.add_argument('--network', default='PDGN_v1', help='which network model to be used')
parser.add_argument('--savename',default = 'PDGN_v1',help='the generate data name')
parser.add_argument('--pretrain_model_G', default=None, help='use the pretrain model G')
parser.add_argument('--pretrain_model_D', default=None, help='use the pretrain model D')
parser.add_argument('--softmax', type=str2bool, default=True, help='softmax for bilaterl interpolation')
parser.add_argument('--dataset', default='shapenet', help='choice dataset [shapenet, modelnet10, modelnet40]')
parser.add_argument('--restore', action='store_true')
parser.add_argument('--mix', action='store_true')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--attn', action='store_true')
parser.add_argument('--n_mix', action='store_true')
parser.add_argument('--w_mix', action='store_true')
parser.add_argument('--trunc', action='store_true')
parser.add_argument('--use_sgd', action='store_true')
parser.add_argument('--n_rand', action='store_true')
parser.add_argument('--sn', action='store_true')
parser.add_argument('--z_norm', action='store_true')
parser.add_argument('--bal', action='store_true')
parser.add_argument('--bal_para', type=float, default=0.15, help='value of noise')
parser.add_argument('--bal_epoch', type=int, default=30, help='how many epochs to save model')
parser.add_argument('--norm',default = 'IN',help='"BN","IN","PN"')
parser.add_argument('--d_iter', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--g_iter', type=int, default=1, help='how many epochs to save model')
parser.add_argument('--no_global', action='store_true')
parser.add_argument('--dp', action='store_true')
parser.add_argument('--use_noise', action='store_true')
parser.add_argument('--noise_label', action='store_true',help='use offset')
parser.add_argument('--flip_d', action='store_true',help='use offset')
parser.add_argument('--flip_g', action='store_true',help='use offset')
parser.add_argument('--ema', action='store_true')
parser.add_argument('--inst_noise', action='store_true')
parser.add_argument('--lr_decay',action='store_true', help='use offset')
parser.add_argument('--small_d',action='store_true', help='use offset')
parser.add_argument('--cut_d',action='store_true', help='use offset')
parser.add_argument('--keep_idx',action='store_true', help='use offset')
parser.add_argument('--cat',action='store_true', help='use offset')
parser.add_argument('--gat',action='store_true', help='use offset')
parser.add_argument('--same_head',action='store_true', help='use offset')
parser.add_argument('--use_head',action='store_true', help='use offset')
parser.add_argument('--lr_decay_g',action='store_true', help='use offset')
parser.add_argument('--lr_decay_d', action='store_true', help='use offset')

parser.add_argument('--ema_rate', type=float, default=0.999, help='value of ema_rate')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--eql', action='store_true')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')
parser.add_argument('--res', action='store_true', help='use PixelNorm in G')
parser.add_argument('--con', action='store_true')
parser.add_argument('--cls',type=int, default=2,help = 'number of the knn graph point')

parser.add_argument('--FPD_path', type=str, default='./evaluation/pre_statistics_chair.npz', help='Statistics file path to evaluate FPD metric. (default:all_class)')

# Network argumentssoftmax
parser.add_argument('--epochs', type=int, default=2000, help='Integer value for epochs.')
parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
parser.add_argument('--support', type=int, default=10, help='Support value for TreeGCN loop term.')
parser.add_argument('--DEGREE', type=int, default=[1, 2, 2, 2, 2, 2, 64], nargs='+',
                          help='Upsample degrees for generator.')
parser.add_argument('--G_FEAT', type=int, default=[96, 256, 256, 256, 128, 128, 128, 3], nargs='+',
                          help='Features for generator.')
parser.add_argument('--D_FEAT', type=int, default=[3, 64, 128, 256, 512, 1024], nargs='+',
                          help='Features for discriminator.')

parser.add_argument('--lr_t', type=float, default=1e-4, help='Float value for learning rate.')
parser.add_argument('--lr_p', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')

opts = check_args(parser.parse_args())



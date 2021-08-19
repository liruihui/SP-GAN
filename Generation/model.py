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
import logging
import random
import imageio
# add for shape-preserving Loss
from Common.point_operation import normalize_point_cloud
from Generation.H5DataLoader import H5DataLoader
from collections import namedtuple
# from pointnet2.pointnet2_modules import PointNet2SAModule, PointNet2SAModuleMSG
from Common.loss_utils import ChamferLoss,pairwise_CD
from Common import point_operation
from Common import data_utils as d_util
from Common.loss_utils import get_local_pair,compute_all_metrics2,AverageValueMeter,dist_simple
from Common import loss_utils
from tensorboardX import SummaryWriter
from Common.visu_utils import plot_pcd_three_views,point_cloud_three_views,plot_pcd_multi_rows
from tqdm import tqdm
from Generation.Generator import Generator
from Generation.Discriminator import Discriminator

from Common.network_utils import *
import copy
from pprint import pprint
cudnn.benchnark=True

seed = 123
random.seed(seed)
#np.random.seed(seed)
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class Model(object):
    def __init__(self, opts):
        self.opts = opts


    def backup(self):
        if self.opts.phase == 'train':
            source_folder = os.path.join(os.getcwd(),"Generation")
            common_folder = os.path.join(os.getcwd(),"Common")
            data_folder = os.path.join(os.getcwd(), "data_utils")
            os.system("cp %s/Generator.py '%s/Generator.py'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/Discriminator.py '%s/Discriminator.py'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/model.py '%s/model.py'" % (source_folder,self.opts.log_dir))
            os.system("cp %s/loss_utils.py '%s/loss_utils.py'" % (common_folder,self.opts.log_dir))
            os.system("cp %s/H5DataLoader.py '%s/H5DataLoader.py'" % (data_folder,self.opts.log_dir))


    def build_model(self):
        """ Models """

        self.G = Generator(self.opts)
        self.D = Discriminator(self.opts)

        self.multi_gpu = False

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)
            self.multi_gpu = True

        print('# generator parameters:', sum(param.numel() for param in self.G.parameters()))
        print('# discriminator parameters:', sum(param.numel() for param in self.D.parameters()))

        self.G.cuda()
        self.D.cuda()

        """ Training """
        
        beta1 = 0.5
        beta2 = 0.99
        self.optimizerG = optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.opts.lr_g, betas=(beta1, beta2))
        self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.opts.lr_d, betas=(beta1, beta2))

        if self.opts.lr_decay:
            if self.opts.use_sgd:
                self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizerG, self.opts.max_epoch, eta_min=self.opts.lr_g)
            else:
                self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optimizerG, step_size=self.opts.lr_decay_feq, gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_G = None

        if self.opts.lr_decay:
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optimizerD, step_size=self.opts.lr_decay_feq, gamma=self.opts.lr_decay_rate)
        else:
            self.scheduler_D = None

        # define tensors
        self.z = torch.FloatTensor(self.opts.bs, self.opts.nz).cuda()
        self.z = Variable(self.z)

        self.ball = None

        label = torch.full((self.opts.bs,), 1).cuda()
        ones = torch.full((self.opts.bs,), 1).cuda()
        self.fix_z = None

    def noise_generator(self, bs=1,masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))
                #scale = self.opts.nv
                #w = np.random.uniform(low=-scale, high=scale, size=(bs, 1, self.opts.nz))
                noise = np.tile(noise,(1,self.opts.np,1))

            if self.opts.n_mix and random.random() < 0.5:
               noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
               for i in range(bs):
                   id = np.random.randint(0,self.opts.np)
                   idx = np.argsort(self.ball_dist[id])[::1]
                   # idx = np.arange(self.opts.np)
                   # np.random.shuffle(idx)
                   num = int(max(random.random(),0.1)*self.opts.np)
                   noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i,idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise

    def sphere_generator(self,bs=2,static=True):

        if self.ball is None:
            self.ball = np.loadtxt('template/balls/%d.xyz'%self.opts.np)[:,:3]
            self.ball = pc_normalize(self.ball)

            N = self.ball.shape[0]
            # xx = torch.bmm(x, x.transpose(2,1))
            xx = np.sum(self.ball ** 2, axis=(1)).reshape(N, 1)
            yy = xx.T
            xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
            self.ball_dist = xy + xx + yy  # [B, N, N]

        if static:
            ball = np.expand_dims(self.ball, axis=0)
            ball = np.tile(ball, (bs, 1, 1))
        else:
            ball = np.zeros((bs, self.opts.np, 3))
            for i in range(bs):
                idx = np.random.choice(self.ball.shape[0], self.opts.np)
                ball[i] = self.ball[idx]

        ball = Variable(torch.Tensor(ball)).cuda()

        return ball

    def train(self):

        global epoch
        self.build_model()
        self.backup()
        start_epoch = 1
        # restore check-point if it exits
        if self.opts.restore:
            could_load, save_epoch = self.load(self.opts.log_dir)
            if could_load:
                start_epoch = save_epoch
                print(" [*] Load SUCCESS")

            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
        else:
            print('training...')
            self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')
            self.LOG_FOUT.write(str(self.opts) + '\n')

        self.log_string('PARAMETER ...')
        with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(self.opts)):
                log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments
        pprint(self.opts)
        self.writer = None#SummaryWriter(logdir=self.opts.log_dir)
        '''DATA LOADING'''
        self.log_string('Load dataset ...')
        self.train_dataset = H5DataLoader(self.opts, augment=self.opts.augment)

        self.dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.opts.bs,
                                                      shuffle=True, num_workers=int(self.opts.workers),drop_last=True,pin_memory=True)
        self.num_batches = len(self.train_dataset) // self.opts.bs


        self.z_test = torch.FloatTensor(self.opts.bs, self.opts.nz).cuda()
        self.z_test.data.resize_(self.opts.bs, self.opts.nz).normal_(0.0, 1.0)


        # loop for epoch
        start_time = time.time()
        d_avg_meter = AverageValueMeter()
        g_avg_meter = AverageValueMeter()
        real_acc_avg_meter = AverageValueMeter()
        fake_acc_avg_meter = AverageValueMeter()

        global_step = 0

        d_para = 1.0
        g_para = 1.0
        x = self.sphere_generator(bs=self.opts.bs)
        self.fix_z = self.noise_generator(bs=64)

        for epoch in range(start_epoch, self.opts.max_epoch+1):
            self.D.train()
            self.G.train()
            step_d = 0
            step_g = 0
            for idx, data in tqdm(enumerate(self.dataloader, 0),total=len(self.dataloader)):
                requires_grad(self.G, False)
                requires_grad(self.D, True)

                self.optimizerD.zero_grad()

                real_points = Variable(data,requires_grad=True)
                z = self.noise_generator(bs=self.opts.bs)

                d_fake_preds =self.G(x, z)
                real_points = real_points.transpose(2, 1).cuda()
                d_fake_preds = d_fake_preds.detach()


                d_real_logit = self.D(real_points)
                d_fake_logit = self.D(d_fake_preds)


                lossD,info= loss_utils.dis_loss(d_real_logit,d_fake_logit,gan=self.opts.gan,noise_label=self.opts.flip_d)

                lossD.backward()
                self.optimizerD.step()

                # -----------------------------------train G-----------------------------------

                requires_grad(self.G, True)
                requires_grad(self.D, False)


                self.optimizerG.zero_grad()

                z = self.noise_generator(bs=self.opts.bs)
                g_fake_preds =self.G(x, z)


                g_real_logit = self.D(real_points)
                g_fake_logit = self.D(g_fake_preds)
                lossG,_ = loss_utils.gen_loss(g_real_logit,g_fake_logit,gan=self.opts.gan,noise_label=self.opts.flip_g)

                lossG.backward()
                self.optimizerG.step()


                d_avg_meter.update(lossD.item())
                g_avg_meter.update(lossG.item())

                real_acc_avg_meter.update(info['real_acc'])
                fake_acc_avg_meter.update(info['fake_acc'])

                if self.writer is not  None:
                    self.writer.add_scalar("loss/d_Loss", lossD.data, global_step)
                    self.writer.add_scalar("loss/g_Loss", lossG.data, global_step)
                    self.writer.add_scalar("acc/real_acc", info['real_acc'], global_step)
                    self.writer.add_scalar("acc/fake_acc", info['fake_acc'], global_step)

                    self.writer.add_histogram('d_real_logit', d_real_logit, global_step)
                    self.writer.add_histogram('d_fake_logit', d_fake_logit, global_step)
                    self.writer.add_histogram('g_fake_logit', g_fake_logit, global_step)

                    #optimizer.param_groups[0]['lr']
                    #scheduler_G.get_lr()[0]
                    self.writer.add_scalar("lr/lr_g", self.optimizerG.param_groups[0]['lr'], global_step)
                    self.writer.add_scalar("lr/lr_d", self.optimizerD.param_groups[0]['lr'], global_step)

                global_step +=1
                if self.opts.save and global_step%20==0:
                    requires_grad(self.G, False)
                    self.draw_sample_save(epoch=epoch,step=global_step)
                    requires_grad(self.G, True)

            if self.scheduler_G is not None:
                self.scheduler_G.step(epoch)
            if self.scheduler_D is not None:
                self.scheduler_D.step(epoch)

            time_tick = time.time() - start_time
            self.log_string("Epoch: [%2d] time: %2dm %2ds d_loss4: %.8f, g_loss: %.8f" \
                            % (epoch, time_tick / 60, time_tick % 60,  d_avg_meter.avg, g_avg_meter.avg))
            self.log_string("real_acc: %f  fake_acc: %f" % (real_acc_avg_meter.avg, fake_acc_avg_meter.avg))
            self.log_string("lr_g: %f  lr_d: %f" % (self.optimizerG.param_groups[0]['lr'], self.optimizerD.param_groups[0]['lr']))
            print("step_d:%d step_g:%d"%(step_d,step_g))
            # if self.scheduler_G is not None and self.scheduler_D is not None:
            #     print("lr_g: %f  lr_d: %f"%(self.scheduler_G.get_lr()[0],self.scheduler_D.get_lr()[0]))

            requires_grad(self.G, False)
            requires_grad(self.D, True)

            if epoch % self.opts.snapshot == 0:
                self.save(self.opts.log_dir, epoch)

            if False and not self.opts.save:
                self.draw_sample(epoch)


        self.save(self.opts.log_dir, epoch)
        self.LOG_FOUT.close()



    def draw_sample(self, epoch):

        eval_dir = os.path.join(self.opts.log_dir, "plot")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        grid_x = 8
        grid_y = 8

        x = self.sphere_generator(bs=grid_y)


        pcds_list = []
        title_list = []
        for i in range(grid_x):
            title = ["S_%d" % (i * grid_y + j) for j in range(grid_y)]
            with torch.no_grad():
                #z = self.noise_generator(bs=grid_y)
                z = self.fix_z[i*grid_y:(i+1)*grid_y]
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)
                sample_pcs = out_pc.cpu().detach().numpy()
                sample_pcs = normalize_point_cloud(sample_pcs)

            pcds_list.append(0.75 * sample_pcs)
            title_list.append(title)

        plot_name = os.path.join(eval_dir, str(epoch) + ".png")

        plot_pcd_multi_rows(plot_name, pcds_list, title_list, cmap="Reds")



    def draw_sample_save(self, epoch, step):

        eval_dir = os.path.join(self.opts.log_dir, "plot")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        grid_x = 8
        grid_y = 8

        x = self.sphere_generator(bs=grid_y)


        pcds_list = []
        title_list = []
        for i in range(grid_x):
            title = ["S_%d" % (i * grid_y + j) for j in range(grid_y)]
            with torch.no_grad():
                #z = self.noise_generator(bs=grid_y)
                z = self.fix_z[i*grid_y:(i+1)*grid_y]
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)
                sample_pcs = out_pc.cpu().detach().numpy()
                sample_pcs = normalize_point_cloud(sample_pcs)

            pcds_list.append(0.75 * sample_pcs)
            title_list.append(title)

        plot_name = os.path.join(eval_dir, str(step) + ".png")

        plot_pcd_multi_rows(plot_name, pcds_list, title_list, cmap="Reds")

        for i in range(grid_x):
            pcs = normalize_point_cloud(pcds_list[i])
            for j in range(grid_y):
                id = i*grid_y+j
                save_folder = os.path.join(eval_dir,"sample",str(id))
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_name = os.path.join(save_folder,"%d_step_%d.xyz"%(id,step))
                np.savetxt(save_name,pcs[j],fmt="%.6f")



    def test_once(self,epoch):

        #self.G.eval()
        eval_dir = os.path.join(self.opts.log_dir,"plot")
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        bs = 4
        # x = np.loadtxt('template/ball.xyz')
        # x = pc_normalize(x)
        # x = np.expand_dims(x, axis=0)
        # x = np.tile(x, (bs, 1, 1))
        # x = Variable(torch.Tensor(x)).cuda()

        x = self.sphere_generator(bs=bs)

        #if self.fix_z is None:
        self.fix_z = self.noise_generator(bs=bs)

        gen_points4 = self.G(x,self.fix_z)
        # print(gen_points.shape)
        gen_points4 = gen_points4.transpose(2, 1).cpu().data.numpy()  # Bx3x2048 -> Bx2048x3
        gen_points4 = point_operation.normalize_point_cloud(gen_points4)
        pcds = [gen_points4[0], gen_points4[1], gen_points4[2], gen_points4[3]]
        # print(type(pcds), len(pcds))
        # np.asarray(pcds).reshape([3,self.opts.num_point,3])
        plot_path = os.path.join(eval_dir, str(epoch) + ".png")
        visualize_titles = ['S1', 'S2', 'S3', 'S4']
        plot_pcd_three_views(plot_path, pcds, visualize_titles)



    def log_string(self, out_str):
        self.LOG_FOUT.write(out_str+'\n')
        self.LOG_FOUT.flush()
        print(out_str)

    def set_logger(self):
        self.logger = logging.getLogger("CLS")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(os.path.join(self.opts.log_dir, "log_%s.txt" % self.opts.phase))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)


    def load(self, checkpoint_dir):
        if self.opts.pretrain_model_G is None  and self.opts.pretrain_model_D is None:
            print('################ new training ################')
            return False, 1

        print(" [*] Reading checkpoints...")
        #checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)
        
        # ----------------- load G -------------------
        if not self.opts.pretrain_model_G is None:
            resume_file_G = os.path.join(checkpoint_dir, self.opts.pretrain_model_G)
            flag_G = os.path.isfile(resume_file_G), 
            if flag_G == False:
                print('G--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_G------>: {}'.format(resume_file_G))
                checkpoint = torch.load(resume_file_G)
                self.G.load_state_dict(checkpoint['G_model'])
                self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint['G_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load D -------------------
        if not self.opts.pretrain_model_D is None:
            resume_file_D = os.path.join(checkpoint_dir, self.opts.pretrain_model_D)
            flag_D = os.path.isfile(resume_file_D)
            if flag_D == False:
                print('D--> Error: no checkpoint directory found!')
                exit()
            else:
                print('resume_file_D------>: {}'.format(resume_file_D))
                checkpoint = torch.load(resume_file_D)
                self.D.load_state_dict(checkpoint['D_model'])
                D_epoch = checkpoint['D_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_D")
            exit()

        print(" [*] Success to load model --> {} & {}".format(self.opts.pretrain_model_G, self.opts.pretrain_model_D))
        return True, G_epoch

    def save(self, checkpoint_dir, index_epoch):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_name = str(index_epoch)+'_'+self.opts.choice
        path_save_G = os.path.join(checkpoint_dir, save_name+'_G.pth')
        path_save_D = os.path.join(checkpoint_dir, save_name+'_D.pth')
        print('Save Path for G: {}'.format(path_save_G))
        print('Save Path for D: {}'.format(path_save_D))
        torch.save({
                'G_model':self.G.module.state_dict() if self.multi_gpu else self.G.state_dict(),
                'G_optimizer': self.optimizerG.state_dict() ,
                'G_epoch': index_epoch,
            }, path_save_G)
        torch.save({
                # 'D_model1': self.discriminator1.state_dict(),
                # 'D_model2': self.discriminator2.state_dict(),
                # 'D_model3': self.discriminator3.state_dict(),
                'D_model': self.D.module.state_dict() if self.multi_gpu else self.D.state_dict(),
                'D_optimizer': self.optimizerD.state_dict(),
                # 'D_optimizer2': self.optimizerD2.state_dict(),
                # 'D_optimizer3': self.optimizerD3.state_dict(),
                # 'D_optimizer4': self.optimizerD.state_dict(),
                'D_epoch': index_epoch,
            }, path_save_D)

        # torch.save(G, os.path.join(opt.outd, opt.outm, f'G_nch-{opt.nch}_epoch-{epoch}.pth'))
        # torch.save(D, os.path.join(opt.outd, opt.outm, f'D_nch-{opt.nch}_epoch-{epoch}.pth'))
        # torch.save(Gs, os.path.join(opt.outd, opt.outm, f'Gs_nch-{opt.nch}_epoch-{epoch}.pth'))




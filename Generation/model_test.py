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
from datetime import datetime
from Generation.point_operation import normalize_point_cloud
from collections import namedtuple
from Generation.point_operation import plot_pcd_multi_rows,plot_pcd_multi_rows_single_color,plot_pcd_three_views_color,plot_pcd_multi_rows_color
from tqdm import tqdm
from pprint import pprint
#from Generation.Generator import Generator
cudnn.benchnark=True

# seed = 123
# seed = 2021
seed = 1990
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
from Generation.Generator import Generator


def pc_normalize(pc,return_len=False):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    if return_len:
        return m
    return pc

plot_folder = "models/plots"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

class Model(object):
    def __init__(self, opts):
        self.opts = opts

    def build_model_eval(self):
        """ Models """
        self.G = Generator(self.opts)
        self.ball = None
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.G = nn.DataParallel(self.G)

        print('# generator parameters:', sum(param.numel() for param in self.G.parameters()))

        self.G.cuda()
        self.G.eval()

    def noise_generator(self, bs=1, masks=None):

        if masks is None:
            if self.opts.n_rand:
                noise = np.random.normal(0, self.opts.nv, (bs, self.opts.np, self.opts.nz))
            else:
                noise = np.random.normal(0, self.opts.nv, (bs, 1, self.opts.nz))
                # scale = self.opts.nv
                # w = np.random.uniform(low=-scale, high=scale, size=(bs, 1, self.opts.nz))
                noise = np.tile(noise, (1, self.opts.np, 1))

            if self.opts.n_mix and random.random() < 0.8:
                noise2 = np.random.normal(0, self.opts.nv, (bs, self.opts.nz))
                for i in range(bs):
                    idx = np.arange(self.opts.np)
                    np.random.shuffle(idx)
                    num = int(random.random() * self.opts.np)
                    noise[i, idx[:num]] = noise2[i]
        else:
            noise = np.zeros((bs, self.opts.np, self.opts.nz))
            for i in range(masks.shape[0]):
                mask = masks[i]
                unique_mask = np.unique(mask)
                for j in unique_mask:
                    noise_once = np.random.normal(0, 0.2, (1, self.opts.nz))
                    idx = np.where(mask == j)
                    noise[i, idx] = idx

        sim_noise = Variable(torch.Tensor(noise)).cuda()

        return sim_noise

    def sphere_generator(self, bs=2, static=True):

        if self.ball is None:
            if static:
                self.ball = np.loadtxt('template/2048.xyz')[:, :3]
            else:
                self.ball = np.loadtxt('template/ball2.xyz')[:, :3]
            self.ball = pc_normalize(self.ball)

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

    def read_ball(self,sort=False):
        x = np.loadtxt("template/balls/2048.xyz")
        ball = pc_normalize(x)

        N = ball.shape[0]
        # xx = torch.bmm(x, x.transpose(2,1))
        xx = np.sum(x ** 2, axis=(1)).reshape(N, 1)
        yy = xx.T
        xy = -2 * xx @ yy  # torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy  # [B, N, N]

        # order = np.argsort(dist[1000])[::1]
        # ball = ball[order]

        return ball




    def simple_gen(self,number=50):

        self.build_model_eval()

        x = self.sphere_generator(self.opts.bs)

        cat = str(self.opts.choice).lower()

        epoch = 300
        while not os.path.exists(os.path.join(self.opts.log_dir,self.opts.pretrain_model_G)):
            epoch = epoch - 50
            self.opts.pretrain_model_G = "%d_Chair_G.pth" % epoch

        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")

        all_sample = []

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))
        n_batches = number // self.opts.bs + 1
        for i in tqdm(range(n_batches)):
            with torch.no_grad():
                z = self.noise_generator(bs=self.opts.bs)
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2,1)

            all_sample.append(out_pc)

        sample_pcs = torch.cat(all_sample, dim=0)[:number]
        sample_pcs = sample_pcs.cpu().detach().numpy()

        sample_pcs = normalize_point_cloud(sample_pcs)

        root = os.getcwd()[:5]

        out_folder = root + "/lirh/pointcloud2/PointGeneration/experiments/%s" % (cat)
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        print(out_folder)

        np.save(os.path.join(out_folder, "%s.npy" % (cat)), sample_pcs)

        if False:
            for i in range(number):
                file_name = str(i).zfill(4) + "_%s.xyz" % (cat)
                out_file = os.path.join(out_folder, file_name)
                np.savetxt(out_file, sample_pcs[i], fmt="%.6f")

        print(cat, "Done")

        del self.G



    def draw_correspondense(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)
        ball = np.expand_dims(ball,axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")


        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5


        number = 5
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        for i in range(sample_num):

            noise = np.random.normal(0, 0.2, (number, self.opts.nz))

            color = np.zeros((number+1, self.opts.np, 3))
            color_c = np.squeeze(ball,axis=0)
            color_c = np.minimum(color_c,1.0)
            color_c = np.maximum(color_c,-1.0)

            for j in range(color.shape[0]):
                color[j] = color_c
            title = ["Raw"]
            title.extend(["Sample_%d"% num for num in range(number)])

            noise = np.expand_dims(noise,axis=1)
            noise = np.tile(noise, (1, self.opts.np, 1))

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2,1)

            sample_pcs = out_pc.cpu().detach().numpy()

            sample_pcs = normalize_point_cloud(sample_pcs)

            pcds = np.concatenate([0.5*ball,0.75*sample_pcs],axis=0)

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_correspondense_%s_%d.png" % (current_time, i))
            print(plot_name)
            plot_pcd_three_views_color(plot_name,pcds,title,colors=color)

        del self.G


    def draw_shape_intepolate(self):

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")


        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5
        pcds_list = []
        title_list = []

        alphas = np.arange(1, -0.1, -0.2)

        x = self.sphere_generator(bs=len(alphas))

        for i in range(sample_num):
            raw_noise = np.random.normal(0, self.opts.nv, (2, self.opts.nz))

            title=[]
            noise = np.zeros((len(alphas),self.opts.nz))
            for id, alpha in enumerate(alphas):
                noise[id] = alpha*raw_noise[0] + (1-alpha)*raw_noise[1]
                title.append("%.1f"%alpha)

            noise = np.expand_dims(noise,axis=1)
            noise = np.tile(noise, (1, self.opts.np, 1))

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2,1)

            sample_pcs = out_pc.cpu().detach().numpy()

            sample_pcs = normalize_point_cloud(sample_pcs)

            title_list.append(title)
            pcds_list.append(sample_pcs)

        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        plot_name = os.path.join(plot_folder, "plot_shape_inte_%s.png" % current_time)
        print(plot_name)
        plot_pcd_multi_rows(plot_name, pcds_list, title_list)

        del self.G

    def draw_part_shape_inte(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)


        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 10

        alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
        masks = [0, 512, 1024, 1536, 2048]

        x = np.tile(x, (len(masks), 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        balls = np.tile(np.expand_dims(0.5*ball,axis=0), ((len(masks), 1, 1)))
        scale = 0.75
        for i in range(sample_num):

            #pcds_list = [balls]
            pcds_list = []

            title_list = []
            colors = []
            black = np.zeros((len(masks), self.opts.np, 3))
            red = np.zeros((len(masks), self.opts.np, 3))
            red[:, :, 0] = 1.0


           # colors.append((len(masks), self.opts.np, 3)) # for ball
            colors.append(np.zeros((len(masks), self.opts.np, 3)) ) # for shape inte

            colors.append(np.zeros((len(masks), self.opts.np, 3)) ) # for part inte
            color = np.zeros((len(masks), self.opts.np, 3)) # for part inte with color
            for id, mask in enumerate(masks):
                color[id, mask:] = black[0, mask:]
                color[id, :mask] = red[0, :mask]
            colors.append(color)


            raw_noise = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
            raw_noise = np.tile(raw_noise, (1, self.opts.np, 1))

            ################## for shape inte
            title = []
            noise = np.zeros((len(alphas), self.opts.np, self.opts.nz))
            for id, alpha in enumerate(alphas):
                noise[id] = alpha * raw_noise[0] + (1 - alpha) * raw_noise[1]
                title.append("Shape_%.1f" % alpha)

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)

            shape_pcs = out_pc.cpu().detach().numpy()

            shape_pcs = scale*normalize_point_cloud(shape_pcs)

            pcds_list.append(shape_pcs)
            title_list.append(title)

            ################## for part inte
            title = ["Part_%d" % mask for mask in masks]
            noise = np.zeros((len(masks), self.opts.np, self.opts.nz))
            for id, mask in enumerate(masks):
                noise[id, mask:] = raw_noise[0, mask:]
                noise[id, :mask] = raw_noise[1, :mask]

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)

            part_pcs = out_pc.cpu().detach().numpy()

            part_pcs = scale*normalize_point_cloud(part_pcs)

            title_list.append(title)
            pcds_list.append(part_pcs)

            title_list.append(title)
            pcds_list.append(part_pcs)

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_part_shape_inte_%s_%d.png" % (current_time,i))
            print(plot_name)
            #print(len(colors),len(colors[0]),len(masks))
            plot_pcd_multi_rows_color(plot_name, pcds_list, title_list, colors=colors)

        del self.G


    def draw_part_shape_inte_detail(self):

        ball = self.read_ball()
        x = np.expand_dims(ball, axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 10
        alphas = [1.0, 0.75, 0.5, 0.25, 0.0]
        masks = [0, 512, 1024, 1536, 2048]
        #masks = [2048, 1536, 1024, 512, 0]


        x = np.tile(x, (len(masks), 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        balls = np.tile(np.expand_dims(0.5 * ball, axis=0), ((len(masks), 1, 1)))
        scale = 0.75
        for sample in range(sample_num):

            pcds_list = [balls]
            title = ["%d" % mask for mask in masks]
            title_list = [title]
            colors = []


            black = np.zeros((self.opts.np, 3))
            red = np.zeros((self.opts.np, 3))
            red[:, 0] = 1.0

            # colors.append((len(masks), self.opts.np, 3)) # for ball

            color = np.zeros((len(masks), self.opts.np, 3))  # for part inte with color
            for id, mask in enumerate(masks):
                color[id] = black
                mask = 2048 - mask
                color[id, mask:] = red[mask:]
            colors.append(color)  # for ball
            colors.append(np.zeros((len(masks), self.opts.np, 3)))  # for shape inte
            for i in range(len(alphas)):
                 colors.append(color)  # for part inte
                #colors.append(np.zeros((len(masks), self.opts.np, 3)))  # for part inte


            raw_noise = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
            raw_noise = np.tile(raw_noise, (1, self.opts.np, 1))

            ################## for shape inte
            title = []
            noise = np.zeros((len(alphas), self.opts.np, self.opts.nz))
            for id, alpha in enumerate(alphas):
                noise[id] = alpha * raw_noise[0] + (1 - alpha) * raw_noise[1]
                title.append("Shape_%.1f" % alpha)

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)

            shape_pcs = out_pc.cpu().detach().numpy()

            shape_pcs = scale * normalize_point_cloud(shape_pcs)

            pcds_list.append(shape_pcs)
            title_list.append(title)

            ################## for part inte

            for i, alpha in enumerate(alphas):

                noise = np.zeros((len(masks), self.opts.np, self.opts.nz))
                title = []
                for j, mask in enumerate(masks):
                    title.append("Part_a=%.1f_m=%d" % (alpha, mask))

                    noise[j] = raw_noise[0]
                    mask = 2048 - mask
                    noise[j, mask:] = alpha * raw_noise[0, mask:] + (1 - alpha) * raw_noise[1, mask:]



                with torch.no_grad():
                    z = Variable(torch.Tensor(noise)).cuda()
                    out_pc = self.G(x, z)
                    out_pc = out_pc.transpose(2, 1)

                sample_pcs = out_pc.cpu().detach().numpy()

                sample_pcs = scale * normalize_point_cloud(sample_pcs)

                title_list.append(title)
                pcds_list.append(sample_pcs)

            root = os.getcwd()[:5]

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_part_shape_inte_detail_%s_%d.png" % (current_time, sample))
            print(plot_name)
            plot_pcd_multi_rows_color(plot_name, pcds_list, title_list, colors=colors)

        del self.G


    def draw_part_edit(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()
        number = 1000

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5

        masks = np.arange(0, 1050, 200)
        x = np.tile(x, (len(masks), 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        balls = np.tile(np.expand_dims(0.5*ball,axis=0), ((len(masks), 1, 1)))

        pcds_list = [balls]
        title = ["%d" % mask for mask in masks]
        title_list = [title]
        colors = []
        black = np.zeros((len(masks), self.opts.np, 3))
        red = np.zeros((len(masks), self.opts.np, 3))
        red[:, :, 0] = 1.0
        color = np.zeros((len(masks), self.opts.np, 3))

        for id, mask in enumerate(masks):
            color[id, mask:] = black[0, mask:]
            color[id, :mask] = red[0, :mask]
            colors.append(color)

        for i in range(sample_num):

            raw_noise = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
            raw_noise = np.tile(raw_noise, (1, self.opts.np, 1))

            title = []
            noise = np.zeros((len(masks), self.opts.np, self.opts.nz))
            for id, mask in enumerate(masks):
                noise[id, mask:] = raw_noise[0, mask:]
                noise[id, :mask] = raw_noise[1, :mask]
                title.append("%d" % mask)

            with torch.no_grad():
                z = Variable(torch.Tensor(noise)).cuda()
                out_pc = self.G(x, z)
                out_pc = out_pc.transpose(2, 1)

            sample_pcs = out_pc.cpu().detach().numpy()

            sample_pcs = normalize_point_cloud(sample_pcs)

            title_list.append(title)
            pcds_list.append(sample_pcs)

        root = os.getcwd()[:5]

        plot_path = root + "/lirh/pointcloud2/PointGeneration/experiments/plots/plot_mask"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        plot_name = os.path.join(plot_folder, "plot_mask_%s.png" % current_time)
        print(plot_name)
        plot_pcd_multi_rows_single_color(plot_name, pcds_list, title_list, colors=color)

        del self.G

    def draw_edit_inte(self):

        ball = self.read_ball()
        x = np.expand_dims(ball, axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()
        number = 1000

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 10

        masks = np.arange(0, 1050, 200)
        alphas = np.arange(1, -0.1, -0.2)

        x = np.tile(x, (len(masks), 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        balls = np.tile(np.expand_dims(0.5 * ball, axis=0), ((len(masks), 1, 1)))

        part_intepolate = False
        for sample in range(sample_num):
            pcds_list = [balls]
            title = ["%d" % mask for mask in masks]
            title_list = [title]
            colors = []
            black = np.zeros((len(masks), self.opts.np, 3))
            red = np.zeros((len(masks), self.opts.np, 3))
            red[:, :, 0] = 1.0
            color = np.zeros((len(masks), self.opts.np, 3))

            for id, mask in enumerate(masks):
                color[id, mask:] = black[0, mask:]
                color[id, :mask] = red[0, :mask]
                colors.append(color)


            raw_noise_A = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
            raw_noise_A = np.tile(raw_noise_A, (1, self.opts.np, 1))

            raw_noise_B = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
            raw_noise_B = np.tile(raw_noise_B, (1, self.opts.np, 1))

            for i, alpha in enumerate(alphas):

                noise = np.zeros((len(masks), self.opts.np, self.opts.nz))
                title = []
                for j, mask in enumerate(masks):

                    noise[j, :mask] = alpha * raw_noise_A[0,:mask] + (1 - alpha) * raw_noise_B[0,:mask]

                    if part_intepolate:
                        noise[j, mask:] = raw_noise_A[1,mask:]
                    else:
                        noise[j, mask:] = alpha * raw_noise_A[1, mask:] + (1 - alpha) * raw_noise_B[1, mask:]

                    title.append("a=%.1f_m=%d" % (alpha, mask))

                with torch.no_grad():
                    z = Variable(torch.Tensor(noise)).cuda()
                    out_pc = self.G(x, z)
                    out_pc = out_pc.transpose(2, 1)

                sample_pcs = out_pc.cpu().detach().numpy()

                sample_pcs = 0.75*normalize_point_cloud(sample_pcs)

                title_list.append(title)
                pcds_list.append(sample_pcs)

            root = os.getcwd()[:5]


            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_mask_inte_%s_%d.png" % (current_time,sample))
            print(plot_name)
            plot_pcd_multi_rows_single_color(plot_name, pcds_list, title_list, colors=color)

        del self.G


    def draw_part_flip(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)

        self.build_model_eval()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5


        number = 8
        balls = np.tile(np.expand_dims(0.5 * ball, axis=0), ((number, 1, 1)))
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        idxs = [np.arange(self.opts.np)]
        idxs.append(np.where(ball[:,0]>0))
        idxs.append(np.where(ball[:,0]<0))
        idxs.append(np.where(ball[:,1]>0))
        idxs.append(np.where(ball[:,1]<0))
        idxs.append(np.where(ball[:,2]>0))
        idxs.append(np.where(ball[:,2]<0))
        idxs.append(np.arange(self.opts.np))


        for sample in range(sample_num):
            pcds_list = [balls]
            title =["Raw","Left_Right","Right_Left","UP_Down","Down_UP","Front_Back","Back_Front","-1*Raw"]
            title_list=[title]

            colors = []
            black = np.zeros((self.opts.np, 3))
            red = np.zeros((self.opts.np, 3))
            red[:, 0] = 1.0

            color = np.zeros((number, self.opts.np, 3))
            for id, mask in enumerate(idxs):
                color[id] = black
                if id>0 and id < number-1:
                    color[id, mask] = red[mask]

            for i in range(5):

                raw_noise = np.random.normal(0, 0.2, (1, self.opts.nz))
                raw_noise = np.tile(raw_noise, (self.opts.np, 1))
                flip_noise = -1.0*raw_noise

                noise = np.zeros((number, self.opts.np, self.opts.nz))

                for id, mask in enumerate(idxs):
                    #[2048,1024,512]
                    noise[id] = raw_noise
                    if id > 0:
                        noise[id, mask] = flip_noise[mask]

                with torch.no_grad():
                    z = Variable(torch.Tensor(noise)).cuda()
                    out_pc = self.G(x, z)
                    out_pc = out_pc.transpose(2, 1)

                sample_pcs = out_pc.cpu().detach().numpy()

                sample_pcs = 0.75*normalize_point_cloud(sample_pcs)

                #title_list.append(title1 + title2)
                title_list.append(title)
                pcds_list.append(sample_pcs)

            root = os.getcwd()[:5]

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_folder, "plot_flip_%s_%d.png" % (current_time,sample))
            print(plot_name)
            plot_pcd_multi_rows_single_color(plot_name, pcds_list, title_list, colors=color)

        del self.G

    def draw_part_exchange(self):

        ball = self.read_ball()

        x = np.expand_dims(ball, axis=0)

        self.G = Generator(self.opts)
        self.G.cuda()

        cat = str(self.opts.choice).lower()
        could_load, save_epoch = self.load(self.opts.log_dir)
        if could_load:
            start_epoch = save_epoch
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            exit(0)

        all_sample = []

        # loop for epoch
        start_time = time.time()
        self.G.eval()
        number = 1000

        print(cat, "Start")

        print('# parameters:', sum(param.numel() for param in self.G.parameters()))

        sample_num = 5




        masks = [2048,1024,512][:2]

        number = int(2*len(masks))
        balls = np.tile(np.expand_dims(0.5 * ball, axis=0), ((number, 1, 1)))
        x = np.tile(x, (number, 1, 1))
        x = Variable(torch.Tensor(x)).cuda()

        for sample in range(sample_num):
            pcds_list = [balls]
            title1 = ["%d" % mask for mask in masks]
            title2 = ["%d" % mask for mask in (masks[::-1])]
            title_list = [title1+title2]
            colors = []
            black = np.zeros((self.opts.np, 3))
            red = np.zeros((self.opts.np, 3))
            red[:, 0] = 1.0
            color = np.zeros((number, self.opts.np, 3))

            for id, mask in enumerate(masks):
                color[id, :mask] = black[:mask]
                color[id, mask:] = red[mask:]
                colors.append(color)

            for id, mask in enumerate(masks[::-1]):
                mask = int(2048 - mask)
                color[len(masks)+id, :mask] = red[:mask]
                color[len(masks)+id, mask:] = black[mask:]

                colors.append(color)




            for i in range(5):

                title1 = ["Sample_%d_%d" % (i,mask) for mask in masks]
                title2 = ["Sample_%d_%d" % (i,mask) for mask in masks[::-1]]


                raw_noise = np.random.normal(0, 0.2, (2, 1, self.opts.nz))
                raw_noise = np.tile(raw_noise, (1, self.opts.np, 1))

                noise = np.zeros((number, self.opts.np, self.opts.nz))

                for id, mask in enumerate(masks):
                    #[2048,1024,512]
                    noise[id, :mask] = raw_noise[0,:mask]
                    noise[id, mask:] = raw_noise[1,mask:]

                for id, mask in enumerate(masks[::-1]):
                    #[512,1024,2048]
                    noise[len(masks)+id, mask:] = raw_noise[0, mask:]
                    noise[len(masks)+id, :mask] = raw_noise[1, :mask]


                with torch.no_grad():
                    z = Variable(torch.Tensor(noise)).cuda()
                    out_pc = self.G(x, z)
                    out_pc = out_pc.transpose(2, 1)

                sample_pcs = out_pc.cpu().detach().numpy()

                sample_pcs = 0.75*normalize_point_cloud(sample_pcs)

                #title_list.append(title1 + title2)
                title_list.append(["Full_PC1","Half_PC1_Half_PC2","Half_PC2_Half_PC1","Full_PC2"])

                pcds_list.append(sample_pcs)

            root = os.getcwd()[:5]

            plot_path = root + "/lirh/pointcloud2/PointGeneration/experiments/plots/plot_mask_inte"
            if not os.path.exists(plot_path):
                os.makedirs(plot_path)

            # for i in range(len(pcds_list)):
            #     print(title_list[i])

            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            plot_name = os.path.join(plot_path, "plot_part_exchange_%s_%d.png" % (current_time,sample))
            print(plot_name)
            plot_pcd_multi_rows_single_color(plot_name, pcds_list, title_list, colors=color)

        del self.G

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
                #self.optimizerG.load_state_dict(checkpoint['G_optimizer'])
                G_epoch = checkpoint['G_epoch']
        else:
            print(" [*] Failed to find the pretrain_model_G")
            exit()

        # ----------------- load D -------------------

        print(" [*] Failed to find the pretrain_model_D")
        #exit()

        print(" [*] Success to load model --> {} & {}".format(self.opts.pretrain_model_G, self.opts.pretrain_model_D))
        return True, G_epoch




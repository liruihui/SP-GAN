import numpy as np
import warnings
import h5py
from torch.utils.data import Dataset
from glob import glob
from Common import point_operation
import os
warnings.filterwarnings('ignore')
from torchvision import transforms
from Common import data_utils as d_utils
from Common import point_operation
import torch

def load_h5(h5_filename,num=2048):
    f = h5py.File(h5_filename)
    data = f['poisson_%d'%num][:]
    return data



point_transform = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudRotate(),
        d_utils.PointcloudRotatePerturbation(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        d_utils.PointcloudJitter(),
        #d_utils.PointcloudRandomInputDropout(),
    ]
)

point_transform2 = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
        d_utils.PointcloudScale(),
        d_utils.PointcloudTranslate(),
        #d_utils.PointcloudRandomInputDropout(),
    ]
)

class H5DataLoader(Dataset):
    def __init__(self, opts,augment=False, partition='train'):
        self.opts = opts
        self.num_points = opts.np

        self.con = opts.con
        if self.con:
            cats = ["chair","table","bench"][:opts.cls]
            pcs = []
            labels = []
            for i, cat in enumerate(cats):
                h5_file = os.path.join(opts.data_root, str(self.num_points), str(cat).lower() + ".h5")
                print("---------------h5_file:", h5_file)
                pc = load_h5(h5_file, self.num_points)
                pc = point_operation.normalize_point_cloud(pc)
                label = np.ones((pc.shape[0],))*i
                pcs.append(pc)
                labels.append(label)
            self.data = np.concatenate(pcs,axis=0)
            self.labels = np.concatenate(labels, axis=0).astype(np.int32)
            print(self.labels.shape)
        # elif opts.choice == "animal":
        #     pcs = []
        #     for i in range(5):
        #         h5_file = os.path.join(opts.data_root, str(self.num_points), str(opts.choice).lower(), str(opts.choice).lower() + "_%d.h5"%i)
        #         print("---------------h5_file:", h5_file)
        #         pc = load_h5(h5_file, self.num_points)
        #         pc = point_operation.normalize_point_cloud(pc)
        #         pcs.append(pc)
        #
        #     self.data = np.concatenate(pcs,axis=0)

        elif opts.choice == "animal_all":
            pcs = []
            cats = ["animal-pose", "animal-deform"]
            for cat in cats:
                h5_file = os.path.join(opts.data_root, str(self.num_points), str(cat).lower() + ".h5")
                print("---------------h5_file:", h5_file)
                pc = load_h5(h5_file, self.num_points)
                pc = point_operation.normalize_point_cloud(pc)
                pcs.append(pc)

            self.data = np.concatenate(pcs, axis=0)
        elif opts.choice == "bottle":
            pcs = []
            cats = ["bottle","jar","pot"]
            for cat in cats:
                h5_file = os.path.join(opts.data_root, str(self.num_points), str(cat).lower() + ".h5")
                print("---------------h5_file:", h5_file)
                pc = load_h5(h5_file, self.num_points)
                pc = point_operation.normalize_point_cloud(pc)
                pcs.append(pc)

            self.data = np.concatenate(pcs,axis=0)

        else:
            h5_file = os.path.join(opts.data_root, str(self.num_points), str(opts.choice).lower()+".h5")
            print("---------------h5_file:",h5_file)
            self.data = load_h5(h5_file,self.num_points)
            self.labels = None

        self.data = self.opts.scale * point_operation.normalize_point_cloud(self.data)
        self.augment = augment
        self.partition = partition


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point_set = self.data[index][:self.num_points,:3].copy()
        np.random.shuffle(point_set)

        if self.augment:
            point_set = point_operation.rotate_point_cloud_and_gt(point_set)
            point_set = point_operation.random_scale_point_cloud_and_gt(point_set)
        point_set = point_set.astype(np.float32)

        if self.con:
            label = self.labels[index].copy()
            return torch.Tensor(point_set), torch.Tensor(label)
        return torch.from_numpy(point_set)

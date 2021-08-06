import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from numpy.linalg import norm
import sys,os
import torch.nn.functional as F
sys.path.append(os.path.join(os.getcwd(),"metrics"))
from pointops import pointops_util
# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
#from StructuralLosses.match_cost import match_cost
#from StructuralLosses.nn_distance import nn_distance
from CD_EMD.emd_ import emd_module
from CD_EMD.cd.chamferdist import ChamferDistance as CD

from evaluation.pointnet import PointNetCls

from scipy.linalg import sqrtm


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self,preds,gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2


    def batch_pairwise_dist(self,x,y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        #xx = torch.bmm(x, x.transpose(2,1))
        xx = torch.sum(x ** 2, dim=2, keepdim=True)
        yy = torch.sum(y ** 2, dim=2, keepdim=True)
        xy = -2 * torch.bmm(x, y.permute(0, 2, 1))
        dist = xy + xx + yy.permute(0, 2, 1)  # [B, N, N]
        return dist


def get_activations(pointclouds, model, batch_size=100, dims=512):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- pointcloud       : pytorch Tensor of pointclouds.
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : If set to device, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    N = pointclouds.size(0)

    n_batches = N // batch_size

    if N % batch_size != 0:
        n_batches = n_batches + 1
    n_used_imgs = n_batches * batch_size

    #pred_arr = np.empty((pointclouds.size(0) , dims))

    pred_acts = []

    #pointclouds = pointclouds.transpose(1, 2)
    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size,N)

        pointcloud_batch = pointclouds[start:end]

        pointcloud_batch = pointcloud_batch.to(pointclouds)
        with torch.no_grad():
            _, actv = model(pointcloud_batch)
        actv = actv.squeeze(1)

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        #pred_acts.append(actv.cpu().data.numpy())
        pred_acts.append(actv)
        #pred_arr[start:end] = actv#actv.cpu().data.numpy().reshape(batch_size, -1)

    pred_acts = torch.cat(pred_acts,dim=0)
    #pred_acts = np.concatenate(pred_acts,axis=0)
    return pred_acts


def _load_pretrain(pretrain, model,verbose=True):
    state_dict = torch.load(pretrain, map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        if key[:10] == 'classifier':
            continue
        new_state_dict[name] = val
    model.load_state_dict(new_state_dict)
    if verbose:
        print(f"Load model from {pretrain}")
    return model

def load_evaluate_model(batch_size=16, model_type="Feat_CLS_1024", dims=512):
    from evaluation.AutoEncoder import ClassificationNet, ReconstructionNet

    class Opts(object):
        def __init__(self):
            self.encoder = "dgcnn_cls"
            self.task = "rec"
            self.dropout = 0.5
            self.k = 40
            self.feat_dims = 1024
            self.num_points = 2048
            self.num_cls = 55
            self.shape = "plane"

    opts = Opts()
    if model_type.startswith("Feat_AE"):
        if model_type == "Feat_AE_Single":
            pretrained_path = 'log/20210101-1818/model_AE_last.pth'
        elif model_type == "Feat_AE":
            pretrained_path = 'evaluation/models/Reconstruct_dgcnn_cls_k20_plane.pkl'
        elif model_type == "Feat_AE_foldnet":
            pretrained_path = 'evaluation/models/Reconstruct_foldnet_plane.pkl'
            opts.encoder = "foldnet"
            opts.k = 16

        opts.task = "rec"
        model = ReconstructionNet(opts)
        #model.load_state_dict(torch.load(pretrained_path))
        model = _load_pretrain(pretrained_path,model)
        model = model.cuda()
    else:
        multi = None
        if model_type.startswith("Feat_CLS_1024"):
            pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k40_1024_b32.pkl'
            opts.k = 40
            opts.feat_dims = 1024

            if model_type ==  "Feat_CLS_1024_max" :
                multi = "max"
            elif model_type == "Feat_CLS_1024_max_avg":
                multi = "max_avg"
            elif model_type == "Feat_CLS_1024_all":
                multi = "all"
            else:
                multi = None
        else:
            pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k20.pkl'
            opts.k = 20
            opts.feat_dims = 512
        opts.task = "cls"
        # pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k40_1024_b32.pkl'
        # opts.feat_dims = 1024
        model = ClassificationNet(opts,eval=True,multi=multi)
        #model.load_state_dict(torch.load(pretrained_path))
        model = _load_pretrain(pretrained_path,model)

    return model

    # del model
    #
    # if pc2 is not None:
    #     return pc_acts1, pc_acts2
    #
    # return pc_acts1

def extract_acts_with_model(pcs,model,batch_size=16):

    with torch.no_grad():
        pc_acts = []
        for pc in pcs:
            # pc = pc.transpose(2, 1)
            pc_act = get_activations(pc, model, batch_size=batch_size)
            # pc_act = torch.from_numpy(pc_act).float().cuda()
            pc_acts.append(pc_act)

    return pc_acts

    # del model
    #
    # if pc2 is not None:
    #     return pc_acts1, pc_acts2
    #
    # return pc_acts1

def extract_acts(pcs, batch_size=16, model_type="AE", dims=512):
    from evaluation.AutoEncoder import ClassificationNet, ReconstructionNet
    import argparse
    parser = argparse.ArgumentParser('AE')
    parser.add_argument('--choice', default='chair', help='chair,airplane,car,all')
    parser.add_argument('--task', type=str, default='rec', metavar='N', choices=['rec', 'cls'],
                        help='Experiment task, [reconstruct, classify]')
    parser.add_argument('--encoder', type=str, default='dgcnn_cls', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--feat_dims', type=int, default=dims, metavar='N', help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='plane', metavar='N', choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--num_cls', type=int, default=55, help='Num of catagory')
    parser.add_argument('--num_points', type=int, default=2048, help='Num of points to use')

    opts = parser.parse_args()
    if model_type.startswith("Feat_AE"):
        if model_type == "Feat_AE_Single":
            pretrained_path = 'log/20210101-1818/model_AE_last.pth'
        elif model_type == "Feat_AE":
            pretrained_path = 'evaluation/models/Reconstruct_dgcnn_cls_k20_plane.pkl'
        elif model_type == "Feat_AE_foldnet":
            pretrained_path = 'evaluation/models/Reconstruct_foldnet_plane.pkl'
            opts.encoder = "foldnet"
            opts.k = 16

        opts.task = "rec"
        model = ReconstructionNet(opts)
        #model.load_state_dict(torch.load(pretrained_path))
        model = _load_pretrain(pretrained_path,model)
        model = model.cuda()
    # elif model_type == "CLS":
    #     PointNet_pretrained_path = 'evaluation/cls_model_39.pth'
    #     model = PointNetCls(k=16,is_eval=True)
    #     model.load_state_dict(torch.load(PointNet_pretrained_path))
    #     print(PointNet_pretrained_path)
    #     pc = pc.transpose(2,1)
    #     model.cuda()
    else:
        multi = None
        if model_type.startswith("Feat_CLS_1024"):
            pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k40_1024_b32.pkl'
            opts.k = 40
            opts.feat_dims = 1024

            if model_type ==  "Feat_CLS_1024_max" :
                multi = "max"
            elif model_type == "Feat_CLS_1024_max_avg":
                multi = "max_avg"
            elif model_type == "Feat_CLS_1024_all":
                multi = "all"
            else:
                multi = None
        else:
            pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k20.pkl'
            opts.k = 20
            opts.feat_dims = 512
        opts.task = "cls"
        # pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k40_1024_b32.pkl'
        # opts.feat_dims = 1024
        model = ClassificationNet(opts,eval=True,multi=multi)
        #model.load_state_dict(torch.load(pretrained_path))
        model = _load_pretrain(pretrained_path,model)
        model = model.cuda()

    with torch.no_grad():

        with torch.no_grad():
            pc_acts = []
            for pc in pcs:
                # pc = pc.transpose(2, 1)
                pc_act = get_activations(pc, model, batch_size=batch_size, dims=dims)
                # pc_act = torch.from_numpy(pc_act).float().cuda()
                pc_acts.append(pc_act)

        return pc_acts

    # del model
    #
    # if pc2 is not None:
    #     return pc_acts1, pc_acts2
    #
    # return pc_acts1


def extract_acts_bt(pcs, batch_size=16, model_type="AE", dims=512):
    from evaluation.AutoEncoder import ClassificationNet, ReconstructionNet
    import argparse
    parser = argparse.ArgumentParser('AE')
    parser.add_argument('--choice', default='chair', help='chair,airplane,car,all')
    parser.add_argument('--task', type=str, default='rec', metavar='N', choices=['rec', 'cls'],
                        help='Experiment task, [reconstruct, classify]')
    parser.add_argument('--encoder', type=str, default='dgcnn_cls', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--feat_dims', type=int, default=dims, metavar='N', help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='plane', metavar='N', choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--num_cls', type=int, default=55, help='Num of catagory')
    parser.add_argument('--num_points', type=int, default=2048, help='Num of points to use')

    opts = parser.parse_args()
    if model_type == "AE":
        #pretrained_path = 'log/20210101-1818/model_AE_last.pth'
        pretrained_path = 'evaluation/models/Reconstruct_dgcnn_cls_k20_plane.pkl'
        opts.task = "rec"
        model = ReconstructionNet(opts)#.cuda()
        #model.load_state_dict(torch.load(pretrained_path))
        #print(f"Load model from {pretrained_path}")
        model = _load_pretrain(pretrained_path,model)
        model = model.cuda()

    # elif model_type == "CLS":
    #     PointNet_pretrained_path = 'evaluation/cls_model_39.pth'
    #     model = PointNetCls(k=16,is_eval=True)
    #     model.load_state_dict(torch.load(PointNet_pretrained_path))
    #     print(PointNet_pretrained_path)
    #     model.cuda()

    elif model_type =="Fold_AE":
        pretrained_path = 'evaluation/models/Reconstruct_foldnet_plane.pkl'
        opts.task = "rec"
        opts.encoder = "foldnet"
        opts.k = 16
        model = ReconstructionNet(opts)  # .cuda()
        # model.load_state_dict(torch.load(pretrained_path))
        # print(f"Load model from {pretrained_path}")
        model = _load_pretrain(pretrained_path, model)
        model = model.cuda()
    else:
        pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k20.pkl'
        opts.task = "cls"
        pretrained_path = 'evaluation/models/Classify_dgcnn_cls_k40_1024_b32.pkl'
        opts.feat_dims = 1024
        model = ClassificationNet(opts,eval=True)
        #model.load_state_dict(torch.load(pretrained_path))
        model = _load_pretrain(pretrained_path,model)
        model = model.cuda()

    with torch.no_grad():
        pc_acts = []
        for pc in pcs:
            #pc = pc.transpose(2, 1)
            pc_act = get_activations(pc,model,batch_size=batch_size,dims=dims)
            #pc_act = torch.from_numpy(pc_act).float().cuda()
            pc_acts.append(pc_act)

    return pc_acts

###############################modules###############################
def distChamferCUDA(x, y):
    cd = CD()
    cd0, cd1, _, _ = cd(x, y)
    return cd0, cd1
    #return nn_distance(x, y)

def CD_dist(x, y):
    dl, dr = distChamferCUDA(x, y)
    dist = (dl.mean(dim=1) + dr.mean(dim=1))
    return dist

def emd_approx2(x, y):
    EMD = emd_module.emdModule()
    dist, ass = EMD(x, y, 0.005, 300)

    return dist


def compute_mean_covariance(points):
    bs, ch, nump = points.size()
    # ----------------------------------------------------------------
    mu = points.mean(dim=-1, keepdim=True)  # Bx3xN -> Bx3x1
    # ----------------------------------------------------------------
    tmp = points - mu.repeat(1, 1, nump)    # Bx3xN - Bx3xN -> Bx3xN
    tmp_transpose = tmp.transpose(1, 2)     # Bx3xN -> BxNx3
    covariance = torch.bmm(tmp, tmp_transpose)
    covariance = covariance / nump
    return mu, covariance   # Bx3x1 Bx3x3




def emd_approx(sample, ref):
    # B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
    #
    # # import ipdb
    # # ipdb.set_trace()
    #
    # assert N == N_ref, "Not sure what would EMD do in this case"
    # emd = match_cost(sample, ref)  # (B,)
    # emd_norm = emd / float(N)  # (B,)
    # return emd_norm
    EMD = emd_module.emdModule()
    dist, ass = EMD(sample, ref, 0.005, 300)

    return dist.mean()

def get_voxel_occ_dist(all_clouds, clouds_flag='gen', res=28, bound=0.5, bs=128, warning=True):
    if np.any(np.fabs(all_clouds) > bound) and warning:
        print('{} clouds out of cube bounds: [-{}; {}]'.format(clouds_flag, bound, bound))

    n_nans = np.isnan(all_clouds).sum()
    if n_nans > 0:
        print('{} NaN values in point cloud tensors.'.format(n_nans))

    p2v_dist = np.zeros((res, res, res), dtype=np.uint64)

    step = 1. / res
    v_bs = -0.5 + np.arange(res + 1) * step

    nbs = all_clouds.shape[0] // bs + 1
    for i in range(nbs):
        clouds = all_clouds[bs * i:bs * (i + 1)]

        preiis = clouds[:, :, 0].reshape(1, -1)
        preiis = np.logical_and(v_bs[:28].reshape(-1, 1) <= preiis, preiis < v_bs[1:].reshape(-1, 1))
        iis = preiis.argmax(0)
        iis_values = preiis.sum(0) > 0

        prejjs = clouds[:, :, 1].reshape(1, -1)
        prejjs = np.logical_and(v_bs[:28].reshape(-1, 1) <= prejjs, prejjs < v_bs[1:].reshape(-1, 1))
        jjs = prejjs.argmax(0)
        jjs_values = prejjs.sum(0) > 0

        prekks = clouds[:, :, 2].reshape(1, -1)
        prekks = np.logical_and(v_bs[:28].reshape(-1, 1) <= prekks, prekks < v_bs[1:].reshape(-1, 1))
        kks = prekks.argmax(0)
        kks_values = prekks.sum(0) > 0

        values = np.uint64(np.logical_and(np.logical_and(iis_values, jjs_values), kks_values))
        np.add.at(p2v_dist, (iis, jjs, kks), values)

    return np.float64(p2v_dist) / p2v_dist.sum()



def JSD(clouds1, clouds2, clouds1_flag='gen', clouds2_flag='ref', warning=True):
    dist1 = get_voxel_occ_dist(clouds1, clouds_flag=clouds1_flag, warning=warning)
    dist2 = get_voxel_occ_dist(clouds2, clouds_flag=clouds2_flag, warning=warning)
    return entropy((dist1 + dist2).flatten() / 2.0, base=2) - \
        0.5 * (entropy(dist1.flatten(), base=2) + entropy(dist2.flatten(), base=2))



def COV(dists, axis=1):
    return float(dists.min(axis)[1].unique().shape[0]) / float(dists.shape[axis])


def MMD(dists, axis=1):
    return float(dists.min((axis + 1) % 2)[0].mean().float())


def KNN(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((-torch.ones(n0), torch.ones(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, 0).float()
    pred[torch.eq(pred, 0)] = -1.

    return float(torch.eq(label, pred).float().mean())

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def FPD(pc1_acts, pc2_acts):
    """Calculates the FPD of two pointclouds"""

    m1, s1 = np.mean(pc1_acts, axis=0), np.cov(pc1_acts, rowvar=False)
    m2, s2 = np.mean(pc2_acts, axis=0), np.cov(pc2_acts, rowvar=False)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value
###############################functions###############################

def pairwise_dists(sample_pcs, ref_pcs, batch_size, dist_type="l2"):
    if dist_type == "CD":
        pair_dists = pairwise_CD(sample_pcs, ref_pcs, batch_size)
    elif dist_type == "CD_M" or dist_type == "CD_C":
        pair_dists = pairwise_local_CD(sample_pcs, ref_pcs, batch_size,dist_type)
    elif dist_type == "EMD":
        pair_dists = pairwise_EMD(sample_pcs, ref_pcs, batch_size)
    else:
        dist_type = "l2"
        pair_dists = pairwise_simple(sample_pcs, ref_pcs, batch_size,dist_type)

    return pair_dists


def pairwise_simple(sample_pcs, ref_pcs, batch_size, dist_type="l2"):
    N1 = sample_pcs.shape[0]
    N2 = ref_pcs.shape[0]
    all_dist = []
    dim = sample_pcs.shape[-1]
    iterator = range(N1)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        dists = []
        for ref_b_start in range(0, N2, batch_size):
            ref_b_end = min(N2, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_b_end - ref_b_start#ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, dim).expand(batch_size_ref, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            if dist_type == "l2":
                dist = torch.sum((sample_batch_exp - ref_batch)**2,dim=-1)
            if dist_type == "l1":
                dist = torch.sum(torch.abs(sample_batch_exp - ref_batch), dim=-1)

            dists.append(dist.unsqueeze(0))

        dists = torch.cat(dists, dim=1)

        all_dist.append(dists)

    all_dist = torch.cat(all_dist, dim=0)  # N_sample, N_ref

    return all_dist


def local_CD(pt1, pt2):
    B,N,C = pt1.size()
    # pt1: Bx3xM    pt2: Bx3XN      (N > M)
    # print('pt1: {}      pt2: {}'.format(pt1.size(), pt2.size()))
    new_xyz = pt1

    K=8
    group = pointops_util.Gen_QueryAndGroupXYZ(radius=None, nsample=K, use_xyz=False)

    g_xyz1 = group(pt1, new_xyz)  # Bx3xMxK
    g_xyz2 = group(pt2, new_xyz)  # Bx3xMxK

    g_xyz1 = g_xyz1.transpose(1, 2).contiguous().view(-1, 3, K)  # Bx3xMxK -> BxMx3xK -> (BM)x3xK
    g_xyz2 = g_xyz2.transpose(1, 2).contiguous().view(-1, 3, K)  # Bx3xMxK -> BxMx3xK -> (BM)x3xK
    mu1, var1 = compute_mean_covariance(g_xyz1)
    mu2, var2 = compute_mean_covariance(g_xyz2)
    mu1 = mu1.view(B, -1, 3)
    mu2 = mu2.view(B, -1, 3)

    var1 = var1.view(B, -1, 9)
    var2 = var2.view(B, -1, 9)
    # like_mu12 = CD_dist(mu1, mu2)
    #
    # like_var12 = CD_dist(var1, var2)
    chamfer_loss = ChamferLoss()
    like_mu12 = chamfer_loss(mu1, mu2) / float(N)

    like_var12 = chamfer_loss(var1, var2) / float(N)
    # print('mu: {} var: {}'.format(like_mu12.item(), like_var12.item()))

    return like_mu12, like_var12

def pairwise_local_CD(sample_pcs, ref_pcs, batch_size,dist_type):
    N1 = sample_pcs.shape[0]
    N2 = ref_pcs.shape[0]
    all_cd = []
    iterator = range(N1)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N2, batch_size):
            ref_b_end = min(N2, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            mu, var = local_CD(sample_batch_exp, ref_batch)
            if dist_type == "CD_M":
                cd_lst.append(mu.view(1, -1))
            else:
                cd_lst.append(var.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd

def pairwise_CD(sample_pcs, ref_pcs, batch_size=16):
    N1 = sample_pcs.shape[0]
    N2 = ref_pcs.shape[0]
    all_cd = []
    iterator = range(N1)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N2, batch_size):
            ref_b_end = min(N2, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref

    return all_cd


def pairwise_EMD(sample_pcs, ref_pcs, batch_size=16):
    N1 = sample_pcs.shape[0]
    N2 = ref_pcs.shape[0]
    all_emd = []
    for i in range(N1):
        sample_batch = sample_pcs[i]

        emd_lst = []
        for ref_b_start in range(0, N2, batch_size):
            ref_b_end = min(N2, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()


            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        emd_lst = torch.cat(emd_lst, dim=1)
        all_emd.append(emd_lst)

    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_emd

def pairwise_EMD_CD(sample_pcs, ref_pcs, batch_size):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def filtering(sample_pcs):
    sample_pcs = sample_pcs.cpu().numpy()
    sample_pcs_inds = set(np.arange(sample_pcs.shape[0]))
    axis = (1, 2) if len(sample_pcs.shape) == 3 else (1)
    nan_gen_clouds_inds = set(np.isnan(sample_pcs).sum(axis=axis).nonzero()[0])
    sample_pcs_inds = list(sample_pcs_inds - nan_gen_clouds_inds)
    dup_gen_clouds_inds = np.random.choice(sample_pcs_inds, size=len(nan_gen_clouds_inds))
    sample_pcs[list(nan_gen_clouds_inds)] = sample_pcs[dup_gen_clouds_inds]
    sample_pcs = torch.from_numpy(sample_pcs).cuda()

    return sample_pcs

def compute_all_metrics_train(sample_pcs, ref_pcs, model, batch_size=16, dist_type="CD",use_FPD_JSD=False):

    ss_dist = pairwise_dists(sample_pcs, sample_pcs, batch_size,dist_type)
    rr_dist = pairwise_dists(ref_pcs, ref_pcs, batch_size,dist_type)
    sr_dist = pairwise_dists(sample_pcs, ref_pcs, batch_size,dist_type)

    metrics = {}

    cd_covs = COV(sr_dist)
    cd_mmds = MMD(sr_dist)
    cd_mmd_refs = MMD(sr_dist.t())
    cd_1nns = KNN(ss_dist, sr_dist, rr_dist, 1)

    fpd = 0
    jsd = 0
    if use_FPD_JSD:
        bufs = extract_acts_with_model([sample_pcs, ref_pcs], model, batch_size=batch_size)
        # sample_buf, ref_buf = bufs[0], bufs[1]
        # fpd = FPD(sample_buf.cpu().numpy(), ref_buf.cpu().numpy())
        jsd = JSD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy(),
                  clouds1_flag='gen', clouds2_flag='ref', warning=False)

    metrics = {
        "JSD": jsd,
        "COV": cd_covs,
        "MMD": cd_mmds,
        "MMD_t": cd_mmd_refs,
        "1NN": cd_1nns,
        "FPD": fpd,
    }

    return metrics


def compute_all_metrics(sample_pcs, ref_pcs, batch_size=16, dist_type="CD"):

    if False:
        sample_pcs = filtering(sample_pcs)
        ref_pcs = filtering(ref_pcs)

    ss_dist = pairwise_dists(sample_pcs, sample_pcs, batch_size,dist_type)
    rr_dist = pairwise_dists(ref_pcs, ref_pcs, batch_size,dist_type)
    sr_dist = pairwise_dists(sample_pcs, ref_pcs, batch_size,dist_type)

    metrics = {}

    cd_covs = COV(sr_dist)
    cd_mmds = MMD(sr_dist)
    cd_1nns = KNN(ss_dist, sr_dist, rr_dist, 1)
    cd_6nns = KNN(ss_dist, sr_dist, rr_dist, 6)

    fpd = 0.0
    jsd = 0.0
    if len(sample_pcs.shape) == 2:
        fpd = FPD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
    else:
        jsd = JSD(sample_pcs.cpu().numpy(), ref_pcs.cpu().numpy(),
                  clouds1_flag='gen', clouds2_flag='ref', warning=False)

    metrics = {
        "JSD": jsd,
        "COV": cd_covs,
        "MMD": cd_mmds,
        "1NN": cd_1nns,
        "6NN": cd_6nns,
        "FPD": fpd,
    }

    return metrics


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s







if __name__ == "__main__":
    B, N = 2, 10
    x = torch.rand(B, N, 3)
    y = torch.rand(B, N, 3)

    distChamfer = CD_loss()
    min_l, min_r = distChamfer(x.cuda(), y.cuda())
    print(min_l.shape)
    print(min_r.shape)

    l_dist = min_l.mean().cpu().detach().item()
    r_dist = min_r.mean().cpu().detach().item()
    print(l_dist, r_dist)

import torch
import numpy as np
import warnings
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from numpy.linalg import norm
import sys,os
import torch.nn.functional as F
#from Common.Const import GPU
from torch.autograd import Variable, grad
sys.path.append(os.path.join(os.getcwd(),"metrics"))
from pointops import pointops_util
# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
# from StructuralLosses.match_cost import match_cost
# from StructuralLosses.nn_distance import nn_distance
from torch.autograd import Variable
from Common.modules import pairwise_dist
from torch.distributions import Beta
from CD_EMD.emd_ import emd_module
from CD_EMD.cd.chamferdist import ChamferDistance as CD
import functools
from numpy import ones,zeros



def dist_o2l(p1, p2):
    # distance from origin to the line defined by (p1, p2)
    p12 = p2 - p1
    u12 = p12 / np.linalg.norm(p12)
    l_pp = np.dot(-p1, u12)
    pp = l_pp*u12 + p1
    return np.linalg.norm(pp)

def para_count(models):
    count = 0
    for model in models:
        count +=  sum(param.numel() for param in model.parameters())
    return count

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



# # Import CUDA version of CD, borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
# try:
#     from . chamfer_distance_ext.dist_chamfer import chamferDist
#     CD = chamferDist()
#     def distChamferCUDA(x,y):
#         return CD(x,y,gpu)
# except:

class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss

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
        yy = torch.bmm(y, y.transpose(2,1))
        zz = torch.bmm(x, y.transpose(2,1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        #brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2,1) + ry - 2*zz)

        return P

# def batch_pairwise_dist(self,x,y):
#
#    bs, num_points_x, points_dim = x.size()
#     _, num_points_y, _ = y.size()
#
#     xx = torch.sum(x ** 2, dim=2, keepdim=True)
#     yy = torch.sum(y ** 2, dim=2, keepdim=True)
#     yy = yy.permute(0, 2, 1)
#
#     xi = -2 * torch.bmm(x, y.permute(0, 2, 1))
#     dist = xi + xx + yy  # [B, N, N]
#     return dist


def dist_simple(x,y,loss="l2"):
    if loss == "l2":
        dist = torch.sum((x - y) ** 2, dim=-1).sum(dim=1).float()
    else:
        dist = torch.sum(torch.abs(x - y), dim=-1).sum(dim=1).float()

    return dist.mean()


def distChamferCUDA(x, y):
    cd = CD()
    cd0, cd1, _, _ = cd(x, y)
    return nn_distance(x, y)


def emd_approx(sample, ref):
    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)

    # import ipdb
    # ipdb.set_trace()

    assert N == N_ref, "Not sure what would EMD do in this case"
    emd = match_cost(sample, ref)  # (B,)
    emd_norm = emd / float(N)  # (B,)
    return emd_norm

def CD_loss(x, y):
    dists_forward, dists_backward = nn_distance(x, y)

    dists_forward = torch.mean(dists_forward,dim=1)
    dists_backward = torch.mean(dists_backward,dim=1)

    cd_dist = torch.mean(dists_forward+dists_backward)
    return cd_dist


def EMD_loss(sample, ref):
    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)

    # import ipdb
    # ipdb.set_trace()

    assert N == N_ref, "Not sure what would EMD do in this case"
    emd = match_cost(sample, ref)  # (B,)
    emd_norm = emd / float(N)  # (B,)
    return emd_norm


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


def get_local_pair(pt1, pt2):
    pt1_batch, pt1_N, pt1_M = pt1.size()
    pt2_batch, pt2_N, pt2_M = pt2.size()
    # pt1: Bx3xM    pt2: Bx3XN      (N > M)
    # print('pt1: {}      pt2: {}'.format(pt1.size(), pt2.size()))
    new_xyz = pt1.transpose(1, 2).contiguous()  # Bx3xM -> BxMx3
    pt1_trans = pt1.transpose(1, 2).contiguous()  # Bx3xM -> BxMx3
    pt2_trans = pt2.transpose(1, 2).contiguous()  # Bx3xN -> BxNx3

    K=20
    group = pointops_util.Gen_QueryAndGroupXYZ(radius=None, nsample=K, use_xyz=False)

    g_xyz1 = group(pt1_trans, new_xyz)  # Bx3xMxK
    # print('g_xyz1: {}'.format(g_xyz1.size()))
    g_xyz2 = group(pt2_trans, new_xyz)  # Bx3xMxK
    # print('g_xyz2: {}'.format(g_xyz2.size()))

    g_xyz1 = g_xyz1.transpose(1, 2).contiguous().view(-1, 3, K)  # Bx3xMxK -> BxMx3xK -> (BM)x3xK
    # print('g_xyz1: {}'.format(g_xyz1.size()))
    g_xyz2 = g_xyz2.transpose(1, 2).contiguous().view(-1, 3, K)  # Bx3xMxK -> BxMx3xK -> (BM)x3xK
    # print('g_xyz2: {}'.format(g_xyz2.size()))
    # print('====================== FPS ========================')
    # print(pt1.shape,g_xyz1.shape)
    # print(pt2.shape,g_xyz2.shape)
    mu1, var1 = compute_mean_covariance(g_xyz1)
    mu2, var2 = compute_mean_covariance(g_xyz2)
    # print('mu1: {} var1: {}'.format(mu1.size(), var1.size()))
    # print('mu2: {} var2: {}'.format(mu2.size(), var2.size()))

    # --------------------------------------------------
    # like_mu12 = self.shape_loss_fn(mu1, mu2)
    # like_var12 = self.shape_loss_fn(var1, var2)
    # ----------------------------------------------------
    # =========$$$  CD loss   $$$===============

    # print("p1,p2:",pt1.shape,pt2.shape)
    # print("mu2:",mu1.shape,mu2.shape,pt1_batch,pt1_N,pt1_M)
    mu1 = mu1.view(pt1_batch, -1, 3)
    mu2 = mu2.view(pt2_batch, -1, 3)

    var1 = var1.view(pt1_batch, -1, 9)
    var2 = var2.view(pt2_batch, -1, 9)
    chamfer_loss = ChamferLoss()
    like_mu12 = chamfer_loss(mu1, mu2) / float(pt1_M)

    like_var12 = chamfer_loss(var1, var2) / float(pt1_M)

    # print('mu: {} var: {}'.format(like_mu12.item(), like_var12.item()))

    return like_mu12, like_var12


# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=False, reduced=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'MMD-CD': cd,
        'MMD-EMD': emd,
    }
    return results


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
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

            if accelerated_cd:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
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


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=False):
    results = {}

    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })

    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)

    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })

    return results



def compute_all_metrics2(sample_pcs, ref_pcs, normalize=False):
    from Common.point_operation import normalize_point_cloud

    gen_clouds_buf = sample_pcs
    ref_clouds_buf = ref_pcs
    if normalize:
        gen_clouds_buf = gen_clouds_buf.cpu().numpy()
        # gen_clouds_inds = set(np.arange(gen_clouds_buf.shape[0]))
        # nan_gen_clouds_inds = set(np.isnan(gen_clouds_buf).sum(axis=(1, 2)).nonzero()[0])
        # gen_clouds_inds = list(gen_clouds_inds - nan_gen_clouds_inds)
        # dup_gen_clouds_inds = np.random.choice(gen_clouds_inds, size=len(nan_gen_clouds_inds))
        # gen_clouds_buf[list(nan_gen_clouds_inds)] = gen_clouds_buf[dup_gen_clouds_inds]
        gen_clouds_buf = normalize_point_cloud(gen_clouds_buf)
        gen_clouds_buf = torch.from_numpy(gen_clouds_buf).cuda()

    gg_cds = pairwise_CD(gen_clouds_buf, gen_clouds_buf)
    tt_cds = pairwise_CD(ref_clouds_buf, ref_clouds_buf)
    gt_cds = pairwise_CD(gen_clouds_buf, ref_clouds_buf)

    metrics = {}
    jsd = JSD(gen_clouds_buf.cpu().numpy(), ref_clouds_buf.cpu().numpy(),
              clouds1_flag='gen', clouds2_flag='ref', warning=False)
    cd_covs = COV(gt_cds)
    cd_mmds = MMD(gt_cds)
    cd_1nns = KNN(gg_cds, gt_cds, tt_cds, 1)

    metrics = {
        "JSD": jsd,
        "COV-CD": cd_covs,
        "MMD-CD": cd_mmds,
        "1NN-CD": cd_1nns,
    }

    return metrics


def f_score(predicted_clouds, true_clouds, threshold=0.001):
    ld, rd = distChamferCUDA(predicted_clouds, true_clouds)
    precision = 100. * (rd < threshold).float().mean(1)
    recall = 100. * (ld < threshold).float().mean(1)
    return 2. * precision * recall / (precision + recall + 1e-7)


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


def pairwise_CD(clouds1, clouds2, bs=2048):
    N1 = clouds1.shape[0]
    N2 = clouds2.shape[0]

    cds = torch.from_numpy(np.zeros((N1, N2), dtype=np.float32)).cuda()

    for i in range(N1):
        clouds1_i = clouds1[i]

        if bs < N1:
            for j_l in range(0, N2, bs):
                j_u = min(N2, j_l + bs)
                clouds2_js = clouds2[j_l:j_u]

                clouds1_is = clouds1_i.unsqueeze(0).expand(j_u - j_l, -1, -1)
                clouds1_is = clouds1_is.contiguous()

                dl, dr = distChamferCUDA(clouds1_is, clouds2_js)
                cds[i, j_l:j_u] = dl.mean(dim=1) + dr.mean(dim=1)

        else:
            clouds1_is = clouds1_i.unsqueeze(0).expand(N1, -1, -1)
            clouds1_is = clouds1_is.contiguous()

            dl, dr = distChamferCUDA(clouds1_is, clouds2)
            cds[i] = dl.mean(dim=1) + dr.mean(dim=1)

    return cds

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

#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))

ZERO = 0.1
ONE = 0.9


def smooth_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random(B) + ran[0]

#y = ones((n_samples, 1))
# example of smoothing class=1 to [0.7, 1.2
def smooth_positive_labels(B,ran=[0.9,1.0]):
    #return y - 0.3 + (np.random.random(y.shape) * 0.5)
    return (ran[1]-ran[0])*np.random.random((B,)) + ran[0]

# example of smoothing class=0 to [0.0, 0.3]
#y = zeros((n_samples, 1))
def smooth_negative_labels(B,ran=[0.0,0.1]):
    #return y + np.random.random(y.shape) * 0.3
    return (ran[1]-ran[0])*np.random.random((B,)) + ran[0]


# randomly flip some labels
#y = ones((n_samples, 1))
#or y = zeros((n_samples, 1))
def noisy_labels(y, p_flip=0.05):
    # determine the number of labels to flip
    n_select = int(p_flip * y.shape[0])
    # choose labels to flip
    flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
    # invert the labels in place
    y[flip_ix] = 1 - y[flip_ix]
    return y

def gen_loss(d_real, d_fake, gan="wgan", weight=1., d_real_p=None, d_fake_p=None,noise_label=False):
    if gan.lower() == "wgan":
        wg_loss_orig = - d_fake.mean()
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_gen_loss": wg_loss.clone().detach().item(),
            "wgan_gen_loss_orig": wg_loss_orig.clone().detach().item(),
        }
    elif gan.lower() == "hinge":
        g_loss = -d_fake.mean()
        d_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        d_acc = d_correct / float(d_real.size(0) + d_fake.size(0))

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            "dis_acc": d_acc.clone().detach(),
            "dis_correct": d_correct.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "ls":
        #mse = nn.MSELoss()
        B = d_fake.size(0)
        #real_label_np = np.ones((B,))
        fake_label_np = np.ones((B,))

        if noise_label:
            # occasionally flip the labels when training the generator to fool the D
            fake_label_np = noisy_labels(fake_label_np, 0.05)

        #real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()

        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())
        g_loss = F.mse_loss(d_fake, fake_label)



        if d_fake_p is not None:
            fake_label_p = Variable(torch.FloatTensor(d_fake_p.size(0), d_fake_p.size(1)).fill_(1).cuda())
            g_loss_p = F.mse_loss(d_fake_p,fake_label_p)
            g_loss = g_loss + 0.2*g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "gan":
        fake_target = torch.tensor([1.0]).cuda()
        fake_loss = functools.partial(BCEfakeloss, target=fake_target)
        g_loss = fake_loss(d_fake)

        if d_fake_p is not None:
            g_loss_p = fake_loss(d_fake_p.view(-1))
            g_loss = g_loss + g_loss_p

        loss = weight * g_loss
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        # https://github.com/weishenho/SAGAN-with-relativistic/blob/master/main.py
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss =  torch.mean((d_real - torch.mean(d_fake) + y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) - y) ** 2)

        # d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        # g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss = (g_loss + d_loss) / 2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)


def mix_loss(d_mix, gan="wgan", weight=1.,d_mix_p=None,target_map_p=None):


    if gan.lower() == "ls":
        fake_label = Variable(torch.FloatTensor(d_mix.size(0)).fill_(0).cuda())
        mix_loss = F.mse_loss(d_mix, fake_label)

        if d_mix_p is not None:
            mix_loss_p = F.mse_loss(d_mix_p, target_map_p)
            mix_loss = (mix_loss + mix_loss_p)/2.0

        loss =  mix_loss
        return loss, {
            'loss': loss.clone().detach(),
        }
    elif gan.lower() =="gan":
        fake_target = torch.tensor([0.0]).cuda()

        mix_loss = F.binary_cross_entropy_with_logits(d_mix, fake_target.expand_as(d_mix),
                                                     reduction="none")

        if d_mix_p is not None:

            consistency_loss = F.mse_loss(d_mix_p, target_map_p)

            mix_list = []
            for i in range(d_mix_p.size(0)):
                # MIXUP LOSS 2D
                mix2d_i = F.binary_cross_entropy_with_logits(d_mix_p[i].view(-1), target_map_p[i].view(-1))
                mix_list.append(mix2d_i)

            D_loss_mixed_2d = torch.stack(mix_list)

            mix_loss = D_loss_mixed_2d + mix_loss
            mix_loss = mix_loss.mean()


            mix_loss = mix_loss + consistency_loss
            # -> D_loss_mixed_2d.mean() is taken later
        else:
            mix_loss = mix_loss.mean()

        loss = mix_loss
        return loss, {
            'loss': loss.clone().detach(),
        }
    else:
        raise NotImplementedError("Not implement: %s" % gan)

def dis_loss(d_real, d_fake, gan="wgan", weight=1.,d_real_p=None, d_fake_p=None, noise_label=False):
    # B = d_fake.size(0)
    # a = 1.0
    # b = 0.9

    if gan.lower() == "wgan":
        loss_fake = d_fake.mean()
        loss_real = d_real.mean()
        wg_loss_orig = loss_fake - loss_real
        wg_loss = wg_loss_orig * weight
        return wg_loss, {
            "wgan_dis_loss": wg_loss.clone().detach().item(),
            "wgan_dis_loss_orig": wg_loss_orig.clone().detach().item(),
            "wgan_dis_loss_real": loss_real.clone().detach().item(),
            "wgan_dis_loss_fake": loss_fake.clone().detach().item()
        }
    elif gan.lower() == "hinge":
        d_loss_real = torch.nn.ReLU()(1.0 - d_real).mean()
        d_loss_fake = torch.nn.ReLU()(1.0 + d_fake).mean()

        # d_loss_real = -torch.min(d_real - 1, d_real * 0).mean()
        # d_loss_fake = -torch.min(-d_fake - 1, d_fake * 0).mean()
        real_correct = (d_real >= 0.).float().sum() + (d_fake < 0.).float().sum()
        real_acc = real_correct / float(d_real.size(0) + d_fake.size(0))

        d_loss = d_loss_real + d_loss_fake
        loss = d_loss * weight
        return loss, {
            "loss": loss.clone().detach(),
            "d_loss": d_loss.clone().detach(),
            "dis_acc": real_acc.clone().detach(),
            "dis_correct": real_correct.clone().detach(),
            "loss_real": d_loss_real.clone().detach(),
            "loss_fake": d_loss_fake.clone().detach(),
        }
    elif gan.lower() == "ls":
        mse = nn.MSELoss()
        B = d_fake.size(0)

        real_label_np = np.ones((B,))
        fake_label_np = np.zeros((B,))

        if noise_label:
            real_label_np = smooth_labels(B,ran=[0.9,1.0])
            #fake_label_np = smooth_labels(B,ran=[0.0,0.1])
            # occasionally flip the labels when training the D to
            # prevent D from becoming too strong
            real_label_np = noisy_labels(real_label_np, 0.05)
            #fake_label_np = noisy_labels(fake_label_np, 0.05)


        real_label = torch.from_numpy(real_label_np.astype(np.float32)).cuda()
        fake_label = torch.from_numpy(fake_label_np.astype(np.float32)).cuda()


        # real_label = Variable((1.0 - 0.9) * torch.rand(d_fake.size(0)) + 0.9).cuda()
        # fake_label = Variable((0.1 - 0.0) * torch.rand(d_fake.size(0)) + 0.0).cuda()

        t = 0.5
        real_correct = (d_real >= t).float().sum()
        real_acc = real_correct / float(d_real.size(0))

        fake_correct  = (d_fake < t).float().sum()
        fake_acc = fake_correct / float(d_fake.size(0))
        # + d_fake.size(0))

        # real_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(1).cuda())
        # fake_label = Variable(torch.FloatTensor(d_fake.size(0)).fill_(0).cuda())

        g_loss = F.mse_loss(d_fake, fake_label)
        d_loss = F.mse_loss(d_real, real_label)

        if d_real_p is not None and d_fake_p is not None:

            real_label_p = Variable((1.0 - 0.9) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.9).cuda()
            fake_label_p = Variable((0.1 - 0.0) * torch.rand(d_fake_p.size(0), d_fake_p.size(1)) + 0.0).cuda()

            # real_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(1).cuda())
            # fake_label_p = Variable(torch.FloatTensor(d_real_p.size(0), d_real_p.size(1)).fill_(0).cuda())
            g_loss_p = F.mse_loss(d_fake_p, fake_label_p)
            d_loss_p = F.mse_loss(d_real_p, real_label_p)

            g_loss = (g_loss + 0.1*g_loss_p)
            d_loss = (d_loss + 0.1*d_loss_p)

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach(),
            "fake_acc": fake_acc.clone().detach(),
            "real_acc": real_acc.clone().detach()
        }
    elif gan.lower() =="gan":
        d_real_target = torch.tensor([1.0]).cuda()
        d_fake_target = torch.tensor([0.0]).cuda()
        discriminator_loss = functools.partial(BCEloss, d_real_target=d_real_target, d_fake_target=d_fake_target)

        g_loss, d_loss = discriminator_loss(d_fake, d_real)

        if d_real_p is not None and d_fake_p is not None:
            g_loss_p,d_loss_p = discriminator_loss(d_fake_p.view(-1),d_real_p.view(-1))
            g_loss = (g_loss + g_loss_p)/2.0
            d_loss = (d_loss + d_loss_p)/2.0

        loss =  (g_loss+d_loss)/2.0
        return loss, {
            'loss': loss.clone().detach(),
            'g_loss': g_loss.clone().detach(),
            'd_loss': g_loss.clone().detach()
        }
    elif gan.lower() == "real":
        y = Variable(torch.Tensor(d_real.size(0)).fill_(1.0), requires_grad=False)
        d_loss = torch.mean((d_real - torch.mean(d_fake) - y) ** 2)
        g_loss = torch.mean((d_fake - torch.mean(d_real) + y) ** 2)
        loss =  (g_loss+d_loss)/2.0

    else:
        raise NotImplementedError("Not implement: %s" % gan)

def BCEloss(D_fake, D_real, d_real_target, d_fake_target):
    real =  F.binary_cross_entropy_with_logits(D_real,d_real_target.expand_as(D_real))
    fake =  F.binary_cross_entropy_with_logits(D_fake,d_fake_target.expand_as(D_fake))
    return real, fake

def BCEfakeloss(D_fake,target):
    return F.binary_cross_entropy_with_logits(D_fake, target.expand_as(D_fake))


def dis_acc(d_real, d_fake, loss_type="wgan", **kwargs):
    if loss_type.lower() == "wgan":
        # No threshold, don't know which one is correct which is not
        return {}
    elif loss_type.lower() == "hinge":
        return {}
    else:
        raise NotImplementedError("Not implement: %s" % loss_type)


def gradient_penalty(x_real, x_fake, d_real, d_fake,
                     lambdaGP=10., gp_type='zero_center', eps=1e-8):
    if gp_type == "zero_center":
        bs = d_real.size(0)
        grad = torch.autograd.grad(
            outputs=d_real, inputs=x_real,
            grad_outputs=torch.ones_like(d_real).to(d_real),
            create_graph=True, retain_graph=True)[0]
        # [grad] should be either (B, D) or (B, #points, D)
        grad = grad.reshape(bs, -1)
        grad_norm = gp_orig = torch.sqrt(torch.sum(grad ** 2, dim=1)).mean()
        gp = gp_orig ** 2. * lambdaGP

        # real_image.requires_grad = True
        # grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
        # grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        # grad_penalty_real = 10 / 2 * grad_penalty_real
        # grad_penalty_real.backward()

        return gp, {
            'gp': gp.clone().detach().cpu(),
            'gp_orig': gp_orig.clone().detach().cpu(),
            'grad_norm': grad_norm.clone().detach().cpu()
        }
    else:
        raise NotImplemented("Invalid gp type:%s" % gp_type)


    #dist, ass = EMD(sample, ref, 0.005, 300)



class CutMix:

    def __init__(self):
        self.EMD = emd_module.emdModule()

    def __call__(self, real_data, fake_data,bs=16):

        real_data = real_data.transpose(1,2)
        fake_data = fake_data.transpose(1,2)

        B = real_data.size(0)
        N = real_data.size(1)
        lam = np.random.beta(1, 1, size=B)
        sample_nums = (lam * 2048).astype(np.int32)
        seeds = [16, 32, 64, 128, 256, 512]
        #sample_nums = np.random.choice(seeds, size=B).astype(np.int32)
        sample_id = np.random.choice(np.arange(N), size=B)
        #print(sample_id)
        sample_id = torch.from_numpy(sample_id).int().to(fake_data)

        alpha = torch.rand(B, 1, 1, requires_grad=True).to(fake_data)

        sample_id = torch.randint(2048,size=(bs,)).to(fake_data).long()
        #rf_dist = pairwise_dist(real_data,fake_data)
        rr_dist = pairwise_dist(real_data,real_data)

        map = torch.ones((B,N)).cuda()
        map_s = torch.ones((B)).cuda()

        for i in range(B):

            idx = rr_dist[i,sample_id[i]].topk(k=int(sample_nums[i]), dim=-1)[1]
            map[i,idx] = 0
            map_s[i] = 1.0 - 1.0*sample_nums[i]/N

        if torch.rand(1) > 0.5:
            map = 1.0 - map
            map_s = 1.0 -map_s

        dist, ass = self.EMD(real_data, fake_data, 0.005, 300)
        temp = fake_data
        ass = ass.long()
        for i in range(B):
            temp[i] = temp[i][ass[i]]


        temp_map = map.view(B, N, 1).repeat(1, 1, 3)
        temp = temp_map * real_data + (1.0 - temp_map) * temp


        return temp.transpose(1,2), map_s, map







class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=torch.device('cpu')):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.EMD = emd_module.emdModule()

        self.device = device

    def __call__(self, netD, real_data, fake_data,mapping=False):
        B = real_data.size(0)

        fake_data = fake_data[:B]

        alpha = torch.rand(B, 1, 1, requires_grad=True).to(fake_data).expand_as(fake_data)
        # randomly mix real and fake data
        #interpolates = real_data + alpha * (fake_data - real_data)
        interpolates = Variable(alpha * real_data + (1 - alpha) * fake_data, requires_grad=True)

        if mapping:
            fake_data = fake_data.transpose(1,2)
            real_data = real_data.transpose(1,2)
            dist, ass = self.EMD(fake_data, real_data, 0.005, 300)
            interpolates = real_data
            ass = ass.long()
            for i in range(B):
                interpolates[i] = interpolates[i][ass[i]]
            interpolates = alpha*fake_data + (1.0-alpha)*interpolates
            interpolates = interpolates.transpose(1,2)

        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(fake_data),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(B, -1)

        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty

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

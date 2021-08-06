import torch
from torch.autograd import Variable, grad

class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, lambdaGP, gamma=1, vertex_num=2500, device=None):
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.vertex_num = vertex_num
        self.device = device

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)

        fake_data = fake_data[:batch_size]
        
        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).cuda()
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs
        
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size,-1)
                         
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty
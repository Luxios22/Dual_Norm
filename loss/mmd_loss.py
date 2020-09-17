import torch
import numpy as np

# def mmd_loss_RBF(x, y, alpha=0.5):
#     # TODO: update the format and explanation.
#     xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
#
#     rx = (xx.diag().unsqueeze(0).expand_as(xx))
#     ry = (yy.diag().unsqueeze(0).expand_as(yy))
#
#     rx1 = (xx.diag().unsqueeze(0).expand_as(
#         torch.Tensor(y.size(0), x.size(0))))
#     ry1 = (yy.diag().unsqueeze(0).expand_as(
#         torch.Tensor(x.size(0), y.size(0))))
#
#     # K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
#     # L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
#     # P = torch.exp(- alpha * (rx1.t() + ry1 - 2*zz))
#
#     K = (torch.exp(- 0.5 * alpha * (rx.t() + rx - 2 * xx)) + torch.exp(- 0.1 * alpha * (rx.t() + rx - 2 * xx))
#          + torch.exp(- 0.05 * alpha * (rx.t() + rx - 2 * xx))) / 3
#     L = (torch.exp(- 0.5 * alpha * (ry.t() + ry - 2 * yy)) + torch.exp(- 0.1 * alpha * (ry.t() + ry - 2 * yy))
#          + torch.exp(- 0.05 * alpha * (ry.t() + ry - 2 * yy))) / 3
#     P = (torch.exp(- 0.5 * alpha * (rx1.t() + ry1 - 2 * zz)) + torch.exp(- 0.1 * alpha * (rx1.t() + ry1 - 2 * zz))
#          + torch.exp(- 0.05 * alpha * (rx1.t() + ry1 - 2 * zz))) / 3
#
#     beta1 = (1. / (x.size(0) * x.size(0)))
#     beta2 = (1. / (y.size(0) * y.size(0)))
#     gamma = (2. / (x.size(0) * y.size(0)))
#
#     # TODO: SUM??? NOT mean???
#     return beta1 * torch.sum(K) + beta2 * torch.sum(L) - gamma * torch.sum(P)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def mmd_loss_RBF(sample_1, sample_2, alphas, ret_matrix=False):
    """Evaluate the statistic.

    The kernel used is

    MMD2(P,Q)
        =∥EX∼Pφ(X)−EY∼Qφ(Y)∥2H
        =⟨EX∼Pφ(X),EX′∼Pφ(X′)⟩H+⟨EY∼Qφ(Y),EY′∼Qφ(Y′)⟩H−2⟨EX∼Pφ(X),EY∼Qφ(Y)⟩H
        =EX,X′∼Pk(X,X′)+EY,Y′∼Qk(Y,Y′)−2EX∼P,Y∼Qk(X,Y)

    .. math::

        k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},

    for the provided ``alphas``.

    Arguments
    ---------
    sample_1: :class:`torch:torch.autograd.Variable`
        The first sample, of size ``(n_1, d)``.
    sample_2: variable of shape (n_2, d)
        The second sample, of size ``(n_2, d)``.
    alphas : list of :class:`float`
        The kernel parameters.
    ret_matrix: bool
        If set, the call with also return a second variable.

        This variable can be then used to compute a p-value using
        :py:meth:`~.MMDStatistic.pval`.

    Returns
    -------
    :class:`float`
        The test statistic.
    :class:`torch:torch.autograd.Variable`
        Returned only if ``ret_matrix`` was set to true."""

    n_1 = sample_1.shape[0]
    n_2 = sample_2.shape[0]

    a00 = 1. / (n_1 * (n_1-1))
    a11 = 1. / (n_2 * (n_2-1))
    a01 = -1. / (n_1 * n_2)

    sample_12 = torch.cat((sample_1, sample_2), 0)
    distances = pairwise_distances(sample_12)

    kernels = None
    for alpha in alphas:
        kernels_a = torch.exp(- alpha * distances)
        if kernels is None:
            kernels = kernels_a
        else:
            kernels = kernels + kernels_a

    k_1 = kernels[:n_1, :n_1]
    k_2 = kernels[n_1:, n_1:]
    k_12 = kernels[:n_1, n_1:]

    mmd = (2 * a01 * k_12.sum() +
           a00 * k_1.sum() +
           a11 * k_2.sum())
    if ret_matrix:
        return mmd, kernels
    else:
        return mmd


class MMD_Loss(object):
    """MMD_Loss measures the difference between two distributions given their samples.

    alpha : float
        the param for RBF (or exponential quaratic funciton)
    """

    def __init__(self, alphas=[0.5, ]):
        self.alphas = alphas

    def __call__(self, hs):
        """
        hs : List[Tensor]
            a list of hidden features from multiple domains
        """
        losses = []
        num_domains = len(hs)
        for domain_x in range(num_domains):
            for domain_y in range(domain_x + 1, num_domains):
                losses.append(mmd_loss_RBF(
                    hs[domain_x], hs[domain_y], alphas=self.alphas))
        return torch.mean(torch.stack(losses), dim=0)

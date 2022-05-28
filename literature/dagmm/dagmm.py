import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

from models import LSTMAE


class EstimationNet(nn.Module):
    def __init__(
        self, input_size: int,
        h_size: int, dropout_p: float, n_gmm: int
    ):
        super(EstimationNet, self).__init__()
        # self.model = nn.Sequential(*[
        #     nn.Linear(input_size, h_size),
        #     nn.Tanh(),
        #     nn.Dropout(p=dropout_p),
        #     nn.Linear(h_size, n_gmm),
        #     nn.Softmax(dim=1)
        # ])
        self.linear1 = nn.Linear(input_size, h_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear2 = nn.Linear(h_size, n_gmm)

    def forward(self, z):
        emb = F.tanh(self.linear1(z))
        emb = self.dropout(emb)
        gamma = self.linear2(emb)
        gamma = F.softmax(gamma, dim=1)
        return gamma


class DAGMM(nn.Module):
    def __init__(
        self, window_size: int, c_in: int, h_size: int, z_c_size: int,
        n_layers: int, n_gmm: int, est_h_size: int, est_dropout_p: int
    ):
        super(DAGMM, self).__init__()
        self.compression_net = LSTMAE(
            c_in=c_in, h_size=h_size, z_size=z_c_size,
            n_layers=n_layers, pass_last_h_state=True)

        self.estNet_input_size = 2 + z_c_size  # window_size * z_c_size
        self.estimation_net = EstimationNet(
            input_size=self.estNet_input_size,
            h_size=est_h_size, n_gmm=n_gmm,
            dropout_p=est_dropout_p,
        )

        self.register_buffer(
            'phi', Variable(torch.zeros(n_gmm)))
        self.register_buffer(
            'mu', Variable(torch.zeros(n_gmm, self.estNet_input_size)))
        self.register_buffer(
            'cov', Variable(torch.zeros(
                n_gmm, self.estNet_input_size, self.estNet_input_size)))

    def relative_euclidean_distance(self, a, b, dim=-1):
        return (a - b).norm(2, dim=dim) /\
            torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        z_c = self.compression_net.encode(x)
        x_hat = self.compression_net.decode(z_c, seq_len)

        rec_euclidean = self.relative_euclidean_distance(
            x.view(batch_size, -1),
            x_hat.view(batch_size, -1),
            dim=1).view(batch_size, -1)
        rec_cosine = F.cosine_similarity(
            x.view(batch_size, -1),
            x_hat.view(batch_size, -1),
            dim=1).view(batch_size, -1)
        z = torch.cat(
            [z_c.view(batch_size, -1), rec_euclidean, rec_cosine],
            dim=1)

        gamma = self.estimation_net(z)
        return x_hat, z_c, z, gamma

    # def compute_gmm_params(self, z, gamma):
    #     N = gamma.size(0)
    #     # K
    #     sum_gamma = torch.sum(gamma, dim=0)

    #     # K
    #     phi = (sum_gamma / N)

    #     self.phi = phi.data

    #     # K x D
    #     mu = torch.sum(
    #         gamma.unsqueeze(2) * z.unsqueeze(1), dim=0) /\
    #         sum_gamma.unsqueeze(1)
    #     self.mu = mu.data
    #     # z = N x D
    #     # mu = K x D
    #     # gamma N x K

    #     # z_mu = N x K x D
    #     z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

    #     # z_mu_outer = N x K x D x D
    #     z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

    #     # K x D x D
    #     cov = torch.sum(
    #       gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0)\
    #         / sum_gamma.unsqueeze(-1).unsqueeze(-1)
    #     self.cov = cov.data

    #     return phi, mu, cov

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        # K
        sum_gamma = torch.sum(gamma, dim=0)

        # K
        phi = (sum_gamma / N)

        self.phi = phi.data

        # K x D
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0)\
            / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        # z = N x D
        # mu = K x D
        # gamma N x K

        # z_mu = N x K x D
        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        # z_mu_outer = N x K x D x D
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)

        # K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0)\
            / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data

        return phi, mu, cov

    def compute_energy(
        self, z, phi=None, mu=None, cov=None, size_average=True
    ):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)

        k, d, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0)).cpu()
        phi = phi.cpu()
        mu = mu.cpu()
        cov = cov.cpu()

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + Variable(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.cpu().numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))

            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            # if np.min(eigvals) < 0:
            #     logging.warning(f'Determinant was negative!
            #     Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(
                eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)

            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(
            z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu,
            dim=-1)
        # for stability (logsumexp)
        max_val = torch.max(
            (exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(Variable(phi.unsqueeze(0)) * exp_term / (
                torch.sqrt(Variable(det_cov)) + eps).unsqueeze(0),
                dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        # sample_energy = torch.clip(sample_energy, max=5)
        # cov_diag = torch.clip(cov_diag, max=5)
        return sample_energy, cov_diag

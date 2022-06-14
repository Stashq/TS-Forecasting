import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.N = N

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False)

        self.Q = self.K = self.V = self.sigma = torch.zeros((N, d_model))

    def forward(self, x):

        self.initialize(x)
        P = self.prior_association()
        S = self.series_association()
        Z = self.reconstruction(S)

        return Z, P, S

    def initialize(self, x):
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        self.sigma = self.Ws(x)

    @staticmethod
    def gaussian_kernel(mean, sigma):
        normalize = 1 / (math.sqrt(2 * torch.pi) * sigma)
        return normalize * torch.exp(-0.5 * (mean / sigma).pow(2))

    def prior_association(self):
        p = torch.from_numpy(
            np.abs(
                np.indices((self.N, self.N))[0]
                - np.indices((self.N, self.N))[1])
        )
        if next(self.parameters()).is_cuda:
            p = p.cuda()
        gaussian = self.gaussian_kernel(p.float(), self.sigma + 1e-8)
        gaussian /= gaussian.sum(dim=-1).unsqueeze(dim=-1)

        return gaussian

    def series_association(self):
        return F.softmax(
            (self.Q @ self.K.transpose(1, 2)) / math.sqrt(self.d_model),
            dim=0)

    def reconstruction(self, S):
        return S @ self.V


class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.attention = AnomalyAttention(self.N, self.d_model)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        x, p, s = self.attention(x)
        z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        x_next = self.ln2(z + z_identity)

        return x_next, p, s


class AnomalyTransformer(nn.Module):
    def __init__(
        self, window_size: int, c_in: int, d_model: int,
        n_layers: int, lambda_: float
    ):
        super().__init__()
        self.params = {
            'c_in': c_in, 'window_size': window_size,
            'd_model': d_model, 'n_layers': n_layers,
            'lambda_': lambda_
        }
        self.N = window_size
        self.d_model = d_model
        self.lambda_ = lambda_

        self.embedding_layer = nn.Linear(c_in, d_model)
        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model)
             for _ in range(n_layers)]
        )
        self.decoding_layer = nn.Linear(d_model, c_in)

    def forward(self, x):
        P_layers = []
        S_layers = []
        x = self.embedding_layer(x)
        for block in self.blocks:
            x, p, s = block(x)
            P_layers.append(p)
            S_layers.append(s)

        x = self.decoding_layer(x)
        return x, P_layers, S_layers

    def layer_association_discrepancy(self, Pl, Sl):
        ad_vector =\
            torch.sum(
                F.kl_div(Pl, Sl, reduction='none'),
                dim=-1
            ) + torch.sum(
                F.kl_div(Sl, Pl, reduction='none'),
                dim=-1
            )
        return ad_vector

    def association_discrepancy(self, P_list, S_list):
        return (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy(P, S)
                for P, S in zip(P_list, S_list)
            ]
        )

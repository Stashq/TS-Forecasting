import torch
import torch.nn as nn
from models import Encoder, Decoder


class ConstraintNet(nn.Module):
    def __init__(self, c_in: int, z_size: int, N: int, threshold: float):
        super(ConstraintNet, self).__init__()

        self.l1 = nn.Linear(c_in, 8)
        self.l2 = nn.Linear(8, 16)
        self.l3 = nn.Linear(16, N)
        self.cos = nn.CosineSimilarity(dim=-1)

        # self.layer1 = layers.Dense(8, input_dim=c_in, activation='relu')
        # self.layer2 = layers.Dense(16, activation='relu')
        # self.out = layers.Dense(z_size * N, activation='relu')
        # self.reshape = layers.Reshape((N, z_size))

        self.N = N
        self.z_size = z_size
        self.threshold = threshold

    def forward(self, x, z):
        x = self.l1(x)
        x = self.l2(x)
        C = self.l3(x)
        # C = x.view(-1, self.N, self.z_size)

        w = torch.concat([
            self.cos(z, C[:, :, i])
            for i in range(self.N)
        ]).view(-1, self.N)

        w_norm = torch.linalg.norm(w, dim=-1).unsqueeze(dim=-1)
        w = w / w_norm

        w_mask = (w > self.threshold).float()
        w_hat = (w * w_mask).unsqueeze(dim=1)

        z_hat = w_hat @ C.transpose(1, 2)
        z_hat = z_hat.squeeze()
        return z_hat


class VELC(nn.Module):
    def __init__(
        self, c_in: int, h_size: int,  n_layers: int,
        z_size: int, N_constraint: int, threshold: float
    ):
        super(VELC, self).__init__()

        self.encoder = Encoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.z_mu_dense = nn.Linear(z_size, z_size)
        self.z_log_sig_dense = nn.Linear(z_size, z_size)
        self.constraint_net_1 = ConstraintNet(
            c_in=c_in, z_size=z_size, N=N_constraint, threshold=threshold)

        self.decoder = Decoder(
            z_size=z_size, h_size=h_size, x_size=c_in, n_layers=n_layers)

        self.re_encoder = Encoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.re_z_mu_dense = nn.Linear(z_size, z_size)
        self.re_z_log_sig_dense = nn.Linear(z_size, z_size)
        self.constraint_net_2 = ConstraintNet(
            c_in=c_in, z_size=z_size, N=N_constraint, threshold=threshold)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def sample1(self, h):
        z_mu = self.z_mu_dense(h)
        z_log_sig = self.z_log_sig_dense(h)
        z = self.reparametrization(z_mu, z_log_sig)
        return z, z_mu, z_log_sig

    def sample2(self, h):
        re_z_mu = self.re_z_mu_dense(h)
        re_z_log_sig = self.re_z_log_sig_dense(h)
        re_z = self.reparametrization(re_z_mu, re_z_log_sig)
        return re_z, re_z_mu, re_z_log_sig

    def forward(self, x):
        h = self.encoder(x)
        z, z_mu, z_log_sig = self.sample1(h)
        z_dash = self.constraint_net_1(x, z)

        x_dash = self.decoder(z_dash, seq_len=x.shape[1])

        h = self.re_encoder(x_dash)
        re_z, re_z_mu, re_z_log_sig = self.sample2(h)
        re_z_dash = self.constraint_net_2(x_dash, re_z)

        return (
            x_dash, z_dash, z_mu, z_log_sig,
            re_z_dash, re_z_mu, re_z_log_sig)

    def anomaly_score(self, x, x_dash, z_dash, re_z_dash, alpha):
        a1 = torch.linalg.norm(x, x_dash, dim=-1)
        a2 = torch.linalg.norm(z_dash, re_z_dash, dim=-1)
        a = alpha * a1 + (1 - alpha) * a2
        a = torch.mean(a, dim=-1)
        return a

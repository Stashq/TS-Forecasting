import torch
import torch.nn as nn
import torch.nn.functional as F
from models import LSTMEncoder, LSTMDecoder


class ConstraintNet(nn.Module):
    def __init__(
        self, c_in: int, window_size: int, z_size: int,
        N: int, threshold: float
    ):
        super(ConstraintNet, self).__init__()

        self.l1 = nn.Linear(c_in*window_size, 8)
        self.l2 = nn.Linear(8, 16)
        self.l3 = nn.Linear(16, z_size*N)
        self.cos = nn.CosineSimilarity(dim=-1)

        # self.layer1 = layers.Dense(8, input_dim=c_in, activation='relu')
        # self.layer2 = layers.Dense(16, activation='relu')
        # self.out = layers.Dense(z_size * N, activation='relu')
        # self.reshape = layers.Reshape((N, z_size))

        self.N = N
        self.z_size = z_size
        self.threshold = threshold

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        C = self.l3(x)
        C = C.view(batch_size, self.N, self.z_size)

        # w shape: batch x window x N
        w = torch.stack([
            self.cos(z, C[:, i, :].unsqueeze(dim=1))
            for i in range(self.N)
        ], dim=-1)  # .view(-1, self.N)

        norm = torch.linalg.norm(w, dim=-1).unsqueeze(dim=-1)
        w_norm = w / norm

        w_mask = (w_norm > self.threshold).float()
        w_hat = (w_norm * w_mask)

        # z_hat shape: batch x window x z_size
        z_hat = w_hat @ C
        return z_hat


class VELC(nn.Module):
    def __init__(
        self, c_in: int, window_size: int, h_size: int,  n_layers: int,
        z_size: int, N_constraint: int, threshold: float
    ):
        super(VELC, self).__init__()
        self.params = {
            'c_in': c_in, 'window_size': window_size, 'h_size': h_size,
            'n_layers': n_layers, 'z_size': z_size,
            'N_constraint': N_constraint, 'threshold': threshold
        }
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.z_mu_dense = nn.Linear(z_size, z_size)
        self.z_log_sig_dense = nn.Linear(z_size, z_size)
        self.constraint_net_1 = ConstraintNet(
            c_in=c_in, window_size=window_size, z_size=z_size,
            N=N_constraint, threshold=threshold)

        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, x_size=c_in, n_layers=n_layers)

        self.re_encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.re_z_mu_dense = nn.Linear(z_size, z_size)
        self.re_z_log_sig_dense = nn.Linear(z_size, z_size)
        self.constraint_net_2 = ConstraintNet(
            c_in=c_in, window_size=window_size, z_size=z_size,
            N=N_constraint, threshold=threshold)

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

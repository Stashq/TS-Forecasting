from torch import nn
import torch
# import torch.nn.functional as F

# from ..predpy.wrapper.autoencoder import Autoencoder


class AEPart(nn.Module):
    def __init__(
        self,
        input_size: int,
        h_size: int,
        n_layers: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, x):
        emb, (_, _) = self.lstm(x)
        return emb


class LSTMPVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMPVAE, self).__init__()
        self.encoder = AEPart(c_in, h_size, n_layers)
        self.decoder = AEPart(h_size, c_in, n_layers)
        self.z_mu_dense = nn.Linear(h_size, h_size)
        self.z_log_sig_dense = nn.Linear(h_size, h_size)
        self.x_mu_dense = nn.Linear(c_in, c_in)
        self.x_log_sig_dense = nn.Linear(c_in, c_in)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def forward(self, x: torch.Tensor):
        z, z_mu, z_log_sig = self.encode(x, return_all=True)
        x_tilda, x_mu, x_log_sig = self.decode(z)
        return (x_tilda, z_mu, z_log_sig, x_mu, x_log_sig)

    def decode(self, z):
        emb = self.decoder(z)
        x_mu, x_log_sig = self.x_mu_dense(emb), self.x_log_sig_dense(emb)
        x_tilda = self.reparametrization(x_mu, x_log_sig)
        return x_tilda, x_mu, x_log_sig

    def encode(self, x: torch.Tensor, return_all: bool = False):
        emb = self.encoder(x)
        z_mu, z_log_sig = self.z_mu_dense(emb), self.z_log_sig_dense(emb)
        z = self.reparametrization(z_mu, z_log_sig)
        if return_all:
            res = z, z_mu, z_log_sig
        else:
            res = z
        return res

from torch import nn
import torch
# import torch.nn.functional as F

# from ..predpy.wrapper.autoencoder import Autoencoder


class Encoder(nn.Module):
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
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class Decoder(nn.Module):
    def __init__(
        self,
        z_size: int,
        h_size: int,
        n_layers: int
    ):
        super().__init__()
        self.z_size = z_size
        self.n_layers = n_layers
        self.h_size = h_size
        self.lstm = nn.LSTM(
            input_size=z_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )

    def forward(self, z):
        x_tilda, (_, _) = self.lstm(z)
        return x_tilda


class LSTMVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMVAE, self).__init__()
        self.encoder = Encoder(c_in, h_size, n_layers)
        self.decoder = Decoder(h_size, c_in, n_layers)
        self.z_mu_dense = nn.Linear(h_size, h_size)
        self.z_log_sig_dense = nn.Linear(h_size, h_size)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def forward(self, x: torch.Tensor):
        z, z_mu, z_log_sig = self.encode(x, return_all=True)
        x_tilda = self.decode(z, seq_len=x.shape[1])
        return (x_tilda, z_mu, z_log_sig)

    def decode(self, z, seq_len: int):
        z_repeated = z.unsqueeze(1).repeat(1, seq_len, 1)
        x_tilda = self.decoder(z_repeated)
        return x_tilda

    def encode(self, x: torch.Tensor, return_all: bool = False):
        emb = self.encoder(x)
        z_mu, z_log_sig = self.z_mu_dense(emb), self.z_log_sig_dense(emb)
        z = self.reparametrization(z_mu, z_log_sig)
        if return_all:
            res = z, z_mu, z_log_sig
        else:
            res = z
        return res

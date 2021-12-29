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
        emb, (_, _) = self.lstm(x)
        return emb


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


class LSTMVariationalAutoencoder(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMVariationalAutoencoder, self).__init__()
        self.encoder = Encoder(c_in, h_size, n_layers)
        self.decoder = Decoder(h_size, c_in, n_layers)
        # TODO: set same dimentions for input (z) as output of encoder
        self.mu_dense = nn.Linear(h_size, h_size)
        self.log_sig_dense = nn.Linear(h_size, h_size)

    def reparametrization(self, z_mu, z_log_sig):
        eps = torch.randn_like(z_mu)
        z = z_mu + eps * torch.exp(z_log_sig/2.0)
        return z

    def forward(self, x):
        emb = self.encoder(x)
        z_mu, z_log_sig = self.mu_dense(emb), self.log_sig_dense(emb)
        z = self.reparametrization(z_mu, z_log_sig)
        x_tilda = self.decoder(z)
        return (x_tilda, z_mu, z_log_sig)

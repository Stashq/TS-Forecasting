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


class LSTMPAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMPAE, self).__init__()
        self.encoder = AEPart(c_in, h_size, n_layers)
        self.decoder = AEPart(h_size, c_in, n_layers)
        self.mu_dense = nn.Linear(c_in, c_in)
        self.log_sig_dense = nn.Linear(c_in, c_in)

    def reparametrization(self, x_mu, x_log_sig):
        eps = torch.randn_like(x_mu)
        x = x_mu + eps * torch.exp(x_log_sig/2.0)
        return x

    def encode(self, x: torch.Tensor):
        emb = self.encoder(x)
        return emb

    def predict(self, x: torch.Tensor):
        emb = self.encode(x)
        emb = self.decoder(emb)
        x_mu, x_log_sig = self.mu_dense(emb), self.log_sig_dense(emb)
        return x_mu, x_log_sig

    def forward(self, x: torch.Tensor):
        x_mu, x_log_sig = self.predict(x)
        x_tilda = self.reparametrization(x_mu, x_log_sig)
        return x_tilda, x_mu, x_log_sig

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
        output_size: int,
        n_layers: int
    ):
        super().__init__()
        self.z_size = z_size
        self.h_size = h_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=z_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.dense = nn.Linear(h_size, output_size)

    def forward(self, z, seq_len: int):
        # emb, (_, _) = self.lstm(z)
        # batch_size, seq_len, _ = emb.shape
        # # imitate time-distributed dense layer
        # emb = emb.reshape(batch_size*seq_len, self.h_size)
        # x_tilda = self.dense(emb)
        # x_tilda = x_tilda.reshape(batch_size, seq_len, self.output_size)
        # return x_tilda
        z = z.repeat(1, seq_len, 1)
        emb, (_, _) = self.lstm(z)
        x_tilda = self.dense(emb)
        return x_tilda


class LSTMAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super().__init__()
        self.c_in = c_in
        self.n_layers = n_layers
        self.h_size = h_size
        self.encoder = Encoder(
            input_size=c_in, h_size=h_size, n_layers=n_layers)
        self.decoder = Decoder(
            z_size=h_size, h_size=h_size,
            output_size=c_in, n_layers=n_layers)

    def forward(self, x):
        emb = self.encoder(x)
        x_tilda = self.decode(emb)
        return x_tilda

    def encode(self, x):
        return self.encoder(x)

    def decode(self, emb):
        x_tilda = self.decoder(emb)
        return x_tilda


class LSTMVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMVAE, self).__init__()
        self.encoder = Encoder(
            input_size=c_in, h_size=h_size, n_layers=n_layers)
        self.z_mu_dense = nn.Linear(h_size, h_size)
        self.z_log_sig_dense = nn.Linear(h_size, h_size)
        self.decoder = Decoder(
            z_size=h_size, h_size=h_size, output_size=c_in, n_layers=n_layers)

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


class LSTMPAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMPAE, self).__init__()
        self.encoder = Encoder(
            input_size=c_in, h_size=h_size, n_layers=n_layers)
        self.decoder = Decoder(
            z_size=h_size, h_size=h_size, output_size=c_in, n_layers=n_layers)
        self.x_mu_dense = nn.Linear(c_in, c_in)
        self.x_log_sig_dense = nn.Linear(c_in, c_in)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def encode(self, x: torch.Tensor):
        emb = self.encoder(x)
        return emb

    def decode(self, emb: torch.Tensor):
        emb = self.decoder(emb)
        x_mu, x_log_sig = self.x_mu_dense(emb), self.x_log_sig_dense(emb)
        x_tilda = self.reparametrization(x_mu, x_log_sig)
        return x_tilda, x_mu, x_log_sig

    def predict(self, x: torch.Tensor):
        emb = self.encode(x)
        _, x_mu, x_log_sig = self.decode(emb)
        return x_mu, x_log_sig

    def forward(self, x: torch.Tensor):
        emb = self.encode(x)
        x_tilda, x_mu, x_log_sig = self.decode(emb)
        return x_tilda, x_mu, x_log_sig


class LSTMPVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int
    ):
        super(LSTMPVAE, self).__init__()
        self.encoder = Encoder(
            input_size=c_in, h_size=h_size, n_layers=n_layers)
        self.decoder = Decoder(
            z_size=h_size, h_size=h_size, output_size=c_in, n_layers=n_layers)
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

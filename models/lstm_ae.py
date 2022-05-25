from torch import nn
import torch
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        x_size: int,
        h_size: int,
        n_layers: int,
        emb_size: int = None
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=x_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=h_size,
            out_features=emb_size
        )
        # self.linear1 = nn.Linear(
        #     in_features=h_size,
        #     out_features=h_size
        # )
        # self.linear2 = nn.Linear(
        #     in_features=h_size,
        #     out_features=emb_size
        # )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        emb = F.relu(h_n[-1])
        # emb = F.relu(self.linear1(emb))
        # emb = F.relu(self.linear2(emb))
        emb = F.relu(self.linear(emb))
        return emb


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        z_size: int,
        h_size: int,
        n_layers: int,
        x_size: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=z_size,
            hidden_size=h_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(
            in_features=h_size,
            out_features=x_size
        )
        # self.linear1 = nn.Linear(
        #     in_features=h_size,
        #     out_features=h_size
        # )
        # self.linear2 = nn.Linear(
        #     in_features=h_size,
        #     out_features=x_size
        # )

    def forward(self, z, seq_len: int):
        z = z.unsqueeze(1)
        z = z.repeat(1, seq_len, 1)
        emb, (_, _) = self.lstm(z)
        emb = torch.flip(emb, dims=[1])
        emb = F.relu(emb)
        # emb = F.relu(self.linear1(emb))
        # x_hat = self.linear2(emb)
        x_hat = self.linear(emb)
        return x_hat


class LSTMAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int,
        z_size: int
    ):
        super().__init__()
        self.c_in = c_in
        self.n_layers = n_layers
        self.h_size = h_size
        self.z_size = z_size
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=c_in)

    def forward(self, x):
        emb = self.encoder(x)
        x_hat = self.decode(emb, seq_len=x.shape[1])
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, emb, seq_len: int):
        x_hat = self.decoder(emb, seq_len)
        return x_hat


class LSTMVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int,
        z_size: int
    ):
        super(LSTMVAE, self).__init__()
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, emb_size=z_size)
        self.z_mu_dense = nn.Linear(z_size, z_size)
        self.z_log_sig_dense = nn.Linear(z_size, z_size)
        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=c_in)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def forward(self, x: torch.Tensor):
        z, z_mu, z_log_sig = self.encode(x, return_all=True)
        x_hat = self.decode(z, seq_len=x.shape[1])
        return (x_hat, z_mu, z_log_sig)

    def decode(self, z, seq_len: int):
        x_hat = self.decoder(z, seq_len=seq_len)
        return x_hat

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
        n_layers: int,
        z_size: int
    ):
        super(LSTMPAE, self).__init__()
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, z_size=z_size)
        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=c_in)
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
        x_hat = self.reparametrization(x_mu, x_log_sig)
        return x_hat, x_mu, x_log_sig

    def predict(self, x: torch.Tensor):
        emb = self.encode(x)
        _, x_mu, x_log_sig = self.decode(emb)
        return x_mu, x_log_sig

    def forward(self, x: torch.Tensor):
        emb = self.encode(x)
        x_hat, x_mu, x_log_sig = self.decode(emb)
        return x_hat, x_mu, x_log_sig


class LSTMPVAE(nn.Module):
    def __init__(
        self,
        c_in: int,
        h_size: int,
        n_layers: int,
        z_size: int
    ):
        super(LSTMPVAE, self).__init__()
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers, z_size=z_size)
        self.z_mu_dense = nn.Linear(h_size, h_size)
        self.z_log_sig_dense = nn.Linear(h_size, h_size)
        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=c_in)
        self.x_mu_dense = nn.Linear(c_in, c_in)
        self.x_log_sig_dense = nn.Linear(c_in, c_in)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def forward(self, x: torch.Tensor):
        z, z_mu, z_log_sig = self.encode(x, return_all=True)
        x_hat, x_mu, x_log_sig = self.decode(z)
        return (x_hat, z_mu, z_log_sig, x_mu, x_log_sig)

    def decode(self, z):
        emb = self.decoder(z)
        x_mu, x_log_sig = self.x_mu_dense(emb), self.x_log_sig_dense(emb)
        x_hat = self.reparametrization(x_mu, x_log_sig)
        return x_hat, x_mu, x_log_sig

    def encode(self, x: torch.Tensor, return_all: bool = False):
        emb = self.encoder(x)
        z_mu, z_log_sig = self.z_mu_dense(emb), self.z_log_sig_dense(emb)
        z = self.reparametrization(z_mu, z_log_sig)
        if return_all:
            res = z, z_mu, z_log_sig
        else:
            res = z
        return res

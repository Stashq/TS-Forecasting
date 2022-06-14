from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
from typing import Dict, Generator, List, Tuple, Union

from predpy.wrapper import Reconstructor
from anomaly_detection import AnomalyDetector


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        x_size: int,
        h_size: int,
        n_layers: int,
        emb_size: int = None,
        pass_last_h_state: bool = False
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
        self.pass_last_h_state = pass_last_h_state
        # self.linear1 = nn.Linear(
        #     in_features=h_size,
        #     out_features=h_size
        # )
        # self.linear2 = nn.Linear(
        #     in_features=h_size,
        #     out_features=emb_size
        # )

    def forward(self, x):
        # _, (h_l, _) = self.lstm(x)
        emb, (h_l, _) = self.lstm(x)
        if self.pass_last_h_state:
            emb = F.relu(h_l[-1].unsqueeze(1))
        else:
            emb = F.relu(emb)
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
        x_size: int,
        last_h_on_input: bool = False
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
        self.last_h_on_input = last_h_on_input
        # self.linear1 = nn.Linear(
        #     in_features=h_size,
        #     out_features=h_size
        # )
        # self.linear2 = nn.Linear(
        #     in_features=h_size,
        #     out_features=x_size
        # )

    def forward(self, z, seq_len: int):
        # z = z.unsqueeze(1)
        if self.last_h_on_input:
            z = z.repeat(1, seq_len, 1)
        emb, (_, _) = self.lstm(z)
        # emb = torch.flip(emb, dims=[1])
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
        z_size: int,
        pass_last_h_state: bool = False
    ):
        super().__init__()
        self.params = dict(
            c_in=c_in, h_size=h_size,
            n_layers=n_layers, z_size=z_size,
            pass_last_h_state=pass_last_h_state
        )
        self.c_in = c_in
        self.n_layers = n_layers
        self.h_size = h_size
        self.z_size = z_size
        self.encoder = LSTMEncoder(
            x_size=c_in, h_size=h_size, n_layers=n_layers,
            emb_size=z_size, pass_last_h_state=pass_last_h_state)
        self.decoder = LSTMDecoder(
            z_size=z_size, h_size=h_size, n_layers=n_layers,
            x_size=c_in, last_h_on_input=pass_last_h_state)

    def forward(self, x):
        emb = self.encoder(x)
        x_hat = self.decode(emb, seq_len=x.shape[1])
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, emb, seq_len: int):
        x_hat = self.decoder(emb, seq_len)
        return x_hat


class LSTMAEWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: LSTMAE = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None,
    ):
        AnomalyDetector.__init__(self, score_names=['norm2_x'])
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.mse = nn.MSELoss(reduction='none')

    def get_loss(self, x, x_hat):
        return self.criterion(x, x_hat)

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat = self.model(x)
        loss = self.get_loss(x, x_hat)
        return loss

    def predict(self, x):
        with torch.no_grad():
            x_hat = self.model(x)
            return x_hat

    def anomaly_score(
        self, x, scale: bool = False, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        batch_size = x.size(0)
        with torch.no_grad():
            x_hat = self.model(x)
            # mse not including batch
            norm2_x = torch.sum(
                torch.sum(torch.square(x - x_hat), dim=-1), dim=-1)
        # score = torch.stack([norm2_x1, norm2_x2], dim=1).tolist()
        score = norm2_x.view(batch_size, 1).tolist()
        if scale:
            score = self.scores_scaler.transform(score).tolist()
        if return_pred:
            return score, x_hat
        return score


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

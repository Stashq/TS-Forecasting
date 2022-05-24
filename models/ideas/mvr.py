import torch
# import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from typing import Dict, Generator, List, Tuple, Union

from models import LSTMEncoder, LSTMDecoder, ConvEncoder, ConvDecoder
from predpy.wrapper import Reconstructor
from literature.anomaly_detector_base import AnomalyDetector


class LSTMMVR(nn.Module):
    """LSTM Multivariate reconstructor"""
    def __init__(self, c_in: int, h_size: int, z_size: int, z_glob_size: int):
        super(LSTMMVR, self).__init__()
        self.encoders = nn.ModuleList([
            LSTMEncoder(x_size=1, h_size=h_size, n_layers=1, emb_size=z_size)
            for _ in range(c_in)
        ])
        self.decoders = nn.ModuleList([
            LSTMDecoder(z_size=z_size, h_size=h_size, n_layers=1, x_size=1)
            for _ in range(c_in)
        ])
        self.c_in = c_in
        self.z_glob_size = z_glob_size

    def forward(self, x):
        seq_len = x.size(1)
        # z = []
        x_hat = []
        for i in range(self.c_in):
            z_i = self.encoders[i](x[:, :, i:i+1])
            x_i_hat = self.decoders[i](z_i, seq_len=seq_len)
            # z += [z_i]
            x_hat += [x_i_hat]
        x_hat = torch.concat(x_hat, dim=-1)
        # z = torch.concat(z)
        # return x_hat, z
        return x_hat


class ConvMVR(nn.Module):
    """Convolutional Multivariate reconstructor"""
    def __init__(
        self,
        window_size: int,
        c_in: int,
        n_kernels: int,
        kernel_size: int,
        emb_size: int,
        z_glob_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super(ConvMVR, self).__init__()
        self.encoders = nn.ModuleList([
            ConvEncoder(
                window_size=window_size, x_chanels=1, emb_chanels=n_kernels,
                kernel_size=kernel_size, emb_size=emb_size, padding=padding,
                stride=stride)
            for _ in range(c_in)
        ])
        self.decoders = nn.ModuleList([
            ConvDecoder(
                window_size=window_size, x_chanels=1, emb_chanels=n_kernels,
                kernel_size=kernel_size, emb_size=emb_size, padding=padding,
                stride=stride)
            for _ in range(c_in)
        ])
        self.c_in = c_in
        self.z_glob_size = z_glob_size

    def forward(self, x):
        # z = []
        x_hat = []
        for i in range(self.c_in):
            z_i = self.encoders[i](x[:, :, i:i+1])
            x_i_hat = self.decoders[i](z_i)
            # z += [z_i]
            x_hat += [x_i_hat]
        x_hat = torch.concat(x_hat, dim=-1)
        # z = torch.concat(z)
        # return x_hat, z
        return x_hat


class MVRWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: Union[LSTMMVR, ConvMVR] = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None,
    ):
        AnomalyDetector.__init__(self)
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
            return self.model(x)

    def anomaly_score(
        self, x, scale: bool = False, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        with torch.no_grad():
            x_hat = self.model.forward(x)
            # mse not including batch
            score = torch.sum(
                torch.sum(torch.square(x - x_hat), dim=-1), dim=-1)
        score = score.tolist()
        if scale:
            score = self.scores_scaler.transform(score).flatten().tolist()
        if return_pred:
            return score, x_hat
        return score

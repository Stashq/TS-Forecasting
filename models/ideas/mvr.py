import torch
# import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
# import torch.nn.functional as F
from typing import Dict, Generator, List, Tuple, Union

from models import LSTMEncoder, LSTMDecoder, ConvEncoder, ConvDecoder
from predpy.wrapper import Reconstructor
from anomaly_detection.anomaly_detector_base import AnomalyDetector


class LSTMMVR(nn.Module):
    """LSTM Multivariate reconstructor"""
    def __init__(
        self, window_size: int, c_in: int, h_size: int, z_size: int,
        n_layers: int = 1, lambda_: float = 0.2
    ):
        super(LSTMMVR, self).__init__()
        self.encoders = nn.ModuleList([
            LSTMEncoder(
                x_size=1, h_size=h_size, n_layers=n_layers, emb_size=z_size)
            for _ in range(c_in)
        ])
        self.decoders = nn.ModuleList([
            LSTMDecoder(
                z_size=z_size, h_size=h_size, n_layers=n_layers, x_size=1)
            for _ in range(c_in)
        ])
        self.l_norm = nn.LayerNorm(z_size*c_in)
        self.redecoder = LSTMDecoder(
            z_size=z_size*c_in, h_size=h_size, n_layers=n_layers, x_size=c_in
        )
        self.window_size = window_size
        self.c_in = c_in
        self.lambda_ = lambda_

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        z = []
        x_hat1 = []
        for i in range(self.c_in):
            z_i = self.encoders[i](x[:, :, i:i+1])
            x_i_hat = self.decoders[i](z_i, seq_len=seq_len)
            z += [z_i]
            x_hat1 += [x_i_hat]
        z = torch.concat(z, dim=-1)
        x_hat1 = torch.concat(x_hat1, dim=-1)

        z = self.l_norm(z)
        z = z + self.lambda_\
            * torch.randn(z.shape).to(
                next(self.l_norm.parameters()).device
            )

        x_hat2 = self.redecoder(z, seq_len=seq_len)

        return x_hat1, x_hat2


class ConvMVR(nn.Module):
    """Convolutional Multivariate reconstructor"""
    def __init__(
        self,
        window_size: int,
        c_in: int,
        n_kernels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
        lambda_: float = 0.1,
    ):
        super(ConvMVR, self).__init__()
        self.params = dict(
            window_size=window_size, c_in=c_in, n_kernels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride, lambda_=lambda_
        )

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
        self.l_norm = nn.LayerNorm(emb_size)
        self.redecoder = ConvDecoder(
            window_size=window_size, x_chanels=c_in,
            emb_chanels=n_kernels*c_in, kernel_size=kernel_size,
            emb_size=emb_size, padding=padding, stride=stride)
        self.c_in = c_in
        self.n_kernels = n_kernels
        self.lambda_ = lambda_

    def forward(self, x):
        z = []
        x_hat1 = []
        for i in range(self.c_in):
            z_i = self.encoders[i](x[:, :, i:i+1])
            x_i_hat = self.decoders[i](z_i)
            z += [z_i]
            x_hat1 += [x_i_hat]
        x_hat1 = torch.concat(x_hat1, dim=-1)
        z = torch.concat(z, dim=1)

        z = self.l_norm(z)
        z = z + self.lambda_\
            * torch.randn(z.shape).to(
                next(self.l_norm.parameters()).device
            )

        x_hat2 = self.redecoder(z)

        return x_hat1, x_hat2


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
        lambda_lr: float = 0.2
    ):
        AnomalyDetector.__init__(self, score_names=['norm2_x1', 'norm2_x2'])
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.lambda_lr = lambda_lr
        self.mse = nn.MSELoss(reduction='none')

    def get_loss(self, x, x_hat):
        return self.criterion(x, x_hat)

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat1, x_hat2 = self.model(x)
        loss1 = self.get_loss(x, x_hat1)
        loss2 = self.get_loss(x, x_hat2)
        loss = loss1 + self.lambda_lr*loss2
        return loss

    def predict(self, x):
        with torch.no_grad():
            x_hat1, x_hat2 = self.model(x)
            return x_hat1

    def anomaly_score(
        self, x, scale: bool = False, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        with torch.no_grad():
            x_hat1, x_hat2 = self.model(x)
            # mse not including batch
            # norm2_x1 = torch.sum(
            #     torch.sum(torch.square(x - x_hat1), dim=-1), dim=-1)
            # norm2_x2 = torch.sum(
            #     torch.sum(torch.square(x - x_hat2), dim=-1), dim=-1)
            norm2_x1 = torch.sum(
                torch.square(x - x_hat1), dim=1)
            norm2_x2 = torch.sum(
                torch.square(x - x_hat2), dim=1)
        # score = torch.stack([norm2_x1, norm2_x2], dim=1).tolist()
        score = torch.concat([norm2_x1, norm2_x2], dim=1).tolist()
        if scale:
            score = self.scores_scaler.transform(score).tolist()
        if return_pred:
            return score, x_hat1
        return score

from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
# import torch.nn.functional as F
from typing import Dict, Generator, List, Tuple, Union

from predpy.wrapper import Reconstructor
from anomaly_detection.anomaly_detector_base import AnomalyDetector


def get_conv_output_size(seq_len, k_size, padding, stride):
    numerator = seq_len + 2*padding - k_size
    res = int(numerator/stride) + 1
    return res


def get_conv_trans_output_size(seq_len, k_size, padding, stride):
    res = (seq_len - 1) * stride + k_size - 2 * padding
    return res


class ConvEncoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        x_chanels: int,
        emb_chanels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=x_chanels,
            out_channels=emb_chanels,
            kernel_size=kernel_size,
            padding=padding,
        )
        output_size = get_conv_output_size(
            window_size, kernel_size, padding, stride
        )
        self.linear = nn.Linear(
            in_features=output_size,
            out_features=emb_size
        )

    def forward(self, x):
        """x shape is <N, L, C>, where:
        - N is a batch size,
        - L is sequence len,
        - C is features."""
        x = torch.transpose(x, -2, -1)
        emb = F.relu(self.conv(x))
        emb = F.relu(self.linear(emb))
        return emb


class ConvDecoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        emb_chanels: int,
        x_chanels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super().__init__()
        self.conv_trans = nn.ConvTranspose1d(
            in_channels=emb_chanels,
            out_channels=x_chanels,
            kernel_size=kernel_size,
            padding=padding,
        )
        output_size = get_conv_trans_output_size(
            emb_size, kernel_size, padding, stride
        )
        self.linear = nn.Linear(
            in_features=output_size,
            out_features=window_size
        )

    def forward(self, z, *args, **kwargs):
        """z shape is <N, C, L>, where:
        - N is a batch size,
        - C is features,
        - L is sequence len."""
        emb = F.relu(self.conv_trans(z))
        x = self.linear(emb)
        x = torch.transpose(x, -2, -1)
        return x


class ConvAE(nn.Module):
    def __init__(
        self,
        window_size: int,
        c_in: int,
        n_kernels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super(ConvAE, self).__init__()
        self.params = dict(
            window_size=window_size, c_in=c_in, n_kernels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride
        )
        self.encoder = ConvEncoder(
            window_size=window_size, x_chanels=c_in, emb_chanels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride)
        self.decoder = ConvDecoder(
            window_size=window_size, x_chanels=c_in, emb_chanels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride)

    def forward(self, x):
        seq_len = x.size(1)
        emb = self.encoder(x)
        x_hat = self.decode(emb, seq_len=seq_len)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, emb, seq_len: int):
        x_hat = self.decoder(emb, seq_len)
        return x_hat


class ConvVAE(nn.Module):
    def __init__(
        self,
        window_size: int,
        c_in: int,
        n_kernels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super(ConvVAE, self).__init__()
        self.encoder = ConvEncoder(
            window_size=window_size, x_chanels=c_in, emb_chanels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride)
        self.decoder = ConvDecoder(
            window_size=window_size, x_chanels=c_in, emb_chanels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride)
        self.z_mu_dense = nn.Linear(emb_size, emb_size)
        self.z_log_sig_dense = nn.Linear(emb_size, emb_size)

    def reparametrization(self, mu, log_sig):
        eps = torch.randn_like(mu)
        res = mu + eps * torch.exp(log_sig/2.0)
        return res

    def forward(self, x: torch.Tensor):
        z, z_mu, z_log_sig = self.encode(x, return_all=True)
        x_hat = self.decode(z)
        return (x_hat, z_mu, z_log_sig)

    def decode(self, z):
        x_hat = self.decoder(z)
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


class MultipleConvAE(nn.Module):
    """Convolutional AE for every feature."""
    def __init__(
        self,
        window_size: int,
        c_in: int,
        n_kernels: int,
        kernel_size: int,
        emb_size: int,
        padding: int = 0,
        stride: int = 1,
    ):
        super(MultipleConvAE, self).__init__()
        self.params = dict(
            window_size=window_size, c_in=c_in, n_kernels=n_kernels,
            kernel_size=kernel_size, emb_size=emb_size, padding=padding,
            stride=stride
        )
        self.c_in = c_in

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

    def forward(self, x):
        x_hat = []
        for i in range(self.c_in):
            z_i = self.encoders[i](x[:, :, i:i+1])
            x_i_hat = self.decoders[i](z_i)
            x_hat += [x_i_hat]
        x_hat = torch.concat(x_hat, dim=-1)

        return x_hat


class ConvAEWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: ConvAE = nn.Module(),
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

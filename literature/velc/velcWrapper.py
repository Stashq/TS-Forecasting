from torch import nn, optim
from typing import Dict, Generator, List
import torch
from torch.nn.parameter import Parameter
from sklearn.preprocessing import MinMaxScaler
# from pathlib import Path

from literature.anomaly_detector_base import AnomalyDetector
from predpy.wrapper import Reconstructor
# from predpy.data_module.multi_time_series_module import (
#     MultiTimeSeriesDataloader)
from .velc import VELC


class VELCWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: VELC = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None,
        alpha: float = 0.5, beta: float = 0.5
    ):
        AnomalyDetector.__init__(self)
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.alpha = alpha
        self.beta = beta

    def get_loss(
        self, x, x_dash, z_dash, z_mu, z_log_sig,
        re_z_dash, re_z_mu, re_z_log_sig
    ):
        loss_x = self.criterion(x, x_dash)
        loss_kl_z = self.get_kld_loss(z_mu, z_log_sig)
        loss_kl_re_z = self.get_kld_loss(re_z_mu, re_z_log_sig)
        loss_z = self.criterion(z_dash, re_z_dash)

        loss = loss_x + loss_kl_z + loss_kl_re_z + loss_z
        return loss

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        res = self.model(x)
        loss = self.get_loss(x, *res)
        return loss

    def predict(self, x):
        with torch.no_grad():
            x_dash, _, _, _, _, _, _ = self(x)
            return x_dash

    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False
    ) -> float:
        x_dash, z_dash, _, _, re_z_dash, _, _ = self.model(x)
        score = self.alpha * torch.linalg.norm(x - x_dash)
        score += self.beta * torch.linalg.norm(z_dash - re_z_dash)
        if scale:
            score = self.scaler(score)
        if return_pred:
            return score, x_dash
        return score

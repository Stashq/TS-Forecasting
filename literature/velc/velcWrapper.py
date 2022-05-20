from torch import nn, optim
from typing import Dict, Generator, List
import torch
from torch.nn.parameter import Parameter
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from literature.anomaly_detector_base import AnomalyDetector
from predpy.wrapper import Reconstructor
from predpy.data_module.multi_time_series_module import (
    MultiTimeSeriesDataloader)
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
        super(VELCWrapper, self).__init__(
            model=model, lr=lr, criterion=criterion,
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

    # def calculate_anomaly_score(self, x) -> float:
    #     x_dash, z_dash, _, _, re_z_dash, _, _ = self.model(x)
    #     score = self.alpha * torch.linalg.norm(x - x_dash, ord=1)
    #     score += self.beta * torch.linalg.norm(z_dash - re_z_dash, ord=1)
    #     return score.float()

    # def fit_anom_score_scaler(self, dataloader):
    #     scores = []
    #     for x, _ in dataloader:
    #         s = self.calculate_anomaly(x)
    #         scores += [s]
    #     self.scaler.fit(scores)

    # def anomaly_score(self, x) -> float:
    #     score = self.calculate_anomaly(x)
    #     score = self.scaler(score)
    #     return score

    def fit_detector(
        self,
        normal_data: MultiTimeSeriesDataloader,
        anomaly_data: MultiTimeSeriesDataloader,
        class_weight: Dict = {0: 0.5, 1: 0.5},  # {0: 0.8, 1: 0.2},
        save_path: Path = None,
        plot: bool = False
    ):
        n_preds, a_preds, n_scores, a_scores = [], [], [], []
        for x, _ in normal_data:
            x_dash, z_dash, _, _, re_z_dash, _, _ = self.model(x)
            score = self.alpha * torch.linalg.norm(x - x_dash, ord=1)
            score += self.beta * torch.linalg.norm(z_dash - re_z_dash, ord=1)
            n_scores += [score.float()]
            n_preds += [x_dash]

        for x, _ in anomaly_data:
            x_dash, z_dash, _, _, re_z_dash, _, _ = self.model(x)
            score = self.alpha * torch.linalg.norm(x - x_dash, ord=1)
            score += self.beta * torch.linalg.norm(z_dash - re_z_dash, ord=1)
            a_scores += [score.float()]
            a_preds += [x_dash]

        scores = n_scores + a_scores
        classes = [0]*len(n_scores) + [1]*len(a_scores)
        if save_path is not None:
            self.save_anom_scores(scores, classes, save_path)

        self.fit_thresholder(
            scores=scores, classes=classes, scaler=self.scaler,
            class_weight=class_weight)

        if plot:
            n_preds = torch.cat(n_preds).numpy()
            a_preds = torch.cat(a_preds).numpy()

            n_preds = self.preds_to_df(
                normal_data, n_preds, return_quantiles=True)
            a_preds = self.preds_to_df(
                anomaly_data, a_preds, return_quantiles=True)
            # TODO: plotting


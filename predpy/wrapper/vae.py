"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
# import torch.nn.functional as F
from typing import Dict, List

from .autoencoder import Autoencoder


class VAE(Autoencoder):
    """Lightning module with functionalities for time series prediction
    models.\n

    To work properly, dataset module __getitem__ method should return
    dictionary with model input sequence named "sequence" and following after
    it target value named "label". It also has to have method called
    *get_labels* which return labels from selected range. Class is compatibile
    with dataset classes from :py:mod:`dataset`.

    Parameters
    ----------
    LightningModule : [type]
        Lightning module class.
    """
    def __init__(
        self,
        model: nn.Module = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        kld_weight: float = 0.1
    ):
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids)
        self.kld_weight = kld_weight

    def get_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        z_mu: torch.Tensor,
        z_log_sig: torch.Tensor
    ) -> dict:
        recons_loss = self.criterion(x_hat, x)
        kld_loss = self.get_kld_loss(z_mu, z_log_sig)
        loss = recons_loss + self.kld_weight * kld_loss
        return loss

    def step(self, batch):
        x, _ = self.get_Xy(batch)

        x_hat, z_mu, z_log_sig = self(x)
        loss = self.get_loss(x_hat, x, z_mu, z_log_sig)
        return loss

    def predict(self, x):
        with torch.no_grad():
            x_hat, _, _ = self(x)
            return x_hat

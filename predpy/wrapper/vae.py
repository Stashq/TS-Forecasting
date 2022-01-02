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
        kld_weight: float = 1.0
    ):
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids)
        self.kld_weight = kld_weight

    def get_loss(
        self,
        recons: torch.Tensor,
        input: torch.Tensor,
        mu: torch.Tensor,
        log_sig: torch.Tensor
    ) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) =
        \\log \frac{1}{\\sigma} + \frac{\\sigma^2 + \\mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        # Account for the minibatch samples from the dataset

        # recons_loss = F.mse_loss(recons, input)
        recons_loss = self.criterion(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_sig - mu ** 2 - log_sig.exp(), dim=-1),
            dim=0
        )

        loss = recons_loss + self.kld_weight * kld_loss
        return loss

    def training_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, z_mu, z_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, sequences, z_mu, z_log_sig)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, z_mu, z_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, sequences, z_mu, z_log_sig)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, z_mu, z_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, sequences, z_mu, z_log_sig)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict(self, sequence):
        with torch.no_grad():
            # x_tilda, z_mu, z_log_sig = self(sequence)
            # result = x_tilda.tolist(), z_mu.tolist(), z_log_sig.tolist()
            result, _, _ = self(sequence)
            return result

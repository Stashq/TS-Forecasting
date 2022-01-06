"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
from typing import Dict, List

from .autoencoder import Autoencoder


class PAE(Autoencoder):
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
        target_cols_ids: List[int] = None
    ):
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids)

    def training_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, x_mu, x_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, x_mu, x_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        x_tilda, x_mu, x_log_sig = self(sequences)
        loss = self.get_loss(x_tilda, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict(self, sequence, get_log_sig: bool = False):
        with torch.no_grad():
            _, x_mu, x_log_sig = self(sequence)

        if get_log_sig:
            res = x_mu, x_log_sig
        else:
            res = x_mu
        return res
"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
from typing import Dict, List

from .base import TSModelWrapper


class Autoencoder(TSModelWrapper):
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
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs)
        self.target_cols_ids = target_cols_ids

    def get_Xy(self, batch):
        if self.target_cols_ids is None:
            X = batch["sequence"]
            y = X
        else:
            X = torch.index_select(
                batch["sequence"], dim=1,
                index=torch.tensor(self.target_cols_ids, device=self.device))
            y = X
        return X, y

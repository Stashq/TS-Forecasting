"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
from typing import Dict, List, Union, Tuple
from sklearn.base import TransformerMixin
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from .base import Reconstructor
from predpy.dataset import MultiTimeSeriesDataloader


class Autoencoder(Reconstructor):
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

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat = self(x)

        loss = self.get_loss(x, x_hat)
        return loss

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, emb):
        return self.model.decode(emb)

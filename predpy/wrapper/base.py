"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import pandas as pd
from pytorch_lightning import LightningModule
from torch import nn, optim
from typing import Dict
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin
from predpy.dataset import MultiTimeSeriesDataloader
import torch


class ModelWrapper(LightningModule, ABC):
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
        optimizer_kwargs: Dict = {}
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.OptimizerClass = OptimizerClass
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.params_to_train = None

    def forward(self, x):
        return self.model(x)

    def get_loss(self, output, labels):
        return self.criterion(output, labels)

    @abstractmethod
    def get_Xy(self, batch):
        pass

    @abstractmethod
    def step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        if self.params_to_train is None:
            opt = self.OptimizerClass(
                self.parameters(), self.lr, **self.optimizer_kwargs)
        else:
            opt = self.OptimizerClass(
                self.params_to_train, self.lr, **self.optimizer_kwargs)
        return opt

    def predict(self, sequence):
        with torch.no_grad():
            return self(sequence)

    @abstractmethod
    def get_dataset_predictions(
        self,
        dataloader: MultiTimeSeriesDataloader,
        scaler: TransformerMixin = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def preds_to_dataframe(
        self,
        dataloader: MultiTimeSeriesDataloader,
        preds: torch.Tensor
    ) -> pd.DataFrame:
        pass

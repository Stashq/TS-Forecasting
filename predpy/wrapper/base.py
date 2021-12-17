"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
from pytorch_lightning import LightningModule
import torch
from torch import nn, optim
from typing import Dict
from torch.utils.data import DataLoader
from sklearn.base import TransformerMixin
from tqdm.auto import tqdm
from abc import ABC, abstractmethod


class TSModelWrapper(LightningModule, ABC):
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

    def forward(self, x):
        return self.model(x)

    def get_loss(self, output, labels):
        return self.criterion(output, labels)

    @abstractmethod
    def get_Xy(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        loss = self.get_loss(self(sequences), labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        loss = self.get_loss(self(sequences), labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences, labels = self.get_Xy(batch)

        loss = self.get_loss(self(sequences), labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self.OptimizerClass(
            self.parameters(), self.lr, **self.optimizer_kwargs)

    def predict(self, sequence):
        with torch.no_grad():
            return self(sequence).tolist()

    def get_dataset_predictions(
        self,
        dataloader: DataLoader,
        scaler: TransformerMixin = None
    ):
        self.eval()
        preds = []

        for batch in tqdm(dataloader, desc="Making predictions"):
            preds += self.predict(self.get_Xy(batch)[0])

        if scaler is not None:
            preds = scaler.inverse_transform([preds]).tolist()

        return preds

"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
from pytorch_lightning import LightningModule
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from typing import Dict
from torch.utils.data import DataLoader
from sklearn.base import TransformerMixin
from tqdm.auto import tqdm


class Predictor(LightningModule):
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

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels.unsqueeze(dim=1))
        return loss, output

    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, _ = self(sequences, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, _ = self(sequences, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, _ = self(sequences, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return self.OptimizerClass(
            self.parameters(), self.lr, **self.optimizer_kwargs)

    def predict(self, sequence):
        with torch.no_grad():
            _, output = self(sequence)
            return output.squeeze().tolist()

    def get_dataset_predictions(
        self,
        dataloader: DataLoader,
        scaler: TransformerMixin = None
    ):
        self.eval()
        preds = []

        for data in tqdm(dataloader, desc="Making predictions"):
            tmp = self.predict(data["sequence"])
            preds += tmp

        if scaler is not None:
            preds = scaler.inverse_transform([preds]).tolist()[0]

        return preds

    def predict_and_plot(
        self,
        dataloader: DataLoader,
        target: str,
        scaler: TransformerMixin = None
    ):
        self.eval()
        preds = self.get_dataset_predictions(dataloader, scaler)

        labels = dataloader.dataset.get_labels()
        x, true_vals = labels.index, labels[target]

        if scaler is not None:
            true_vals = scaler.inverse_transform([true_vals])[0]

        plt.plot(x, preds, label="predictions")
        plt.plot(x, true_vals, label="true values")
        plt.legend()
        plt.show()

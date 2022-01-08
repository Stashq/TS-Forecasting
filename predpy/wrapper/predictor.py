"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
from torch import nn, optim
from typing import Dict
from .base import ModelWrapper
from sklearn.base import TransformerMixin
from predpy.dataset import MultiTimeSeriesDataloader
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import torch


class Predictor(ModelWrapper):
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
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs)

    def get_Xy(self, batch):
        X = batch["sequence"]
        y = batch["label"]
        return X, y

    def step(self, batch):
        sequences, labels = self.get_Xy(batch)

        loss = self.get_loss(self(sequences), labels)
        return loss

    def get_dataset_predictions(
        self,
        dataloader: MultiTimeSeriesDataloader,
        scaler: TransformerMixin = None
    ) -> pd.DataFrame:
        self.eval()
        preds = []

        for batch in tqdm(dataloader, desc="Making predictions"):
            x = self.get_Xy(batch)[0]
            preds += [self.predict(x)]

        preds = torch.cat(preds)
        preds = preds.numpy()

        if scaler is not None:
            preds =\
                scaler.inverse_transform([preds]).tolist()

        return self.preds_to_dataframe(dataloader, preds)

    def preds_to_dataframe(
        self,
        dataloader: MultiTimeSeriesDataloader,
        preds: np.ndarray
    ) -> pd.DataFrame:
        indices = dataloader.dataset.get_indices_like_recs(labels=True)
        columns = dataloader.dataset.target
        df = pd.DataFrame(preds, columns=columns, index=indices)

        return df

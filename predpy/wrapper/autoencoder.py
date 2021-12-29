"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
from typing import Dict, List
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

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

    def get_dataset_predictions(
        self,
        dataloader: DataLoader,
        scaler: TransformerMixin = None
    ) -> pd.DataFrame:
        self.eval()
        preds = []

        for batch in tqdm(dataloader, desc="Making predictions"):
            x = self.get_Xy(batch)[0]
            preds += [self.predict(x)]

        preds = torch.cat(preds)
        preds = preds.transpose(1, 2)
        preds = preds.view(-1, preds.shape[-1])
        preds = preds.tolist()

        if scaler is not None:
            preds =\
                scaler.inverse_transform([preds]).tolist()

        ids = dataloader.dataset.get_indices_like_recs(labels=False)
        ids = pd.concat([ids.to_series() for ids in ids])
        columns = dataloader.dataset.target
        preds_df = pd.DataFrame(preds, columns=columns, index=ids)
        df = pd.DataFrame(ids.unique(), columns=["datetime"])\
            .set_index("datetime", drop=True)

        for col in columns:
            grouped = preds_df.groupby(preds_df.index)
            df[col + "_q000"] = grouped.quantile(0.0)
            df[col + "_q025"] = grouped.quantile(0.25)
            df[col + "_q050"] = grouped.quantile(0.5)
            df[col + "_q075"] = grouped.quantile(0.75)
            df[col + "_q100"] = grouped.quantile(1.0)

        return df

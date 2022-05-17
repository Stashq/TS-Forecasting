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

    def get_Xy(self, batch):
        if self.target_cols_ids is None:
            X = batch["sequence"]
            y = X
        else:
            X = torch.index_select(
                batch["sequence"], dim=-1,
                index=torch.tensor(self.target_cols_ids, device=self.device))
            y = X
        return X, y

    def step(self, batch):
        sequences, labels = self.get_Xy(batch)
        preds = self(sequences)

        loss = self.get_loss(preds, labels)
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
        # preds = preds.transpose(1, 2)
        # preds = preds.reshape(-1, preds.shape[-1])
        preds = preds.numpy()

        if scaler is not None:
            preds =\
                scaler.inverse_transform([preds]).tolist()

        return self.preds_to_df(dataloader, preds, return_quantiles=True)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, emb):
        return self.model.decode(emb)

    def preds_to_df(
        self,
        dataloader: MultiTimeSeriesDataloader,
        preds: np.ndarray,
        return_quantiles: bool = True,
        return_raw_preds: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame]]:
        if len(preds.shape) == 3:
            batch_size, seq_len = preds.shape[0], preds.shape[1]
            preds = preds.reshape(batch_size * seq_len, -1)
        ids = dataloader.dataset.get_indices_like_recs(labels=False)
        ids = pd.concat(ids)
        columns = dataloader.dataset.target
        preds_df = pd.DataFrame(preds, columns=columns, index=ids)

        if return_quantiles and return_raw_preds:
            return [
                self._preds_df_to_quantiles_df(preds_df, columns), preds_df]
        if return_quantiles:
            return self._preds_df_to_quantiles_df(preds_df, columns)
        elif return_raw_preds:
            return preds_df
        else:
            return None

    def _preds_df_to_quantiles_df(
        self,
        preds_df: pd.Series,
        columns: List[str]
    ) -> pd.DataFrame:
        # df = pd.DataFrame(preds_df.index.unique(), columns=["datetime"])\
        #     .set_index("datetime", drop=True)

        data = {}
        for col in columns:
            grouped = preds_df[col].groupby(preds_df.index)
            data[str(col) + "_q000"] = grouped.quantile(0.0)
            data[str(col) + "_q025"] = grouped.quantile(0.25)
            data[str(col) + "_q050"] = grouped.quantile(0.5)
            data[str(col) + "_q075"] = grouped.quantile(0.75)
            data[str(col) + "_q100"] = grouped.quantile(1.0)

        df = pd.DataFrame(data, index=preds_df.index.unique())
        return df

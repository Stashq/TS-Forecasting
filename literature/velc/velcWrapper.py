import numpy as np
import pandas as pd
from torch import nn, optim
from typing import Dict, List, Tuple, Union
from sklearn.base import TransformerMixin
from predpy.dataset import MultiTimeSeriesDataloader
import torch
from tqdm.auto import tqdm

from predpy.wrapper import Reconstructor
from .velc import VELC


class VELCWrapper(Reconstructor):
    def __init__(
        self,
        model: VELC = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        target_cols_ids: List[int] = None,
        optimizer_kwargs: Dict = {}
    ):
        super(VELCWrapper, self).__init__()
        self.model = model
        self.criterion = criterion
        self.OptimizerClass = OptimizerClass
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.params_to_train = None
        self.target_cols_ids = target_cols_ids

    def forward(self, x):
        return self.model(x)

    def get_loss(
        self, x, x_dash, z_dash, z_mu, z_log_sig,
        re_z_dash, re_z_mu, re_z_log_sig
    ):
        loss_x = self.criterion(x, x_dash)
        loss_kl_z = self.get_kld_loss(z_mu, z_log_sig)
        loss_kl_re_z = self.get_kld_loss(re_z_mu, re_z_log_sig)
        loss_z = self.criterion(z_dash, re_z_dash)

        loss = loss_x + loss_kl_z + loss_kl_re_z + loss_z
        return loss

    def get_kld_loss(self, mu, log_sig):
        loss = torch.mean(
            -0.5 * torch.sum(1 + log_sig - mu ** 2 - log_sig.exp(), dim=-1),
            dim=0
        )
        return loss

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
        x, _ = self.get_Xy(batch)
        res = self.model(x)
        loss = self.get_loss(x, *res)
        return loss

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
            x_dash, _, _, _, _, _, _ = self(sequence)
            return x_dash

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

        return self.preds_to_df(dataloader, preds, return_quantiles=True)

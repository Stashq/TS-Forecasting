import torch
from torch import nn, optim
from typing import Dict, List, Union, Tuple
from sklearn.base import TransformerMixin
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

from .dagmm import DAGMM
from predpy.dataset import MultiTimeSeriesDataloader
from predpy.wrapper import ModelWrapper


class DAGMMWrapper(ModelWrapper):
    def __init__(
        self,
        model: DAGMM = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        lambda_energy: float = 0.1,
        lambda_cov_diag: float = 0.005
    ):
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs)
        self.target_cols_ids = target_cols_ids
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag

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

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, _ = self.get_Xy(batch)

        x_hat, z_c, z, gamma = self(x)
        loss = self.get_loss(
            x, x_hat, z, gamma)
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
        preds = preds.transpose(1, 2)
        preds = preds.reshape(-1, preds.shape[-1])
        preds = preds.numpy()

        if scaler is not None:
            preds =\
                scaler.inverse_transform([preds]).tolist()

        return self.preds_to_quantiles_df(dataloader, preds)

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
            preds = preds.reshape(preds.shape[0] * preds.shape[2], -1)
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
        df = pd.DataFrame(preds_df.index.unique(), columns=["datetime"])\
            .set_index("datetime", drop=True)

        for col in columns:
            grouped = preds_df.groupby(preds_df.index)
            df[col + "_q000"] = grouped.quantile(0.0)
            df[col + "_q025"] = grouped.quantile(0.25)
            df[col + "_q050"] = grouped.quantile(0.5)
            df[col + "_q075"] = grouped.quantile(0.75)
            df[col + "_q100"] = grouped.quantile(1.0)

        return df

    def get_loss(
        self, x, x_hat, z, gamma
    ):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.model.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.model.compute_energy(z, phi, mu, cov)
        loss = recon_error + self.lambda_energy * sample_energy\
            + self.lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag

    def predict(self, sequence):
        with torch.no_grad():
            x_hat, z_c, z, gamma = self(sequence)
            return x_hat

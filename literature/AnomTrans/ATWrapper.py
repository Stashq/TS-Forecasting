import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
from torch import nn, optim
from typing import Dict, List, Tuple, Union
from sklearn.base import TransformerMixin
from predpy.dataset import MultiTimeSeriesDataloader
import torch
from tqdm.auto import tqdm

from .AnomTrans import AnomalyTransformer


class ATWrapper(LightningModule):
    def __init__(
        self,
        model: AnomalyTransformer = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        target_cols_ids: List[int] = None,
        optimizer_kwargs: Dict = {}
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.OptimizerClass = OptimizerClass
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.params_to_train = None
        self.target_cols_ids = target_cols_ids
        # self.automatic_optimization = False

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, x):
        return self.model(x)

    def get_loss(self, output, labels):
        return self.criterion(output, labels)

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

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        return frob_norm - (
            lambda_
            * torch.linalg.norm(
                self.model.association_discrepancy(P_list, S_list, x),
                ord=1)
        )

    def min_loss(self, x, x_hat):
        P_list = self.model.P_layers
        S_list = [S.detach() for S in self.model.S_layers]
        lambda_ = -self.model.lambda_
        return self.loss_function(x_hat, P_list, S_list, lambda_, x).mean()

    def max_loss(self, x, x_hat):
        P_list = [P.detach() for P in self.model.P_layers]
        S_list = self.model.S_layers
        lambda_ = self.model.lambda_
        return self.loss_function(x_hat, P_list, S_list, lambda_, x).mean()

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat = self.model(x)
        min_loss = self.min_loss(x, x_hat)
        max_loss = 0  # self.max_loss(x, x_hat)
        return min_loss, max_loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def training_step(self, batch, batch_idx):
        opt = self.configure_optimizers()
        # opt.zero_grad()
        min_loss, max_loss = self.step(batch)
        self.manual_backward(
            min_loss
            # self.opt,
            # retain_graph=True
        )
        # self.manual_backward(
        #     max_loss,
        #     # self.opt
        # )
        self.log("train_min_loss", min_loss, prog_bar=True, logger=True)
        # self.log("train_max_loss", max_loss, prog_bar=True, logger=True)
        # opt.step()
        # opt.zero_grad()
        # return min_loss + max_loss

    def validation_step(self, batch, batch_idx):
        min_loss, max_loss = self.step(batch)
        self.log("val_min_loss", min_loss, prog_bar=True, logger=True)
        self.log("val_max_loss", max_loss, prog_bar=True, logger=True)
        return min_loss + max_loss

    def test_step(self, batch, batch_idx):
        min_loss, max_loss = self.step(batch)
        self.log("test_min_loss", min_loss, prog_bar=True, logger=True)
        self.log("test_max_loss", max_loss, prog_bar=True, logger=True)
        return min_loss + max_loss

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

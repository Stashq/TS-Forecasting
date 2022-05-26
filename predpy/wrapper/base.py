"""ModelWrapper, Predictor and Reconstructor.
"""
import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule
from torch import nn, optim
from typing import Dict, Generator, List, Tuple, Union
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin
from predpy.dataset import MultiTimeSeriesDataloader
import torch
from torch.nn.parameter import Parameter
from tqdm.auto import tqdm


class ModelWrapper(LightningModule, ABC):
    """Lightning module with functionalities for time series
    prediction / reconstruction models.\n

    To work properly, dataset module __getitem__ method should return
    dictionary with model input sequence named "sequence" and following after
    it target value named "label". It also has to have method called
    *get_labels* which return labels from selected range. Class is compatibile
    with dataset classes from :py:mod:`predpy.dataset`.

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
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None
    ):
        super().__init__()
        # LightningModule.__init__(self)
        # ABC.__init__(self)
        self.model = model
        self.criterion = criterion
        self.OptimizerClass = OptimizerClass
        self.optimizer_kwargs = optimizer_kwargs
        self.lr = lr
        self.target_cols_ids = target_cols_ids
        self.params_to_train = params_to_train
        self.val_mse = nn.MSELoss()

    @abstractmethod
    def get_loss(self, output, labels):
        pass

    @abstractmethod
    def step(self, batch):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def get_Xy(self, batch):
        pass

    @abstractmethod
    def get_dataset_predictions(
        self,
        dataloader: MultiTimeSeriesDataloader,
        scaler: TransformerMixin = None
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def preds_to_df(
        self,
        dataloader: MultiTimeSeriesDataloader,
        preds: np.ndarray
    ) -> pd.DataFrame:
        pass

    def forward(self, x):
        return self.model(x)

    def val_step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat = self.predict(x)
        loss = self.val_mse(x, x_hat)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.val_step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.val_step(batch)
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

        return self.preds_to_df(dataloader, preds)

    def preds_to_df(
        self,
        dataloader: MultiTimeSeriesDataloader,
        preds: np.ndarray
    ) -> pd.DataFrame:
        indices = dataloader.dataset.get_indices_like_recs(labels=True)
        columns = dataloader.dataset.target
        df = pd.DataFrame(preds, columns=columns, index=indices)

        return df


class Reconstructor(ModelWrapper):
    def __init__(
        self,
        model: nn.Module = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None
    ):
        super(Reconstructor, self).__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids, params_to_train=params_to_train)
        self.target_cols_ids = target_cols_ids
        self.params_to_train = params_to_train

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

    def get_kld_loss(self, mu, log_sig):
        loss = torch.mean(torch.mean(
            -0.5 * torch.sum(1 + log_sig - mu ** 2 - log_sig.exp(), dim=-1),
            dim=1), dim=0)
        # loss = torch.mean(
        #     -0.5 * torch.sum(1 + log_sig - mu ** 2 - log_sig.exp(), dim=-1),
        #     dim=0)
        return loss

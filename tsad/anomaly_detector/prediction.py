from .base import AnomalyDetector
from predpy.wrapper import ModelWrapper
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from predpy.dataset import MultiTimeSeriesDataloader


class PredictionAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: ModelWrapper,
        distributor: Distributor = Gaussian(),
    ):
        super().__init__(
            time_series_model, distributor)

    def forward(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        with torch.no_grad():
            preds = self.time_series_model.predict(sequences)
            errors = torch.abs(labels - preds).cpu().detach().numpy()

        if return_predictions:
            return errors, preds
        return errors

    def dataset_forward(
        self,
        dataloader: MultiTimeSeriesDataloader,
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        iterator = None
        if verbose:
            iterator = tqdm(dataloader, desc="Time series predictions")
        else:
            iterator = dataloader

        preds = []
        labels = []
        for batch in iterator:
            labels += [batch["label"]]
            with torch.no_grad():
                preds += self.time_series_model.predict(batch["sequence"])
        preds = torch.cat(preds, 0)
        if len(preds.shape) == 1:
            preds = torch.unsqueeze(preds, 1)
        labels = torch.cat(labels, 0)
        errors = torch.abs(labels - preds).cpu().detach().numpy()

        if return_predictions:
            preds = self.time_series_model.preds_to_dataframe(
                dataloader, preds.numpy())
            return errors, preds
        return errors

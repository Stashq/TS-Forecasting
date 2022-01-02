from .base import AnomalyDetector
from predpy.wrapper import TSModelWrapper
import torch
import numpy as np
from typing import Union, Tuple
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


class PredictionAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: TSModelWrapper,
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
        data: Union[DataLoader, Dataset],
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        iterator = None
        if verbose:
            iterator = tqdm(data, desc="Time series predictions")
        else:
            iterator = data

        preds = []
        labels = []
        for data in iterator:
            labels += [data["label"]]
            with torch.no_grad():
                preds += self.time_series_model.predict(data["sequence"])
        preds = torch.cat(preds, 0)
        if len(preds.shape) == 1:
            preds = torch.unsqueeze(preds, 1)
        labels = torch.cat(labels, 0)
        errors = torch.abs(labels - preds).cpu().detach().numpy()

        if return_predictions:
            return errors, preds
        return errors

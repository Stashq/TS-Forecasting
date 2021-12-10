from .base import AnomalyDetector
import torch
import numpy as np
from typing import Type
from tsad.distributor import Distributor, GaussianDistributor


class PredictionAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: torch.nn.Module,
        DistributorCls: Type[Distributor] = GaussianDistributor,
        **distributor_kwargs
    ):
        super().__init__(
            time_series_model, DistributorCls, **distributor_kwargs)

    def forward(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        with torch.no_grad():
            preds = self.time_series_model(sequences)
            errors = torch.abs(labels - preds).cpu().detach().numpy()
        return errors

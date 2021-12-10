from tsad.distributor import Distributor, GaussianDistributor

import torch
from tqdm.auto import tqdm
import numpy as np
from abc import abstractmethod
from typing import Type, Union, Tuple, Dict
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from string import Template

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, DataLoader.")


class AnomalyDetector:
    def __init__(
        self,
        time_series_model: torch.nn.Module,
        DistributorCls: Type[Distributor] = GaussianDistributor,
        **distributor_kwargs
    ):
        self.time_series_model = time_series_model
        self.time_series_model.eval()
        self.distributor = DistributorCls(**distributor_kwargs)
        self.threshold = None

    @abstractmethod
    def forward(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        pass

    def dataset_forward(
        self,
        dataloader: DataLoader,
        verbose: bool = False
    ) -> np.ndarray:
        results = []
        iterator = None
        if verbose:
            iterator = tqdm(dataloader, desc="Time series predictions")
        else:
            iterator = dataloader

        for data in iterator:
            results += [self.forward(data["sequence"], data["label"])]
        results = np.concatenate(results, axis=0)
        return results

    def fit_distributor(
        self,
        data: Union[torch.Tensor, DataLoader],
        verbose: bool = False
    ):
        result = self._any_forward(data, verbose)
        self.distributor.fit(result)

    def _any_forward(
        self,
        data: Union[torch.Tensor, DataLoader],
        verbose: bool = False
    ) -> np.ndarray:
        result = None
        if isinstance(data, torch.Tensor):
            result = self.forward(data["sequence"], data["label"])
        elif isinstance(data, DataLoader):
            result = self.dataset_forward(data, verbose=verbose)
        else:
            raise ValueError(
                UNKNOWN_TYPE_MSG.substitute(data_type=type(data)))
        return result

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, DataLoader],
        anomaly_data: Union[torch.Tensor, DataLoader],
        class_weight: Dict = {0: 0.05, 1: 0.95},
        verbose: bool = False
    ):
        """Fits logistic regressor on distributor probabilities of normal
        and anomaly data.

        Parameters
        ----------
        normal_data : Union[torch.Tensor, DataLoader]
            Data that is not anomaly.
        anomaly_data : Union[torch.Tensor, DataLoader]
            Anomalies.
        class_weight : Dict, optional
            Normal data (0) and anomaly (1) weights,
            by default {0: 0.05, 1: 0.95}.
        """
        normal_data = self._any_forward(normal_data, verbose)
        anomaly_data = self._any_forward(anomaly_data, verbose)
        X = np.concatenate([
            self.distributor.probs(normal_data),
            self.distributor.probs(anomaly_data)])
        y = [0]*len(normal_data) + [1]*len(anomaly_data)
        self.threshold = LogisticRegression(
            class_weight=class_weight
        ).fit(X, y)

    def fit(
        self,
        train_data: Union[torch.Tensor, DataLoader],
        anomaly_data: Union[torch.Tensor, DataLoader],
        normal_data: Union[torch.Tensor, DataLoader] = None,
        class_weight: Dict = {0: 0.05, 1: 0.95}
    ):
        # if normal_data is None:
        #     normal_data = np.array([
        #         seq.cpu().detach().numpy()
        #         for seq, _ in train_data.dataset
        #     ])
        self.fit_distributor(train_data)
        self.fit_threshold(normal_data, anomaly_data, class_weight)

    def find_anomalies(
        self,
        dataloader: DataLoader,
        return_indices: bool = True,
        verbose: bool = False
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        preds = self.dataset_forward(dataloader, verbose)
        probs = self.distributor.probs(preds)
        result = self.threshold.predict(probs)
        if return_indices is True:
            result = np.argwhere(result == 1)
        return result

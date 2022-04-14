from .base import AnomalyDetector
from predpy.wrapper import Autoencoder
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from predpy.dataset import MultiTimeSeriesDataloader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


class EmbeddingAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: Autoencoder,
        distributor: Distributor = Gaussian(),
        target_cols_ids: List[int] = None,
    ):
        if not issubclass(type(time_series_model), Autoencoder):
            raise ValueError(
                "Time series model has to inherit from Autoencoder")
        super().__init__(
            time_series_model, distributor)
        self.target_cols_ids = target_cols_ids

    def get_seqences(self, batch):
        if next(self.time_series_model.parameters()).is_cuda:
            device = self.time_series_model.get_device()
        else:
            device = "cpu"

        if self.target_cols_ids is None:
            seqs = batch["sequence"]
        else:
            seqs = torch.index_select(
                batch["sequence"], dim=1,
                index=torch.tensor(self.target_cols_ids, device=device))
        return seqs

    def forward(
        self,
        records: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        with torch.no_grad():
            seqs = self.get_seqences(records)
            embs = self.time_series_model.encode(seqs).cpu().detach().numpy()

        if return_predictions:
            return embs, None
        return embs

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

        embs = []
        for batch in iterator:
            batch_seqs = self.get_seqences(batch)
            with torch.no_grad():
                embs += self.time_series_model.encode(batch_seqs)
        embs = torch.cat(embs, 0).cpu().detach().numpy()

        if return_predictions:
            return embs, None
        return embs

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        class_weight: Dict = {0: 0.5, 1: 0.5},  # {0: 0.05, 1: 0.95},
        verbose: bool = False,
        plot_distribution: bool = False,
        plot_time_series: bool = False,
        plot_embeddings: bool = False
    ):
        """Fits logistic regressor on distributor probabilities of normal
        and anomaly data.

        Parameters
        ----------
        normal_data : Union[torch.Tensor, MultiTimeSeriesDataloader]
            Data that is not anomaly.
        anomaly_data : Union[torch.Tensor, MultiTimeSeriesDataloader]
            Anomalies.
        class_weight : Dict, optional
            Normal data (0) and anomaly (1) weights,
            by default {0: 0.05, 1: 0.95}.
        """
        n_embs = self._any_forward(
            normal_data, verbose, return_predictions=False)
        a_embs = self._any_forward(
            anomaly_data, verbose, return_predictions=False)
        probs = np.concatenate([
                self.distributor.predict(n_embs),
                self.distributor.predict(a_embs)
            ], axis=0)
        classes = [0]*len(n_embs) + [1]*len(a_embs)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(probs, classes)
        pred_cls = self.thresholder.predict(probs)

        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        self._plot_fit_results(
            plot_time_series=False,
            plot_embeddings=plot_embeddings,
            plot_distribution=False,
            n_res=None, a_res=None,
            classes=classes, pred_cls=pred_cls,
            n_data=normal_data, a_data=anomaly_data,
            model_n_ts_preds=None,
            model_a_ts_preds=None,
            n_embs=n_embs, a_embs=a_embs)

from .base import AnomalyDetector
from predpy.wrapper import Autoencoder
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from predpy.dataset import MultiTimeSeriesDataloader


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

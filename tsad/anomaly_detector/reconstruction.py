from .base import AnomalyDetector
from predpy.wrapper import ModelWrapper
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from predpy.dataset import MultiTimeSeriesDataloader


class ReconstructionAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: ModelWrapper,
        distributor: Distributor = Gaussian(),
        target_cols_ids: List[int] = None,
    ):
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
            preds = self.time_series_model.predict(seqs)
            errors = torch.abs(seqs - preds).cpu().detach().numpy()

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
        seqs = []
        for batch in iterator:
            batch_seqs = self.get_seqences(batch)
            seqs += [batch_seqs]
            with torch.no_grad():
                preds += self.time_series_model.predict(batch_seqs)
        preds = torch.cat(preds, 0)
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(dim=1)
        seqs = torch.cat(seqs, 0)
        errors = torch.abs(seqs - preds).cpu().detach().numpy()

        if return_predictions:
            preds = self.time_series_model.preds_to_df(
                dataloader, preds.numpy())
            return errors, preds
        return errors

from .base import AnomalyDetector
import torch
import numpy as np
from typing import Type, Union, Tuple, List
from tsad.distributor import Distributor, GaussianDistributor
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset


class ReconstructionAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: torch.nn.Module,
        DistributorCls: Type[Distributor] = GaussianDistributor,
        target_cols_ids: List[int] = None,
        **distributor_kwargs
    ):
        super().__init__(
            time_series_model, DistributorCls, **distributor_kwargs)
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
            preds = self.time_series_model()
            errors = torch.abs(seqs - preds).cpu().detach().numpy()

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
        seqs = []
        for batch in iterator:
            batch_seqs = self.get_seqences(batch)
            seqs += [batch_seqs]
            with torch.no_grad():
                preds += self.time_series_model(batch_seqs)
        preds = torch.cat(preds, 0)
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(dim=1)
        seqs = torch.cat(seqs, 0)
        errors = torch.abs(seqs - preds).cpu().detach().numpy()

        if return_predictions:
            return errors, preds
        return errors

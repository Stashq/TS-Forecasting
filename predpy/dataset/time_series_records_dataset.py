"""Module contains *TimeSeriesRecordsDataset* - custom pytorch Dataset class.\n

Splits time series to records.
Every record is a tuple containing a sequence (model input data)
and a single target value following after sequence (predicted value).\n

Warning:\n
Because of redundancy, class instance size grows linearly proportional
to window size.
"""
import torch
import pandas as pd
from predpy.preprocessing import seq_to_records
from .time_series_dataset import TimeSeriesDataset
from typing import Dict


class TimeSeriesRecordsDataset(TimeSeriesDataset):
    """Custom Pytorch Dataset class for single time series.\n

    Splits time series to records.
    Every record is a tuple containing a sequence (model input data)
    and a single target value following after sequence (predicted value).

    Parameters
    ----------
    BaseTimeSeriesDataset : [type]
        [description]
    """
    def __init__(
        self,
        sequence: pd.DataFrame,
        window_size: int,
        target: str
    ):
        """Creates *TimeSeriesRecordsDataset* instance.\n

        Class object stores dataset as records.
        Every record is a tuple containing a sequence (model input data)
        and a single target value following after sequence (predicted value).

        Parameters
        ----------
        sequence : pd.DataFrame
            Single time series (can have multi columns).
        window_size : int
            Window size, defines how long should be one sample.
        target : str
            Name of column containing values to be predicted.
        """
        self.records = seq_to_records(sequence, window_size, target)
        self.window_size = window_size
        self.target = target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[torch.Tensor, torch.Tensor]:
        """Returns dict of:
        * sequence - records sequence,
        * label - records target value followed after sequence.

        Parameters
        ----------
        idx : [int]
            Index of record.

        Returns
        -------
        Dict[torch.Tensor, torch.Tensor]
            Dict containing sequence and label.
        """
        seq, label = self.records[idx]
        return dict(
            sequence=torch.tensor(seq.to_numpy().T).float(),
            label=torch.tensor(label).float()
        )

    def get_labels(
        self,
        start_idx: int = None,
        end_idx: int = None
    ) -> pd.Series:
        """Returns dataset labels from provided range.\n

        If start_idx is None, range will start from 0,
        if end_idx is None, range will end with end of dataset.

        Parameters
        ----------
        start_idx : [type], optional
            First returning index position.
        end_idx : [type], optional
            Last returning index position.

        Returns
        -------
        pd.Series
            Dataset labels.
        """
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = self.__len__()
        labels = [rec[1] for rec in self.records[start_idx:end_idx]]
        return pd.Series(labels)

"""Module contains *SingleTimeSeriesDataset* - custom pytorch Dataset class.\n

Samples shares same memory, so it is strongly advised not to change them during
usage. Created object is much lighter than *TimeSeriesRecodsDataset* object.
"""
import torch
import pandas as pd
from .time_series_dataset import TimeSeriesDataset
from typing import Union, List, Dict


class SingleTimeSeriesDataset(TimeSeriesDataset):
    """Custom Pytorch Dataset class for single time series.\n

    Samples shares same memory, so it is strongly advised not to change them
    during usage. Created object is much lighter than *TimeSeriesRecodsDataset*
    object.

    X are sequences created with moving window method from passed single
    time series, y are target values following after them.

    Parameters
    ----------
    BaseTimeSeriesDataset : [type]
        Abstract class for time series datasets classes.
    """
    def __init__(
        self,
        sequence: pd.DataFrame,
        window_size: int,
        target: Union[List[str], str]
    ):
        """Creates *SingleTimeSeriesDataset* instance.

        Parameters
        ----------
        sequence : pd.DataFrame
            Single time series (can have multi columns).
        window_size : int
            Window size, defines how long should be one sample.
        target : str
            Name of column containing values to be predicted.
        """
        self.sequence = sequence
        self.window_size = window_size
        if not isinstance(target, list):
            target = [target]
        self.target = target

    def __len__(self) -> int:
        """Returns number of samples in dataset.\n

        Due to length of single sample number of records is length of
        sequence minus window size.

        Returns
        -------
        int
            Length of dataset.
        """
        return self.sequence.shape[0] - self.window_size

    def __getitem__(self, idx: int) -> Dict[torch.Tensor, torch.Tensor]:
        """Returns dict of:
        * sequence - sequence starting from indicated position,
        * label - target value followed after sequence.

        Parameters
        ----------
        idx : [int]
            Position in primary sequence of staring the sample.

        Returns
        -------
        Dict[torch.Tensor, torch.Tensor]
            Dict containing sequence and label.
        """
        seq = self.sequence.iloc[idx:idx + self.window_size]
        label = self.sequence.iloc[idx + self.window_size][self.target]
        return dict(
            sequence=torch.tensor(seq.to_numpy().T).float(),
            label=torch.tensor(label.to_numpy()).float()
        )

    def get_labels(
        self,
        start_idx: int = None,
        end_idx: int = None
    ) -> pd.DataFrame:
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
        start = start_idx + self.window_size
        end = end_idx + self.window_size
        return self.sequence.iloc[start:end][self.target]

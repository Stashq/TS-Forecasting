"""Module contains *TimeSeriesDataset* - an abstract class inherited by time
series datasets classes.
"""
from torch.utils.data import Dataset
import pandas as pd
from abc import abstractmethod
from typing import Union, List


class TimeSeriesDataset(Dataset):
    """Abstract class for time series datasets classes.

    Parameters
    ----------
    Dataset : [type]
        Pytorch Dataset class.
    """
    @abstractmethod
    def __init__(
        self,
        sequence: pd.DataFrame,
        window_size: int,
        target: Union[List[str], str]
    ):
        """Creates *TimeSeriesDataset* instance.

        Parameters
        ----------
        sequence : pd.DataFrame
            Single time series (can have multi columns).
        window_size : int
            Window size, defines how long should be one sample.
        target : str
            Name of column containing values to be predicted.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int):
        pass

    @abstractmethod
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
        pass

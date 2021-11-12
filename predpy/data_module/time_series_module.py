"""Lightning data module customized for time series.\n

Split data between training, validation, test datasets and stores them using
*TimeSeriesDataset* classes.
"""
import math
from typing import Union, Tuple, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from predpy.dataset import SingleTimeSeriesDataset, TimeSeriesDataset
import pandas as pd


class TimeSeriesModule(LightningDataModule):
    """Lightning data module customized for time series.

    Parameters
    ----------
    LightningDataModule : [type]
        Lightning data module.
    """
    def __init__(
        self,
        sequence: pd.DataFrame,
        dataset_name: str,
        target: str,
        split_proportions: Union[Tuple[float], List[float]],
        window_size: int,
        batch_size: int = 8,
        DatasetCls: TimeSeriesDataset = SingleTimeSeriesDataset
    ):
        """Creates TimeSeriesModule instance.

        Parameters
        ----------
        sequence : pd.DataFrame
            Single time series (can have multi columns).
        dataset_name : str
            A name with which the dataset will be associated.
        target : str
            A name of column storing values to be predicted.
        split_proportions : List[float]
            List of percentage share in the data set of training, validation
            and test data. Have to add up to 1. If length is 3,
            create 3 datasets in mentioned order,
            if 2, validation and test data will be the same.
        window_size : int
            Length of sequence in every sample.
        batch_size : int, optional
            Batch size, by default 8.
        DatasetCls : TimeSeriesDataset, optional
            Type of dataset class to use, by default SingleTimeSeriesDataset.
        """
        super().__init__()

        # init setup
        assert len(sequence) > window_size, "Sequence too short."
        self._splits_assert(split_proportions)
        self._save_splits_as_ids(
            split_proportions, len(sequence), window_size)

        self.split_proportions = split_proportions
        self.sequence = sequence
        self.name_ = dataset_name
        self.target = target
        self.window_size = window_size
        self.batch_size = batch_size
        self.DatasetCls = DatasetCls

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _splits_assert(
        self,
        split_proportions: Union[Tuple[float], List[float]]
    ):
        """Assert:
        * *split_proportions* is a list or a tuple,
        * split values are non-negative and no greater than 1,
        * *split_proportions* is equal 1,
        otherwise raise AssertionError.
        """
        assert isinstance(split_proportions, (Tuple, List)),\
            "\"split_proportions\" should be a list or a tuple."
        for split_ in split_proportions:
            assert split_ >= 0, "Split can not be negative."
            assert split_ <= 1, "Split can not be greater than 1."
        assert len(split_proportions) in [2, 3], "Wrong number of splits."
        assert math.isclose(sum(split_proportions), 1, rel_tol=1e-6), \
            "Values don't add up to 1."

    def _save_splits_as_ids(
        self,
        split_proportions: Union[Tuple[float], List[float]],
        seq_len: int,
        window_size: int
    ):
        """Saves ranges of training, validation and test datasets.

        Parameters
        ----------
        split_proportions : Union[Tuple[float], List[float]]
            3 values of proportions between datasets.
        seq_len : int
            Original time series length.
        window_size : int
            Length of sequence in every sample.
        """
        # shift is equal to one dataset sample (sequence and target value)
        shift = window_size + 1
        n_samples = seq_len - shift
        train_val_split = int(n_samples * split_proportions[0])
        val_test_split =\
            int(n_samples * (split_proportions[0] + split_proportions[1]))

        self.train_range = (0, train_val_split + shift)
        if len(split_proportions) == 3:
            self.val_range = \
                (train_val_split, val_test_split + shift)
            self.test_range = (val_test_split, seq_len)
        elif len(split_proportions) == 2:
            self.val_range = \
                (train_val_split, seq_len)
            self.test_range = (train_val_split, seq_len)

    def setup(self, stage: str = None):
        start, end = self.train_range
        self.train_dataset = self.DatasetCls(
            self.sequence[start:end], self.window_size, self.target)
        start, end = self.val_range
        self.val_dataset = self.DatasetCls(
            self.sequence[start:end], self.window_size, self.target)
        start, end = self.test_range
        self.test_dataset = self.DatasetCls(
            self.sequence[start:end], self.window_size, self.target)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

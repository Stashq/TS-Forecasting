"""Lightning data module customized for time series.\n

Al

Split data between training, validation, test datasets and stores them using
*TimeSeriesDataset* classes.
"""
import math
from typing import Union, Tuple, List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from predpy.dataset import MultiTimeSeriesDataset, TimeSeriesDataset
import pandas as pd


class MultiTimeSeriesModule(LightningDataModule):
    """Lightning data module customized for time series.

    Parameters
    ----------
    LightningDataModule : [type]
        Lightning data module.
    """
    def __init__(
        self,
        sequences: List[pd.DataFrame],
        dataset_name: str,
        target: str,
        split_proportions: Union[Tuple[float], List[float]],
        window_size: int,
        batch_size: int = 8,
        DatasetCls: TimeSeriesDataset = MultiTimeSeriesDataset
    ):
        """Creates MultiTimeSeriesModule instance.

        Parameters
        ----------
        sequences : List[pd.DataFrame]
            Time series with same columns.
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

        # setting sequences
        assert len(sequences) > 0, "No time series passed"
        sequences = self._drop_too_short_seqs(sequences, window_size)

        # setting proportions
        self._proportions_assert(split_proportions)

        self.n_records = self._get_records_number(sequences, window_size)
        self._save_proportions_as_ids(split_proportions)

        self._ending_seqs_ids = self._get_ending_sequences_ids(
            sequences, window_size)

        self.split_proportions = split_proportions
        self.sequences = sequences
        self.name_ = dataset_name
        self.target = target
        self.window_size = window_size
        self.batch_size = batch_size
        self.DatasetCls = DatasetCls

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_records_number(
        self,
        sequences: List[pd.DataFrame],
        window_size: int
    ) -> int:
        n_records = sum([
            seq.shape[0] - window_size for seq in sequences
        ])
        return n_records

    def _get_ending_sequences_ids(
        self,
        sequences: List[pd.DataFrame],
        window_size: int
    ) -> List[int]:
        """Specifies records indices ending each time series dataframe.

        Parameters
        ----------
        sequences : List[pd.DataFrame]
            Time series with same columns.
        window_size : int
            Length of sequence in every sample.

        Returns
        -------
        List[int]
            List of starting records indices.
        """
        first_idx = 0
        ending_ids = []
        for seq in sequences:
            n_records = seq.shape[0] - window_size
            ending_ids += [first_idx + n_records - 1]
            first_idx += n_records
        return ending_ids

    def _drop_too_short_seqs(
        self,
        sequences: List[pd.DataFrame],
        window_size: int
    ) -> List[pd.DataFrame]:
        """Removes sequences shorter than window size. If no sequence is
        appropriate, raise assert error.

        Parameters
        ----------
        sequences : List[pd.DataFrame]
            Time series with same columns.
        window_size : int
            Length of sequence in every sample.

        Returns
        -------
        List[pd.DataFrame]
            Time series dataframes longer than window size.
        """
        good_seqs_ids = [
            i for i, seq in enumerate(sequences)
            if seq.shape[0] > window_size
        ]
        assert len(good_seqs_ids) > 0,\
            f"No sequence longer than window size {window_size}"

        sequences = [sequences[i] for i in good_seqs_ids]
        return sequences

    def _proportions_assert(
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

    def _save_proportions_as_ids(
        self,
        split_proportions: Union[Tuple[float], List[float]]
    ):
        """Saves ranges of training, validation and test datasets.

        Parameters
        ----------
        split_proportions : Union[Tuple[float], List[float]]
            3 values of proportions between datasets.
        n_records : int
            Number of records from all time series.
        window_size : int
            Length of sequence in every sample.
        """
        train_val_split = int(self.n_records * split_proportions[0])
        val_test_split =\
            int(self.n_records * (split_proportions[0] + split_proportions[1]))

        self.train_range = (0, train_val_split - 1)
        if len(split_proportions) == 3:
            self.val_range = \
                (train_val_split, val_test_split - 1)
            self.test_range = (val_test_split, self.n_records - 1)
        elif len(split_proportions) == 2:
            # validation and test datasets will be same
            self.val_range = \
                (train_val_split, self.n_records - 1)
            self.test_range = (train_val_split, self.n_records - 1)

    def _get_seqs_id_by_global_id(self, idx: int) -> int:
        """Returns a sequence index containing record with provided global
        index.

        Parameters
        ----------
        idx : int
            Global record index.

        Returns
        -------
        int
            Sequence index containing record with provided global index.
        """
        if idx > self._ending_seqs_ids[-1] or idx < 0:
            raise IndexError("Index out of range.")

        seqs_id = None
        for i, end_id in enumerate(self._ending_seqs_ids):
            if idx <= end_id:
                seqs_id = i
                break
        return seqs_id

    def _global_id_to_seq_rec_id(self, idx: int) -> Tuple[int, int]:
        """Transform global record index to sequence index and record index in
        that sequence.

        Parameters
        ----------
        idx : int
            Global record index.

        Returns
        -------
        Tuple[int, int]
            Sequence index and record relative index.
        """
        seq_id = self._get_seqs_id_by_global_id(idx)
        rec_id_in_seq = None
        if seq_id == 0:
            rec_id_in_seq = idx
        else:
            first_rec_id = self._ending_seqs_ids[seq_id - 1] + 1
            rec_id_in_seq = idx - first_rec_id
        return seq_id, rec_id_in_seq

    def _get_with_records_range(
        self,
        start: int,
        end: int
    ) -> List[pd.DataFrame]:
        """Treats sequences as they would already contain a records and returns
        their data based from provided records range.

        Parameters
        ----------
        start : int
            First record index.
        end : int
            Last record index.

        Returns
        -------
        List[pd.DataFrame]
            Sequences cut as they would already contain a records from provided
            range.
        """
        starting_seqs_id = self._get_seqs_id_by_global_id(start)
        ending_seqs_id = self._get_seqs_id_by_global_id(end)
        record_len = self.window_size + 1

        result = None
        # if sequences ids are the same
        if starting_seqs_id == ending_seqs_id:
            # return data only from it including ranges
            _, start = self._global_id_to_seq_rec_id(start)
            _, end = self._global_id_to_seq_rec_id(end)
            result = [
                self.sequences[starting_seqs_id].iloc[start:end+record_len]]
        # if sequences ids are different
        else:
            # collect data from starting sequence
            _, start = self._global_id_to_seq_rec_id(start)
            _, end = self._global_id_to_seq_rec_id(end)
            result = [self.sequences[starting_seqs_id].iloc[start:]]
            # append all sequences between starting and ending
            result += [
                self.sequences[i]
                for i in range(starting_seqs_id+1, ending_seqs_id)
            ]
            # append data from ending sequence
            result += [self.sequences[ending_seqs_id].iloc[:end+record_len]]

        return result

    def setup(self, stage: str = None):
        start, end = self.train_range
        seqs = self._get_with_records_range(start, end)
        self.train_dataset = self.DatasetCls(
            seqs, self.window_size, self.target)
        start, end = self.val_range
        seqs = self._get_with_records_range(start, end)
        self.val_dataset = self.DatasetCls(
            seqs, self.window_size, self.target)
        start, end = self.test_range
        seqs = self._get_with_records_range(start, end)
        self.test_dataset = self.DatasetCls(
            seqs, self.window_size, self.target)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
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

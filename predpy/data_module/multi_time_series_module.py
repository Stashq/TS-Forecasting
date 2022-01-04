"""Lightning data module customized for time series.\n

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! (coś o multi a nie single)

Split data between training, validation, test datasets and stores them using
*TimeSeriesDataset* classes.
"""
import math
import numpy as np
import pandas as pd
from string import Template
from typing import Union, Tuple, List
from pytorch_lightning import LightningDataModule
from predpy.dataset import MultiTimeSeriesDataloader

NOT_ENOUGH_RECORDS = Template("Not enough records. \
If $n_splits dataset types, dataset should\
contain at least $min_n_records records.")


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
        target: Union[str, List[str]],
        split_proportions: Union[Tuple[float], List[float]],
        window_size: int,
        batch_size: int = 8,
        overlapping: bool = False
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

        self.split_proportions = split_proportions
        self.sequences = sequences
        self.name_ = dataset_name
        if not isinstance(target, list):
            target = [target]
        self.target = target
        self.window_size = window_size
        self.batch_size = batch_size
        self.overlapping = overlapping

        # setting proportions
        self._proportions_assert(split_proportions)

        self.n_records, self.n_overlapping_records = self._get_records_number(
            sequences, window_size, split_proportions)

        self._save_proportions_as_ids(split_proportions, overlapping)

        self._ending_seqs_rec_ids = self._get_ending_sequences_rec_ids(
            sequences, window_size)

        self.train_seqs = None
        self.val_seqs = None
        self.test_seqs = None

    def copy(self):
        return MultiTimeSeriesModule(
            sequences=[
                seqs.copy(deep=True)
                for seqs in self.sequences
            ],
            dataset_name=self.name_,
            target=self.target[:],
            split_proportions=self.split_proportions[:],
            window_size=self.window_size,
            batch_size=self.batch_size,
            overlapping=self.overlapping
        )

    def _get_records_number(
        self,
        sequences: List[pd.DataFrame],
        window_size: int,
        split_proportions: Union[Tuple[float], List[float]]
    ) -> Tuple[int]:
        n_records = sum([
            seq.shape[0] - window_size for seq in sequences
        ])
        # removing overlapping records
        n_splits = len(split_proportions)
        n_overlapping_records =\
            (n_splits - 1) * (window_size - 1)
        if n_records < n_splits:
            min_n_records = n_splits + n_overlapping_records
            raise ValueError(NOT_ENOUGH_RECORDS.substitute(
                n_splits=n_splits, min_n_records=min_n_records))
        return n_records, n_overlapping_records

    def _get_ending_sequences_rec_ids(
        self,
        sequences: List[pd.DataFrame],
        window_size: int
    ) -> List[int]:
        """Specifies records global indices ending each time series dataframe.

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
        ending_rec_ids = []
        for seq in sequences:
            n_records = seq.shape[0] - window_size
            ending_rec_ids += [first_idx + n_records - 1]
            first_idx += n_records
        return ending_rec_ids

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
        split_proportions: Union[Tuple[float], List[float]],
        overlapping: bool = False
    ):
        """Saves ranges of training, validation and test datasets.

        If *overlapping* is False, prevent different datasets types from
        overlapping. This can cause leaving some data between dataset without
        dataset.
        Parameters
        ----------
        split_proportions : Union[Tuple[float], List[float]]
            3 values of proportions between datasets.
        n_records : int
            Number of records from all time series.
        window_size : int
            Length of sequence in every sample.
        """
        if overlapping:
            train_val_split = int(self.n_records * split_proportions[0])
            val_test_split = int(
                self.n_records * (split_proportions[0] + split_proportions[1]))

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
        else:
            # setting records range based on not overlapping n records
            n_records = self.n_records - self.n_overlapping_records
            overlap = self.window_size - 1

            val_test_split = int(
                n_records * (split_proportions[0] + split_proportions[1])
                + overlap)

            train_end = int(n_records * split_proportions[0])
            self.train_range = (0, train_end - 1)
            if len(split_proportions) == 3:
                val_start = train_end + overlap
                val_end = int(val_start + n_records * split_proportions[1])
                self.val_range = (val_start, val_end - 1)

                test_start = val_end + overlap
                test_end = int(test_start + n_records * split_proportions[2])
                self.test_range = (test_start, test_end)
            elif len(split_proportions) == 2:
                # validation and test datasets will be same
                val_test_start = train_end + overlap
                val_test_end = int(
                    val_test_start + n_records * split_proportions[1])
                self.val_range = (val_test_start, val_test_end - 1)
                self.test_range = (val_test_start, val_test_end - 1)

    def _get_seqs_id_by_global_rec_id(self, idx: int) -> int:
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
        if idx > self._ending_seqs_rec_ids[-1] or idx < 0:
            raise IndexError("Index out of range.")

        seqs_id = None
        for i, end_id in enumerate(self._ending_seqs_rec_ids):
            if idx <= end_id:
                seqs_id = i
                break
        return seqs_id

    def _global_rec_id_to_seq_rec_id(self, idx: int) -> Tuple[int, int]:
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
        seq_id = self._get_seqs_id_by_global_rec_id(idx)
        rec_id_in_seq = None
        if seq_id == 0:
            rec_id_in_seq = idx
        else:
            first_rec_id = self._ending_seqs_rec_ids[seq_id - 1] + 1
            rec_id_in_seq = idx - first_rec_id
        return seq_id, rec_id_in_seq

    def get_recs_range(
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
        starting_seqs_id = self._get_seqs_id_by_global_rec_id(start)
        ending_seqs_id = self._get_seqs_id_by_global_rec_id(end)
        record_len = self.window_size + 1

        result = None
        # if sequences ids are the same
        if starting_seqs_id == ending_seqs_id:
            # return data only from it including ranges
            _, start = self._global_rec_id_to_seq_rec_id(start)
            _, end = self._global_rec_id_to_seq_rec_id(end)
            result = [
                self.sequences[starting_seqs_id].iloc[start:end+record_len]]
        # if sequences ids are different
        else:
            # collect data from starting sequence
            _, start = self._global_rec_id_to_seq_rec_id(start)
            _, end = self._global_rec_id_to_seq_rec_id(end)
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
        self.train_seqs = self.get_recs_range(*self.train_range)
        self.val_seqs = self.get_recs_range(*self.val_range)
        self.test_seqs = self.get_recs_range(*self.test_range)

    def train_dataloader(self):
        return MultiTimeSeriesDataloader(
            self.train_seqs,
            window_size=self.window_size,
            target=self.target,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )

    def val_dataloader(self):
        return MultiTimeSeriesDataloader(
            self.val_seqs,
            window_size=self.window_size,
            target=self.target,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def test_dataloader(self):
        return MultiTimeSeriesDataloader(
            self.test_seqs,
            window_size=self.window_size,
            target=self.target,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

    def _global_data_id_to_seq_data_id(
        self,
        idx: int,
        ending_idx: bool = False
    ) -> Tuple[int, int]:

        def condition(idx, end):
            if ending_idx:
                return idx <= end
            else:
                return idx < end

        start = 0
        for i, seqs in enumerate(self.sequences):
            end = start + seqs.shape[0]
            if condition(idx, end):
                time_series_idx = i
                local_idx = idx - start
                return time_series_idx, local_idx
            start = end
        return None, None

    def _global_data_ranges_to_seq_data_ids(
        self,
        start: int = None,
        end: int = None
    ):
        max_end = sum([seqs.shape[0] for seqs in self.sequences])

        if start is None:
            start = 0
            # start_ts, start_local_idx = (0, 0)
        elif start < 0:
            start = max_end + start
            if start < 0:
                raise ValueError("Index out of range.")
        elif start > max_end - 1:
            raise ValueError(
                f"Start index cannot be greater than {max_end - 1}.")

        if end is None or end > max_end:
            end = max_end
        elif end < 0:
            end = max_end + end
            if end < 0:
                raise ValueError("Index out of range.")

        start_ts, start_local_idx =\
            self._global_data_id_to_seq_data_id(start)
        end_ts, end_local_idx =\
            self._global_data_id_to_seq_data_id(end, ending_idx=True)

        if start > end:
            raise ValueError("Start index cannot be greater than end index.")

        return (start_ts, start_local_idx, end_ts, end_local_idx)

    def get_data_from_range(
        self,
        start: int = None,
        end: int = None,
        copy: bool = True
    ) -> List[pd.DataFrame]:
        start_ts, start_local_idx, end_ts, end_local_idx =\
            self._global_data_ranges_to_seq_data_ids(start, end)

        result = []
        if start_ts < end_ts:
            result += [self.sequences[start_ts].iloc[start_local_idx:]]
            for ts_idx in range(start_ts + 1, end_ts):
                result += [self.sequences[ts_idx]]
            result += [self.sequences[end_ts].iloc[:end_local_idx]]
        elif start_ts == end_ts:
            result += [
                self.sequences[start_ts].iloc[start_local_idx:end_local_idx]]
        else:
            raise ValueError("Starting time series index is \
                greater than ending time series.")

        if copy:
            result = [seqs.copy() for seqs in result]
        return result

    def target_cols_ids(self) -> List[int]:
        return [
            self.sequences[0].columns.get_loc(t)
            for t in self.target
        ]

    def global_ids_to_data(self, global_ids: List[int]):
        seq_ids, rec_ids = list(zip(*[
            self._global_id_to_seq_rec_id(idx)
            for idx in global_ids
        ]))

        res = None
        for seq_idx in set(seq_ids):
            filter_ = np.argwhere(np.array(seq_ids) == seq_idx)
            res += [self._rec_ids_to_data(rec_ids[filter_])]

        return res

    def _rec_ids_to_data(self, rec_ids: List[int], seq_id: int):
        rec_ids = rec_ids.sort()
        data_ids = []
        s_start = rec_ids[0]
        s_end = rec_ids[0] + self.window_size
        for rec_idx in rec_ids[1:]:
            if s_end < rec_idx:
                data_ids += list(range(s_start, s_end))
                s_start = rec_idx
            s_end = rec_idx + self.window_size
        data_ids += list(range(s_start, s_end))

        return self.sequences[seq_id].iloc[data_ids][self.target]

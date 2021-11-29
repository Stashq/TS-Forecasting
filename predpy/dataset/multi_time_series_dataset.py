"""Module contains *MultiTimeSeriesDataset* - custom pytorch Dataset class.\n

Samples shares same memory, so it is strongly advised not to change them during
usage. Created object is much lighter than *TimeSeriesRecodsDataset* object.
"""
import torch
import pandas as pd
from .time_series_dataset import TimeSeriesDataset
from typing import Dict, List, Tuple


class MultiTimeSeriesDataset(TimeSeriesDataset):
    """Custom Pytorch Dataset class for single time series.\n

    Samples shares same memory, so it is strongly advised not to change them
    during usage. Created object is much lighter than *TimeSeriesRecodsDataset*
    object.

    X are sequences created with moving window method from passed
    time series, y are target values following after them.

    Parameters
    ----------
    BaseTimeSeriesDataset : [type]
        Abstract class for time series datasets classes.
    """
    def __init__(
        self,
        sequences: List[pd.DataFrame],
        window_size: int,
        target: str
    ):
        """Creates *MultiTimeSeriesDataset* instance.

        Parameters
        ----------
        sequences : pd.DataFrame
            Time series with same columns.
        window_size : int
            Window size, defines how long should be one sample.
        target : str
            Name of column containing values to be predicted.
        """
        sequences = self._drop_too_short_seqs(sequences, window_size)

        self.window_size = window_size
        self.target = target
        self.sequences = sequences

        self._ending_seqs_ids = self._get_ending_sequences_ids(
            sequences, window_size)

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

    def _get_record(self, idx: int) -> pd.Series:
        """Transform global record index to sequence index and record index in
        that sequence, then returns record.

        Parameters
        ----------
        idx : int
            Global record index.

        Returns
        -------
        pd.Series
            Selected record.
        """
        seq_id, rec_id_in_seq = self._global_id_to_seq_rec_id(idx)
        seq = self.sequences[seq_id]\
            .iloc[rec_id_in_seq:rec_id_in_seq + self.window_size]
        label = self.sequences[seq_id]\
            .iloc[rec_id_in_seq + self.window_size][self.target]
        return seq, label

    def __len__(self) -> int:
        """Returns number of samples in dataset.\n

        Due to length of single sample number of records is length of
        sequence minus window size.

        Returns
        -------
        int
            Length of dataset.
        """
        n_records = 0
        for seq in self.sequences:
            n_records += seq.shape[0] - self.window_size
        return n_records

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
        seq, label = self._get_record(idx)
        return dict(
            sequence=torch.tensor(seq.to_numpy().T).float(),
            label=torch.tensor(label).float()
        )

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
            end_idx = self._ending_seqs_ids[-1]
        seqs = self._get_with_records_range(start_idx, end_idx)
        labels = pd.concat([
            seq[self.target] for seq in seqs])
        return labels

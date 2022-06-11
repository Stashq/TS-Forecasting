"""Module contains *MultiTimeSeriesDataset* - custom pytorch Dataset class.\n

Samples shares same memory, so it is strongly advised not to change them during
usage. Created object is much lighter than *TimeSeriesRecodsDataset* object.
"""
import torch
import numpy as np
import pandas as pd
from .time_series_dataset import TimeSeriesDataset
from typing import Dict, List, Literal, Tuple, Union
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import scipy


default_wdd_w_f = scipy.stats.norm(loc=0, scale=1).pdf(-1.5)
default_wdd_ma_f = scipy.stats.norm(loc=0, scale=1).pdf(-2)


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
        target: Union[str, List[str]] = None,
        for_reconstruction: bool = True
    ):
        """Creates *MultiTimeSeriesDataset* instance.
        If its use for reconstruction, last element of last sequence
        is duplicated to mantain number of samples.

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
        if not isinstance(target, list):
            target = [target]
        elif target is None:
            target = sequences[0].columns.tolist()
        self.target = target
        self.sequences = sequences
        self.n_points = sum([seq.shape[0] for seq in sequences])
        self.for_reconstruction = for_reconstruction
        # if for_reconstruction:
        #     self._repeat_last_element()

        self._ending_seqs_ids = self._get_ending_sequences_ids(
            sequences, window_size)

    # def _repeat_last_element(self):
    #     try:
    #         index = self.sequences[-1].index
    #         step = index[-1] - index[-2]
    #         new_point_id = index[-1] + step
    #     except TypeError:
    #         new_point_id = 'new_point_id'
    #     last_point = self.sequences[-1].iloc[-1:]
    #     last_point.index = [new_point_id]
    #     self.sequences[-1] = pd.concat(
    #         [self.sequences[-1], last_point])

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
            f"No sequence longer than window size {window_size}."

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
        if idx < 0:
            raise IndexError("Index out of range.")
        elif self.for_reconstruction\
                and (idx == self._ending_seqs_ids[-1] + 1):
            idx -= 1
        elif idx > self._ending_seqs_ids[-1]:
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

    def get_record(self, idx: int) -> pd.Series:
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

        label_idx = rec_id_in_seq + self.window_size
        if self.for_reconstruction:
            label_idx -= 1
        label = self.sequences[seq_id]\
            .iloc[label_idx][self.target]
        return seq, label

    def set_record(self, idx: int, target: str, vals):
        seq_id, rec_id_in_seq = self._global_id_to_seq_rec_id(idx)
        col_idx = self.sequences[0].columns.get_loc(target)
        self.sequences[seq_id].iloc[
            rec_id_in_seq:rec_id_in_seq + self.window_size, col_idx] =\
            vals

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
        if self.for_reconstruction:
            n_records += 1
        return n_records

    def __getitem__(self, idx: int) -> Dict[torch.Tensor, torch.Tensor]:
        """Returns dict of:
        * sequence - sequence starting from indicated position
        with shape (N, L, H_in) where:
            - N = batch size
            - L = sequence length
            - H_in = input size
        * label - target value/-s followed after sequence.

        Parameters
        ----------
        idx : [int]
            Position in primary sequence of staring the sample.

        Returns
        -------
        Dict[torch.Tensor, torch.Tensor]
            Dict containing sequence and label.
        """
        seq, label = self.get_record(idx)
        return dict(
            sequence=torch.tensor(seq.to_numpy()).float(),
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
            end_idx = self._ending_seqs_ids[-1]
        seqs = self._get_with_records_range(start_idx, end_idx)
        label_start_idx = self.window_size
        if self.for_reconstruction:
            label_start_idx -= 1
        labels = pd.concat([
            seq[self.target].iloc[label_start_idx:] for seq in seqs])
        return labels

    def get_indices_like_recs(
        self, labels: bool = True
    ) -> Union[pd.Series, List[pd.Series]]:
        """Return indices of data as they would be records
        """
        indices = []
        if labels:
            indices = self.get_labels().index.to_series()
        else:
            indices = []
            for seq in self.sequences:
                indices += list(
                    seq.index.to_series()
                    .rolling(self.window_size)
                )[self.window_size-1:-1]
        return indices

    def copy(self):
        return MultiTimeSeriesDataset(
            sequences=[
                seqs.copy(deep=True)
                for seqs in self.sequences
            ],
            window_size=self.window_size,
            target=self.target[:]
        )

    def global_ids_to_data(self, global_ids: List[int]):
        if len(global_ids) == 0:
            return pd.DataFrame(columns=self.target)

        seq_ids, rec_ids = list(zip(*[
            self._global_id_to_seq_rec_id(idx)
            for idx in global_ids
        ]))

        res = []
        for seq_idx in set(seq_ids):
            filter_ = np.argwhere(np.array(seq_ids) == seq_idx).T[0].tolist()
            res += [self._rec_ids_to_data(
                np.array(rec_ids)[filter_], seq_idx)]
        res = pd.concat(res)
        return res

    def _rec_ids_to_data(self, rec_ids: List[int], seq_id: int):
        rec_ids.sort()
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

    def get_data_ids_by_rec_ids(self, rec_ids: List[int]) -> List:
        ids = set()
        for rec_id in rec_ids:
            seq, _ = self.get_record(rec_id)
            ids.update(seq.index.tolist())
        return list(ids)

    def get_recs_cls_by_data_cls(
        self, data_cls: List, min_points: int = 1
    ) -> List[int]:
        col_name = 'class'
        self._add_column(col_name, data_cls)
        res = []
        for i in tqdm(range(self.__len__()), 'Collecting record classes'):
            res += [self._get_rec_class(
                idx=i, col_name=col_name, min_points=min_points)]
        self._remove_column(col_name)
        return res

    def _get_rec_class(
        self, idx, col_name, min_points: int = 1
    ) -> Literal[0, 1]:
        seq, _ = self.get_record(idx)
        n_cls_points = seq[seq[col_name] == 1].shape[0]
        if n_cls_points < min_points:
            return 0
        else:
            return 1

    def calculate_rec_wdd(
        self, pred_rec_cls: List[int],
        true_rec_cls: List[int], t_max: int,
        w_f: float = default_wdd_w_f,
        ma_f: float = default_wdd_ma_f
    ) -> float:
        """Calculate WDD score from article
        'Evaluation metrics for anomaly detection algorithms in time-series'
        based on gaussian distribution function.
        Compares distance between records in time series.

        Args:
            pred_rec_cls (List[int]): predicted records classes.
            true_rec_cls (List[int]): true records classes.
            t_max (int): maximum distance between paired
                predicted and true anomaly position.
            w_f (float): false anomaly detenction penalty.
            ma_f (float, optional): missed anomaly penalty. Defaults to 0.

        Returns:
            float: wdd score.
        """
        ids = self.get_labels().index
        cls_df = pd.DataFrame(zip(
            true_rec_cls, pred_rec_cls
        ), index=ids, columns=['true_cls', 'pred_cls'])
        nd = scipy.stats.norm(loc=0, scale=1)

        def score(row):
            # calculate w
            if row['true_cls'] == 1:
                idx = row.name
                frame = cls_df.loc[idx-t_max:idx+t_max]
                preds = frame[frame['pred_cls'] == 1]
                # missed anomaly penalty
                if preds.shape[0] == 0:
                    return -ma_f
                else:
                    diff = abs(preds.index - idx).min()
                    return nd.pdf(diff/t_max)
            # find FA
            elif row['pred_cls'] == 1:
                idx = row.name
                frame = cls_df.loc[idx-t_max:idx+t_max]
                preds = frame[frame['true_cls'] == 1]
                # detected anomaly (DA)
                if preds.shape[0] > 0:
                    return 0
                # false anomaly penalty
                else:
                    return -w_f
            else:
                return 0
        scores = cls_df.apply(score, axis=1)
        wdd = scores.sum()

        return wdd

    def calculate_point_wdd(
        self, pred_rec_cls: List[int],
        true_rec_cls: List[int], t_max: int,
        w_f: float = default_wdd_w_f,
        ma_f: float = default_wdd_ma_f
    ) -> float:
        """Calculate WDD score from article
        'Evaluation metrics for anomaly detection algorithms in time-series'
        based on gaussian distribution function.
        Compares distance between points in time series.

        Args:
            pred_rec_cls (List[int]): predicted records classes.
            true_rec_cls (List[int]): true records classes.
            t_max (int): maximum distance between paired
                predicted and true anomaly position.
            w_f (float): false anomaly detenction penalty.
            ma_f (float, optional): missed anomaly penalty. Defaults to 0.

        Returns:
            float: wdd score.
        """
        self._add_column('true_cls', 0)
        self._add_column('pred_cls', 0)

        positive_ids = np.argwhere(
            (np.array(pred_rec_cls) == 1) | (np.array(true_rec_cls) == 1)
        ).flatten().tolist()
        for idx in tqdm(positive_ids, 'Collecting data classes'):
            self.set_record(idx, 'true_cls', true_rec_cls[idx])
            self.set_record(idx, 'pred_cls', pred_rec_cls[idx])

        seqs = pd.concat(self.sequences)
        nd = scipy.stats.norm(loc=0, scale=1)

        def score(row):
            # calculate w
            if row['true_cls'] == 1:
                idx = row.name
                frame = seqs.loc[idx-t_max:idx+t_max]
                preds = frame[frame['pred_cls'] == 1]
                # missed anomaly penalty
                if preds.shape[0] == 0:
                    return -ma_f
                else:
                    diff = abs(preds.index - idx).min()
                    return nd.pdf(diff/t_max)
            # find FA
            elif row['pred_cls'] == 1:
                idx = row.name
                frame = seqs.loc[idx-t_max:idx+t_max]
                preds = frame[frame['true_cls'] == 1]
                # detected anomaly (DA)
                if preds.shape[0] > 0:
                    return 0
                # false anomaly penalty
                else:
                    return -w_f
            else:
                return 0
        scores = seqs.apply(score, axis=1)
        wdd = scores.sum()

        self._remove_column('true_cls')
        self._remove_column('pred_cls')
        return wdd

    def _add_column(
        self, new_col_name, values: Union[List[int], int] = 0,
        include_col_in_target: bool = False
    ):
        len_ = sum([seq.shape[0] for seq in self.sequences])
        if type(values) is not int and len_ != len(values):
            raise ValueError(
                'Data len %d different than values len %d.'
                % (len_, len(values)))

        for seq in self.sequences:
            seq_len = seq.shape[0]
            if type(values) == int:
                seq[new_col_name] = values
            else:
                seq[new_col_name] = values[:seq_len]
                values = values[seq_len:]

        if include_col_in_target:
            self.target = self.target + [new_col_name]

    def _remove_column(self, col_name):
        for seq in self.sequences:
            seq.drop([col_name], axis=1, inplace=True)

        if col_name in self.target:
            self.target.remove(col_name)


class MultiTimeSeriesDataloader(DataLoader):
    def __init__(
        self,
        sequences: List[pd.DataFrame],
        window_size: int,
        target: Union[str, List[str]],
        *args, **kwargs
    ):
        super().__init__(
            MultiTimeSeriesDataset(
                sequences=sequences, window_size=window_size, target=target
            ), *args, **kwargs)

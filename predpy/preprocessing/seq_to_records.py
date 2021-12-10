"""Module provides function *seq_to_records* converting time series to records.

Every record is a tuple containing a sequence (model input data) and a single
target value following after sequence (predicted value).
"""
from typing import Tuple, List, Union
import pandas as pd
from tqdm.notebook import tqdm

Record = Tuple[pd.DataFrame, pd.Series]


def _get_seq_record(
    time_series: pd.DataFrame,
    window_size: int,
    target: Union[List[str], str],
    idx: int
) -> Record:
    """Create single record from time series starting from provided position.

    Parameters
    ----------
    time_series : pd.DataFrame
        Input time series.
    window_size : int
        Length of sequence in record.
    target : str
        Input column name containing values to predict.
    idx : int
        Index of sequence starting position in input time series.

    Returns
    -------
    Tuple[pd.DataFrame, float]
        Record containing sequence and target value following after it.
    """
    label_position = idx + window_size
    if not isinstance(target, list):
        target = [target]

    sequence = time_series[idx:label_position]
    label = time_series.iloc[label_position][target]
    return sequence, label


def seq_to_records(
    input_data: pd.DataFrame,
    window_size: int,
    target: Union[List[str], str]
) -> List[Record]:
    """Creates list of records.\n\n

    Record is tuple with sequence dataframe and single target value following
    after it.\n
    Function splits time series to records using moving window method.

    Parameters
    ----------
    input_data : pd.DataFrame
        Input time series.
    window_size : int
        Length of sequences in records.
    target : str
        Input column name containing values to predict.

    Returns
    -------
    List[Tuple[pd.DataFrame, float]]
        List of records containing sequences and target values following after
        those.
    """
    records = []
    data_size = len(input_data)

    for i in tqdm(range(data_size - window_size)):
        records.append(
            _get_seq_record(input_data, window_size, target, i))

    return records

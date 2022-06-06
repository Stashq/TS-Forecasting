import os
import csv
import re
from sklearn.base import TransformerMixin
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from predpy.dataset import MultiTimeSeriesDataset


def get_dataset(
    path: Path, window_size: int, ts_scaler: TransformerMixin = None,
    read_csv_kwargs: Dict = {'header': None}
) -> MultiTimeSeriesDataset:
    df = pd.read_csv(
        path, **read_csv_kwargs
    )
    try:
        df.columns = df.columns.astype(int)
    except TypeError:
        pass
    if ts_scaler is not None:
        df[:] = ts_scaler.transform(df)
    dataset = MultiTimeSeriesDataset(
        sequences=[df],
        window_size=window_size,
        target=df.columns.tolist()
    )
    return dataset


def get_dataset_names(path: str):
    """Dataset path should follow pattern:
    .*/data/{topic}/{collection}/{"train", "test" or "test_labels"/{dataset}"""
    dir_names = path.split(os.sep)
    start_id = dir_names.index('data')
    topic = dir_names[start_id + 1]
    collection_name = dir_names[start_id + 2]
    dataset_name = dir_names[start_id + 4][:-4]
    return topic, collection_name, dataset_name


def _str_to_float_list(text: str) -> List[float]:
    floats = re.findall(r'\d+.\d+', text)
    res = []
    for f in floats:
        res += [float(f)]
    return res


def load_anom_scores(
    path: Path
) -> Tuple[List[float], List[int]]:
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        scores = []
        classes = []
        for row in reader:
            scores += [_str_to_float_list(row['score'])]
            classes += [int(row['class'])]
    return scores, classes
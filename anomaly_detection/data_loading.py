import os
from sklearn.base import TransformerMixin
import pandas as pd
from typing import Dict
from pathlib import Path

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

"""Dataclasses representing fields in Experimentator class from
:py:mod:`experimentator` module.

Module describes representation parameters of datasets, models and resutls of
single experiment step (predictions made with one model on one dataset test
part).
"""
from dataclasses import dataclass, field
from predpy.dataset import TimeSeriesDataset
from typing import List, Dict, Tuple, Callable, Any
import torch


@dataclass
class DatasetParams:
    '''Data class representing dataset init arguments.

    If passed to Experimentator, *true_values* and *name* will be set during
    run_experiments.
    '''
    path: str
    target: str
    split_proportions: List[float]
    window_size: int
    batch_size: int
    DatasetCls: TimeSeriesDataset
    pipeline: List[Tuple[Callable, Dict]]
    load_params: Dict[str, Any] = field(default_factory=dict)
    true_values: List[float] = None
    name_: str = None


@dataclass
class ModelParams:
    '''Data class representing pytorch model details: name, class definition
    and init parameters.
    '''
    name_: str
    cls_: torch.nn.Module
    init_params: Dict


@dataclass
class PredictionRecord:
    '''Data class representing single experiment record.

    Contains experiment dataset and model indices, also predictions of model.
    '''
    dataset_id: int
    model_id: int
    predictions: List[float]
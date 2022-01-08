"""Dataclasses representing fields in Experimentator class from
:py:mod:`experimentator` module.

Module describes representation parameters of datasets, models and resutls of
single experiment step (predictions made with one model on one dataset test
part).
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Any, Type
from sklearn.base import TransformerMixin
import pandas as pd
import torch
from torch import nn, optim

# from predpy.dataset import TimeSeriesDataset
from predpy.wrapper import ModelWrapper, Predictor
# from predpy.dataset import MultiTimeSeriesDataset


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
    drop_refill_pipeline: List[Tuple[Callable, Dict]] =\
        field(default_factory=list)
    preprocessing_pipeline: List[Tuple[Callable, Dict]] =\
        field(default_factory=list)
    detect_anomalies_pipeline: List[Tuple[Callable, Dict]] =\
        field(default_factory=list)
    load_params: Dict[str, Any] = field(default_factory=dict)
    scaler: TransformerMixin = None
    true_values: List[float] = None
    name_: str = None

    # def __post_init__(self):
    #     if self.DatasetCls is None:
    #         self.DatasetCls = MultiTimeSeriesDataset


@dataclass
class LearningParams:
    lr: float = 1e-4
    criterion: nn.Module = nn.MSELoss()
    OptimizerClass: optim.Optimizer = optim.Adam
    optimizer_kwargs: Dict = field(default_factory=dict)


@dataclass
class ModelParams:
    '''Data class representing pytorch model details: name, class definition
    and init parameters.
    '''
    name_: str
    cls_: torch.nn.Module
    init_params: Dict
    WrapperCls: Type[ModelWrapper] = None
    wrapper_kwargs: Dict[str, Any] = field(default_factory=dict)
    learning_params: LearningParams = field(default_factory=LearningParams)

    def __post_init__(self):
        if self.WrapperCls is None:
            self.WrapperCls = Predictor


@dataclass
class PredictionRecord:
    '''Data class representing single experiment record.

    Contains experiment dataset and model indices, also predictions of model.
    '''
    dataset_id: int
    model_id: int
    predictions: pd.DataFrame

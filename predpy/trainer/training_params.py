"""Dataclasses representing pytorch lightning modules arguments.

Contains init parameters for lightning trainer, early stopping and checkpoint
callbacks, logger and learning parameters for lightning model optimizer.
"""
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List, Iterable
from torch import nn, optim

# from pytorch_lightning.accelerators import Accelerator
# from pytorch_lightning.plugins.training_type import TrainingTypePlugin
# from pytorch_lightning.plugins.precision import PrecisionPlugin
# from pytorch_lightning.plugins.environments import ClusterEnvironment
# from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.callbacks import Callback
from datetime import timedelta
from pathlib import Path


@dataclass
class TrainerParams:
    logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool] =\
        True
    callbacks: Union[List[Callback], Callback, None] = None
    overfit_batches: Union[int, float] = 0.0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Union[List[int], str, int, None] = None
    max_epochs: Optional[int] = None
    min_epochs: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Union[str, timedelta, Dict[str, int], None] = None
    flush_logs_every_n_steps: Optional[int] = 50
    log_every_n_steps: int = 50
    auto_lr_find: Union[bool, str] = False


@dataclass
class CheckpointParams:
    dirpath: Union[str, Path, None] = None
    filename: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[timedelta] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None


@dataclass
class EarlyStoppingParams:
    monitor: str = 'val_loss'
    min_delta: float = 0.0
    patience: int = 3
    verbose: bool = False
    mode: str = "min"
    strict: bool = True
    check_finite: bool = True
    stopping_threshold: Optional[float] = None
    divergence_threshold: Optional[float] = None
    check_on_train_epoch_end: Optional[bool] = None


@dataclass
class LoggerParams:
    save_dir: str = "./"
    name: str = 'default'
    version: str = None


@dataclass
class LearningParams:
    lr: float = 1e-4
    criterion: nn.Module = nn.MSELoss()
    OptimizerClass: optim.Optimizer = optim.Adam
    optimizer_kwargs: Dict = field(default_factory=dict)

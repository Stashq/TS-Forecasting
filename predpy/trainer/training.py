"""Module provide functions for training lightning module purpose.

*get_trainer* creates pytchorch lightning trainer,
*get_train_pl_model* creates pythorch lightning module wrapping
passed pytorch model and trains it.
"""
from typing import Dict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LightningLoggerBase
import torch
from predpy.wrapper import Predictor
from dataclasses import asdict
from .training_params import (
    CheckpointParams, TrainerParams, LoggerParams, EarlyStoppingParams,
    LearningParams)


def get_trainer(
    trainer_params: TrainerParams,
    logger_params: LoggerParams = None,
    checkpoint_params: CheckpointParams = None,
    early_stopping_params: EarlyStoppingParams = None,
    LoggerCls: LightningLoggerBase = TensorBoardLogger
) -> pl.Trainer:
    """Create trainer object.\n

    If *trainer_params* contain not empty:
    * logger and *logger_params* are passed, logger will be overwritten,
    * callbacks and *checkpoint_params* or *early_stopping_params* are passed,
    callbacks, checkpoint and early stopping will be concatenated.

    Parameters
    ----------
    trainer_params : TrainerParams
        Lightning trainer parameters (can contain logger and additional
        callbacks).
    logger_params : LoggerParams, optional
        Chosen lightning logger parameters, by default None.
    checkpoint_params : CheckpointParams, optional
        Lightning *ModelCheckpoint* init arguments, by default None.
    early_stopping_params : EarlyStoppingParams, optional
        Lightning *EarlyStopping* init arguments, by default None.
    LoggerCls : LightningLoggerBase, optional
        Lightning logger class, by default TensorBoardLogger.

    Returns
    -------
    pl.Trainer
        Trainer with the set parameters.
    """
    if logger_params is not None:
        trainer_params.logger = LoggerCls(**asdict(logger_params))

    # Setting callbacks
    callbacks = []
    if checkpoint_params is not None:
        callbacks += [
            ModelCheckpoint(**asdict(checkpoint_params))]
    if early_stopping_params is not None:
        callbacks += [
            EarlyStopping(**asdict(early_stopping_params))]

    # Concatenating callbacks
    if trainer_params.callbacks is None:
        trainer_params.callbacks = callbacks
    else:
        trainer_params.callbacks += callbacks

    return pl.Trainer(**asdict(trainer_params))


def get_trained_pl_model(
    model: torch.nn.Module,
    data_module: pl.LightningDataModule,
    trainer_params: TrainerParams,
    logger_params: LoggerParams = None,
    checkpoint_params: CheckpointParams = None,
    early_stopping_params: EarlyStoppingParams = None,
    learning_params: LearningParams = LearningParams(),
    WrapperCls: pl.LightningModule = Predictor,
    wrapper_kwargs: Dict = {},
    LoggerCls: LightningLoggerBase = TensorBoardLogger
) -> pl.LightningModule:
    """Creates and train pytorch lightning model.

    Wraps *model* into pytorch lightning module defined with *WrapperCls* and
    pass learning and additional arguments to it.
    Creates pythorch lightning trainer instance with provided parameters,
    train module and return it.

    If *trainer_params* contain not empty:
    * logger and *logger_params* are also passed, logger will be overwritten,
    * callbacks and *checkpoint_params* or *early_stopping_params* are also
    passed, callbacks, checkpoint and early stopping will be concatenated.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch time series predicting model.
    data_module : pl.LightningDataModule
        Lightning data module containing time series samples.
    trainer_params : TrainerParams
        Lightning trainer parameters (can contain logger and additional
        callbacks).
    logger_params : LoggerParams, optional
        Chosen lightning logger parameters, by default None.
    checkpoint_params : CheckpointParams, optional
        Lightning *ModelCheckpoint* init arguments, by default None.
    early_stopping_params : EarlyStoppingParams, optional
        Lightning *EarlyStopping* init arguments, by default None.
        [description], by default None
    learning_params : LearningParams, optional
        Learning parameters. If not provided, argument will be replaced
        with parameters default LearningParams dataclass.
    WrapperCls : pl.LightningModule, optional
        Lightning class wrapping pytorch model. By default Predictor.
    wrapper_kwargs : Dict, optional
        Additional wrapper named arguments. By default {}.
    LoggerCls : LightningLoggerBase, optional
        Lightning logger class, by default TensorBoardLogger.

    Returns
    -------
    pl.LightningModule
        Trained lightning model.
    """

    pl_model = WrapperCls(model=model, **asdict(learning_params),
                          **wrapper_kwargs)

    trainer = get_trainer(trainer_params, logger_params, checkpoint_params,
                          early_stopping_params, LoggerCls)

    trainer.fit(pl_model, data_module)

    return pl_model.eval()

"""Module provide functions for training lightning module purpose.

*get_trainer* creates pytchorch lightning trainer,
*get_train_pl_model* creates pythorch lightning module wrapping
passed pytorch model and trains it.
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataclasses import asdict
from .training_params import (
    CheckpointParams, TrainerParams, EarlyStoppingParams)


def get_trainer(
    trainer_params: TrainerParams,
    checkpoint_params: CheckpointParams = None,
    early_stopping_params: EarlyStoppingParams = None
) -> pl.Trainer:
    """Create trainer object.\n

    If *trainer_params* contain not empty callbacks
    and *checkpoint_params* or *early_stopping_params* are passed,
    callbacks, checkpoint and early stopping will be concatenated.

    Parameters
    ----------
    trainer_params : TrainerParams
        Lightning trainer parameters (can contain logger and additional
        callbacks).
    checkpoint_params : CheckpointParams, optional
        Lightning *ModelCheckpoint* init arguments, by default None.
    early_stopping_params : EarlyStoppingParams, optional
        Lightning *EarlyStopping* init arguments, by default None.

    Returns
    -------
    pl.Trainer
        Trainer with the set parameters.
    """
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
    pl_model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer_params: TrainerParams,
    checkpoint_params: CheckpointParams = None,
    early_stopping_params: EarlyStoppingParams = None
) -> pl.LightningModule:
    """Trains pytorch lightning model.

    If *trainer_params* contain not empty callbacks
    and *checkpoint_params* or *early_stopping_params* are also passed,
    callbacks, checkpoint and early stopping will be concatenated.

    Parameters
    ----------
    pl_model : pl.LightningModule
        Pytorch lightning time series predicting module.
    data_module : pl.LightningDataModule
        Lightning data module containing time series samples.
    trainer_params : TrainerParams
        Lightning trainer parameters (can contain logger and additional
        callbacks).
    checkpoint_params : CheckpointParams, optional
        Lightning *ModelCheckpoint* init arguments, by default None.
    early_stopping_params : EarlyStoppingParams, optional
        Lightning *EarlyStopping* init arguments, by default None.
        [description], by default None

    Returns
    -------
    pl.LightningModule
        Trained lightning model.
    """
    trainer = get_trainer(
        trainer_params, checkpoint_params, early_stopping_params)

    trainer.fit(pl_model, data_module)

    return pl_model.eval()

"""Module for automation experiments, versioning and reproducting them,
also getting desirable parts of them.

In order to make possible to reconstruct preprocessing pipeline,
it is highly recomended to use preprocessing functions defined
in :py:mod:`preprocessing` module or any library function compatible
to the pipeline (guidelines defined in :py:mod:`preprocessing` module).
If you use custom functions, be sure you saved them in safe place
that they won't be changed or lost and pipelines using them will be possible
to reconstruct.


!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
from __future__ import annotations
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Type, Tuple, Union
import time
from datetime import timedelta
import pathlib
import pickle
import pandas as pd
from sklearn.base import TransformerMixin
from string import Template
import os
from dataclasses import asdict
import torch

from predpy.wrapper import Autoencoder, VAE, TSModelWrapper
from predpy.data_module import MultiTimeSeriesModule
from predpy.preprocessing import load_and_preprocess
from predpy.trainer import get_trained_pl_model
from predpy.trainer import (
    TrainerParams, LoggerParams, EarlyStoppingParams, CheckpointParams)

from .experimentator_params import (
    DatasetParams, ModelParams, PredictionRecord)

dataset_setup_exception_msg = Template(
    "Error during setup $dataset_idx dataset named $dataset_name.")
training_exception_msg = Template(
    "Problem with training $model_idx model named $model_name on "
    "$dataset_idx dataset named $dataset_name")


class Experimentator:
    """Class for automation experiments, versioning and reproducting them,
    also getting desirable parts of them.

    Automates:
    * data loading, preprocessing and creating new features with
    :py:mod:`preprocessing` module,
    * creating datasets for time series with :py:mod:`dataset` module,
    * splitting datasets for training, validation and test taking into account
    the sequential nature of the data
    * training process with :py:mod:`trainer` module,
    * creating model checkpoints, early stopping and logging with lightning
    callbacks,
    * collecting predictions and plotting them with plotly,
    * experiments versioning,
    * experiments reproduction,
    * loading and selecting experiments setups and pipelines.
    """
    def __init__(
        self,
        models_params: List[ModelParams],
        datasets_params: List[DatasetParams],
        trainer_params: TrainerParams = TrainerParams(),
        checkpoint_params: CheckpointParams = CheckpointParams(),
        early_stopping_params: EarlyStoppingParams = EarlyStoppingParams(),
        loggers_params: List[LoggerParams] = [LoggerParams()],
        LoggersClasses: List[Type[LightningLoggerBase]] = [TensorBoardLogger]
    ):
        """Creates experimentator instance and sets experiments parameters.

        Parameters
        ----------
        models_params : List[ModelParams]
            Models names, classes and init_parameters.
        datasets_params : List[DatasetParams]
            Parameters to create time series dataset.
        learning_params : LearningParams, optional
            Learning parameters, by default LearningParams().
        WrapperCls : Type[pl.LightningDataModule], optional
            Lightning class wrapping pytorch model. By default Predictor.
        wrapper_kwargs : Dict, optional
            Additional wrapper named arguments. By default {}.
        trainer_params : TrainerParams, optional
            Lightning trainer parameters (can contain logger and additional
            callbacks), by default TrainerParams().
        checkpoint_params : CheckpointParams, optional
            Lightning *ModelCheckpoint* init arguments,
            by default CheckpointParams().
        early_stopping_params : EarlyStoppingParams, optional
            Lightning *EarlyStopping* init arguments,
            by default EarlyStoppingParams().
        """
        self.set_experiment_params(models_params, datasets_params)

        # Trainer params
        self.trainer_params = trainer_params
        self.checkpoint_params = checkpoint_params
        self.early_stopping_params = early_stopping_params
        self.loggers_params = loggers_params
        self.LoggersClasses = LoggersClasses
        self.last_step_end = (-1, -1)

        # Experiment variables init
        self.predictions = None
        self.exp_date = None

    def set_experiment_params(
        self,
        models_params: List[ModelParams],
        datasets_params: List[DatasetParams]
    ):
        """Set models and datasets parameters.

        Store them as dataframes. Set datasets names same names of files
        they come from.

        Parameters
        ----------
        models_params : List[ModelParams]
            Models names, classes and init_parameters.
        datasets_params : List[DatasetParams]
            Parameters to create time series dataset.
        """
        self.models_params = pd.DataFrame(models_params)
        self.datasets_params = pd.DataFrame(datasets_params)
        self.datasets_params["name_"] = self.datasets_params.apply(
            lambda row: pathlib.Path(row["path"]).stem,
            axis=1
        )

    def load_time_series_module(
        self,
        dataset_idx: int,
        setup: bool = True,
        verbose: bool = False
    ) -> MultiTimeSeriesModule:
        """Create time series module.

        Takes parameters for dataset with provided index,
        load and preprocess data, then pass it to time series module.
        At the end setup created module.

        Parameters
        ----------
        dataset_idx : int
            Index of parameters stored in *datasets_params*.

        Returns
        -------
        MultiTimeSeriesModule
            Data module created based on provided parameters.
        """
        ds_params = self.datasets_params.iloc[dataset_idx]
        max_consecutive_nans = 5
        df = load_and_preprocess(
            dataset_path=ds_params.path,
            load_params=ds_params.load_params,
            drop_pipeline=ds_params.drop_refill_pipeline,
            resample_params=dict(
                resampler_method_str="fillna", rule="1min", resample_kwargs={},
                resampler_method_kwargs=dict(method="backfill")),
            preprocessing_pipeline=ds_params.preprocessing_pipeline,
            detect_anomalies_pipeline=ds_params.detect_anomalies_pipeline,
            undo_resample_before_interpolation=True,
            interpolate_params=dict(method="akima"),
            nan_window_size=300,
            max_nan_in_window=50,
            max_consecutive_nans=max_consecutive_nans,
            scaler=ds_params.scaler,
            training_proportion=ds_params.split_proportions[0],
            verbose=verbose)

        max_break =\
            df.index.to_series().diff().mode()[0] * max_consecutive_nans
        sequences = self._split_ts_where_breaks(
            df, max_break=max_break)
        sequences = self._filter_too_short_series(
            sequences, ds_params.window_size)
        tsm = MultiTimeSeriesModule(
            sequences=sequences,
            dataset_name=ds_params.name_,
            target=ds_params.target,
            split_proportions=ds_params.split_proportions,
            window_size=int(ds_params.window_size),
            batch_size=int(ds_params.batch_size)
        )
        if setup:
            tsm.setup()
        return tsm

    def _split_ts_where_breaks(
        self,
        time_series: Union[pd.Series, pd.DataFrame],
        max_break: Union[int, timedelta] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        index = time_series.index.to_series()
        diffs = index.diff()

        if max_break is None:
            max_break = diffs.mode()[0]
        elif isinstance(max_break, int):
            time_step = diffs.mode()[0]
            max_break *= time_step

        splits = diffs[diffs > max_break].index
        if splits.shape[0] == 0:
            splitted_time_series = [time_series]
        else:
            index = time_series.index
            splitted_time_series = [
                time_series.iloc[:index.get_loc(splits[0])]
            ]
            splitted_time_series += [
                time_series.iloc[
                    index.get_loc(splits[i]):index.get_loc(splits[i+1])]
                for i in range(len(splits)-1)
            ]
            splitted_time_series += [
                time_series.iloc[
                    index.get_loc(splits[-1]):]
            ]

        return splitted_time_series

    def _filter_too_short_series(
        self,
        multi_time_series: List[pd.DateFrame],
        window_size: int
    ) -> List[pd.DateFrame]:
        result = list(filter(
            lambda seq: seq.shape[0] > window_size,
            multi_time_series))
        return result

    def create_lightning_module(
        self,
        model_params: Union[torch.nn.Module, int] = None
    ) -> pl.LightningModule:
        pl_model = model_params.WrapperCls(
            model=model_params.cls_(**model_params.init_params),
            **model_params.learning_params,
            **model_params.wrapper_kwargs)
        return pl_model

    def load_pl_model(
        self,
        model_idx: int,
        dir_path: str,
        file_name: str = None,
        find_last: bool = True,
        create_if_not_found: bool = False,
        get_epoch: bool = False
    ):
        pl_model = self.create_lightning_module(
            model_params=self.models_params.iloc[model_idx])

        if file_name is None:
            file_name = self.exp_date
        if dir_path is None:
            dir_path = self.datasets_params["path"]
        if find_last:
            file_name = self._find_last_model(dir_path, file_name)

        epoch = None
        try:
            loaded_data = torch.load(
                os.path.join(dir_path, file_name + ".ckpt"))
            pl_model.load_state_dict(
                loaded_data["state_dict"])
            if get_epoch:
                epoch = loaded_data["epoch"]
        except FileNotFoundError as e:
            if not create_if_not_found:
                raise e
        if get_epoch:
            return pl_model, epoch
        return pl_model

    def _find_last_model(
        self,
        dir_path: str,
        file_name: str
    ):
        files = [
            file[:-5]
            for file in os.listdir(dir_path)
            if file[:len(file_name)] == file_name and file[-5:] == ".ckpt"]
        i = 1
        found = False
        last_model_file = file_name

        while found is False:
            next_model_file = file_name + f"-v{i}"
            if next_model_file in files:
                last_model_file = next_model_file
                i += 1
            else:
                found = True

        return last_model_file

    def _experiment_step(
        self,
        tsm: MultiTimeSeriesModule,
        model_idx: int,
        continue_run: bool = False
    ):
        pl_model = self.train_model(
            model_idx=model_idx, tsm=tsm, load_state=continue_run)

        # collect predictions made on test dataset
        preds = pl_model.get_dataset_predictions(tsm.test_dataloader())
        return preds

    def _deliver_exception(self, msg: str, exception: Exception, safe: bool):
        if safe:
            print("\n\n==== Warning ====")
            print(msg)
            print(*exception.args)
            print("\n\n")
        else:
            exception.args += (msg,)
            raise exception

    def _set_err_msg(
        self,
        dataset_idx: int,
        model_idx: int = None
    ) -> str:
        ds_name = self.datasets_params.iloc[dataset_idx]["name_"]
        if model_idx is None:
            msg = dataset_setup_exception_msg.substitute(
                dataset_idx=dataset_idx, dataset_name=ds_name)
        else:
            m_name = self.models_params.iloc[dataset_idx]["name_"]
            msg = training_exception_msg.substitute(
                model_idx=model_idx, model_name=m_name,
                dataset_idx=dataset_idx, dataset_name=ds_name)
        return msg

    def _data_setup(self, dataset_idx: int):
        # data loading and transforming
        tsm = self.load_time_series_module(dataset_idx)

        # saving true values
        self.datasets_params["true_values"].iat[dataset_idx] =\
            tsm.test_dataloader().dataset.get_labels()
        return tsm

    def _if_dataset_experiment_finished(
        self, dataset_idx: int
    ):
        return (dataset_idx < self.last_step_end[0] or
                (dataset_idx == self.last_step_end[0] and
                self.last_step_end[0] < self.models_params.shape[0]-1))

    def run_experiments(
        self,
        skip_steps: List[Tuple[int, int]] = [],
        experiments_path: str = None,
        continue_run: bool = False,
        safe: bool = True
    ) -> Experimentator:
        """Executes experiment.

        For every possible pair of provided dataset configuration and model
        setup run step of experiment where: model is trained, validated and
        predictions on test dataset are collected.
        Only steps included in *skip_steps* will be omitted.

        Parameters
        ----------
        skip_steps : List[Tuple[int, int]], optional
            Experiment steps to be skipped. Single element should be tuple
            where first argument is dataset id and second is model id.
            By default [].
        experiments_path : str
            Path where experiment instance will be saved.
        safe : bool, optional
            If True, handles error, prints message and continues experiments,
            if False, pass error. By default True.

        Returns
        -------
        Experimentator
            Executed experiment.
        """

        if continue_run is False:
            self.last_step_end = (-1, -1)
            self.exp_date = time.strftime("%Y-%m-%d_%H:%M:%S")
            # save initial settings in file
            if experiments_path is not None:
                self.predictions = pd.DataFrame(
                    [], columns=["dataset_id", "model_id", "predictions"])
                self.save(experiments_path, safe=False)
        self.checkpoint_params.filename = self.exp_date

        # run experiment
        for dataset_idx in self.datasets_params.index:
            if continue_run and\
                    self._if_dataset_experiment_finished(dataset_idx):
                continue

            # data setup and true values saving
            try:
                tsm = self._data_setup(dataset_idx)
            except Exception as e:
                self._deliver_exception(
                    msg=self._set_err_msg(dataset_idx), exception=e, safe=safe)
                continue

            for model_idx in self.models_params.index:
                if (dataset_idx, model_idx) in skip_steps:
                    # skipping experiment step
                    continue

                try:
                    if continue_run and\
                            model_idx < self.last_step_end[1]:
                        continue
                    # start = time.time()
                    model_preds_df = self._experiment_step(
                        tsm=tsm, model_idx=model_idx,
                        continue_run=continue_run)
                    # print("Time: " + str(time.time() - start))
                    # print("Predictions memory size:\n" +
                    #       str(model_preds_df.memory_usage(deep=True)))

                    # saving store predictions as dataframe
                    self.predictions = pd.concat([
                        self.predictions,
                        pd.DataFrame(
                            [PredictionRecord(
                                dataset_id=dataset_idx,
                                model_id=model_idx,
                                predictions=model_preds_df)]
                        )])
                    self.last_step_end = (dataset_idx, model_idx)

                    # override experiments run file
                    if experiments_path is not None:
                        self.save(experiments_path, safe=True)
                except Exception as e:
                    self._deliver_exception(
                        msg=self._set_err_msg(dataset_idx, model_idx),
                        exception=e,
                        safe=safe,
                    )
                    continue
        print("Experiments ended sucessfully")
        return self

    def get_targets_scaler(
        self,
        dataset_idx: int,
    ) -> TransformerMixin:
        return self.datasets_params.iloc[dataset_idx]["scaler"]

    def _check_if_has_predictions(self):
        assert self.predictions is not None and self.predictions.shape[0] > 0,\
            "Before plotting experiments must be runned."

    def get_models_predictions(
        self,
        dataset_idx: int,
        models_ids: List[int]
    ) -> pd.DataFrame:
        self._check_if_has_predictions()

        if models_ids is None:
            df = self.predictions.loc[
                self.predictions["dataset_id"] == dataset_idx
            ]
        else:
            df = self.predictions.loc[
                (self.predictions["dataset_id"] == dataset_idx) &
                (self.predictions["model_id"].isin(models_ids))
            ]
        return df

    def train_model(
        self,
        model_idx: int,
        dataset_idx: int = None,
        tsm: MultiTimeSeriesModule = None,
        load_state: bool = False
    ) -> TSModelWrapper:
        """Train single model on single dataset from experimentator data.

        Parameters
        ----------
        model_idx : int
            Index of parameters stored in *models_params*.
        dataset_idx : int
            Index of parameters stored in *datasets_params*.
        logs_path : str
            Path where logs will be saved.
        trainer_params : TrainerParams
            Lightning trainer parameters (can contain logger and additional
            callbacks).
        checkpoint_params : CheckpointParams
            Lightning *ModelCheckpoint* init arguments.
        early_stopping_params : EarlyStoppingParams
            Lightning *EarlyStopping* init arguments.
        learning_params : LearningParams, optional
            Learning parameters. If not provided, argument will be replaced
            with parameters saved in experimentator instance. By default None.

        Returns
        -------
        TSModelWrapper
            Trained lightning module wrapping time series model.
        """
        m_params = self.models_params.iloc[model_idx]

        if tsm is None and dataset_idx is not None:
            tsm = self.load_time_series_module(dataset_idx)
        elif tsm is None and dataset_idx is None:
            raise ValueError(
                "Dataset or its index has to be passed.")

        pl.seed_everything(42)

        m_params = self.models_params.iloc[model_idx]
        chp_dirpath = self.checkpoint_params.dirpath

        checkpoint_params = CheckpointParams(**asdict(self.checkpoint_params))
        trainer_params = TrainerParams(**asdict(self.trainer_params))
        early_stopping_params = EarlyStoppingParams(
            **asdict(self.early_stopping_params))
        loggers_params = [
            LoggerParams(**asdict(params))
            for params in self.loggers_params]

        # setting files paths and logger with model name
        for log in loggers_params:
            log.name = m_params.name_
            log.version = self.exp_date
        checkpoint_params.dirpath =\
            os.path.join(chp_dirpath, tsm.name_, m_params.name_)

        if m_params.WrapperCls == Autoencoder or m_params.WrapperCls == VAE:
            m_params.wrapper_kwargs["target_cols_ids"] = tsm.target_cols_ids()

        if load_state:
            pl_model, epoch = self.load_pl_model(
                model_idx=model_idx,
                dir_path=checkpoint_params.dirpath,
                file_name=checkpoint_params.filename,
                find_last=True,
                create_if_not_found=True,
                get_epoch=True)
            trainer_params.max_epochs = trainer_params.max_epochs - epoch
        else:
            pl_model = m_params.WrapperCls(
                model=m_params.cls_(**m_params.init_params),
                **m_params.learning_params,
                **m_params.wrapper_kwargs)

        # training model
        pl_model = get_trained_pl_model(
            pl_model=pl_model,
            data_module=tsm,
            trainer_params=trainer_params,
            checkpoint_params=checkpoint_params,
            early_stopping_params=early_stopping_params,
            loggers_params=loggers_params,
            LoggersClasses=self.LoggersClasses)

        # load last saved model
        pl_model = self.load_pl_model(
            model_idx=model_idx,
            dir_path=checkpoint_params.dirpath,
            file_name=checkpoint_params.filename,
            find_last=True)

        return pl_model

    def change_dataset_path(
        self,
        dataset_idx: int,
        path: str,
        set_name: bool = True
    ):
        """Changes stored path to selected dataset.

        Default changes also name of dataset to name same as name of provided
        path.

        Parameters
        ----------
        dataset_idx : int
            Index of parameters stored in *datasets_params*.
        path : str
            Path to dataset file.
        set_name : bool, optional
            If True, set file name as dataset name. By default True.
        """
        self.datasets_params.at[dataset_idx, 'path'] = path
        if set_name:
            name = os.path.basename(os.path.splitext(path)[0])
            self.datasets_params.at[dataset_idx, 'name_'] = name

    def save(self, path: str, safe: bool = False):
        """Save experimentator to pickle file.

        If path to target directory can't be found and *safe* is True,
        prints message, if *safe* is False, raise *FileNotFoundError*.

        Parameters
        ----------
        path : str
            Path where experimentator instance will be saved.
        safe : bool, optional
            Determines whether *FileNotFoundError* will be
            handled or not. By default False.

        Raises
        ------
        file_not_found_error
            File not found error.
        """
        try:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            with open(f'{path}/{self.exp_date}.pkl', "wb") as file:
                pickle.dump(
                    {
                        "models_params": self.models_params,
                        "datasets_params": self.datasets_params,
                        "trainer_params": self.trainer_params,
                        "checkpoint_params": self.checkpoint_params,
                        "early_stopping_params": self.early_stopping_params,
                        "loggers_params": self.loggers_params,
                        "LoggersClasses": self.LoggersClasses,
                        "predictions": self.predictions,
                        "last_step_end": self.last_step_end,
                        "exp_date": self.exp_date
                    },
                    file
                )

        except FileNotFoundError as file_not_found_error:
            if safe:
                print(f"{path} not found.")
            else:
                raise file_not_found_error


# @staticmethod
def load_experimentator(path: str) -> Experimentator:
    """Loads experimentator instance with saved attributes.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    Experimentator
        Loaded experimentator instance.
    """
    with open(path, "rb") as file:
        attrs = pickle.load(file)
        exp = Experimentator(
            models_params=attrs["models_params"],
            datasets_params=attrs["datasets_params"],
            trainer_params=attrs["trainer_params"],
            checkpoint_params=attrs["checkpoint_params"],
            early_stopping_params=attrs["early_stopping_params"],
            loggers_params=attrs["loggers_params"],
            LoggersClasses=attrs["LoggersClasses"]
        )
        exp.predictions = attrs["predictions"]
        exp.exp_date = attrs["exp_date"]
        return exp

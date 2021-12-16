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
from typing import List, Dict, Type, Tuple, Union
import time
import pathlib
import pickle
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.base import TransformerMixin
from string import Template
import numpy as np
from dataclasses import dataclass, field
from datetime import timedelta, datetime
import os
from dataclasses import asdict
import torch

from predpy.wrapper import Predictor
from predpy.data_module import MultiTimeSeriesModule
from predpy.preprocessing import load_and_preprocess
from predpy.trainer import get_trained_pl_model
from predpy.trainer import (
    TrainerParams, LoggerParams, EarlyStoppingParams, CheckpointParams,
    LearningParams)

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
        learning_params: LearningParams = LearningParams(),
        WrapperCls: Type[pl.LightningDataModule] = Predictor,
        wrapper_kwargs: Dict = {},
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

        # Wrapper params
        self.WrapperCls = WrapperCls
        self.learning_params = learning_params
        self.wrapper_kwargs = wrapper_kwargs

        # Trainer params
        self.trainer_params = trainer_params
        self.checkpoint_params = checkpoint_params
        self.early_stopping_params = early_stopping_params
        self.loggers_params = loggers_params
        self.LoggersClasses = LoggersClasses

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
        setup: bool = True
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
        df = load_and_preprocess(
            dataset_path=ds_params.path,
            load_params=ds_params.load_params,
            drop_refill_pipeline=ds_params.drop_refill_pipeline,
            preprocessing_pipeline=ds_params.preprocessing_pipeline,
            detect_anomalies_pipeline=ds_params.detect_anomalies_pipeline,
            scaler=ds_params.scaler,
            training_proportion=ds_params.split_proportions[0])

        sequences = self._split_ts_where_breaks(df, max_break=4)
        sequences = self._filter_too_short_series(
            sequences, ds_params.window_size)
        tsm = MultiTimeSeriesModule(
            sequences=sequences,
            dataset_name=ds_params.name_,
            target=ds_params.target,
            split_proportions=ds_params.split_proportions,
            window_size=int(ds_params.window_size),
            batch_size=int(ds_params.batch_size),
            DatasetCls=ds_params.DatasetCls
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
        model: Union[torch.nn.Module, int] = None,
        learning_params: Dict = {}
    ) -> pl.LightningModule:
        if isinstance(model, int):
            m_params = self.models_params.iloc[model]
            model = m_params.cls_(**m_params.init_params)
        pl_model = self.WrapperCls(
            model=model, **asdict(learning_params),
            **self.wrapper_kwargs)
        return pl_model

    def load_pl_model(
        self,
        model_idx: int,
        dir_path: str,
        file_name: str = None,
        find_last: bool = True
    ):
        pl_model = self.create_lightning_module(
            model=model_idx, learning_params=self.learning_params)

        if file_name is None:
            file_name = self.exp_date
        if dir_path is None:
            dir_path = self.datasets_params["path"]
        if find_last:
            file_name = self._find_last_model(dir_path, file_name)
        pl_model.load_state_dict(
            torch.load(
                os.path.join(dir_path, file_name + ".ckpt")
            )["state_dict"])
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
        model_idx: int
    ):
        pl_model = self.train_model(model_idx=model_idx, tsm=tsm)

        # collect predictions made on test dataset
        preds = pl_model.get_dataset_predictions(tsm.test_dataloader())
        labels = tsm.test_dataloader().dataset.get_labels()

        if isinstance(labels, pd.DataFrame):
            columns = labels.columns
        else:
            columns = [labels.name]

        return pd.DataFrame(
            data=preds, columns=columns, index=labels.index
        )

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
        msg = None
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

    def run_experiments(
        self,
        skip_steps: List[Tuple[int, int]] = [],
        experiments_path: str = None,
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
        # setting experiment date
        self.exp_date = time.strftime("%Y-%m-%d_%H:%M:%S")
        self.checkpoint_params.filename = self.exp_date

        # init variables
        predictions = []

        # run experiment
        for dataset_idx in self.datasets_params.index:

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
                    model_preds_df = self._experiment_step(
                        tsm=tsm, model_idx=model_idx)
                    predictions.append(
                        PredictionRecord(
                            dataset_id=dataset_idx,
                            model_id=model_idx,
                            predictions=model_preds_df
                        ))
                except Exception as e:
                    self._deliver_exception(
                        msg=self._set_err_msg(dataset_idx, model_idx),
                        exception=e,
                        safe=safe,
                    )
                    continue

        # saving store predictions as dataframe
        self.predictions = pd.DataFrame(predictions)

        # saving experiments run to file
        if len(predictions) == 0:
            print("\n\n==== Warning ====\nNo predictions were made\n\n")
        elif experiments_path is not None:
            self.save(experiments_path, safe=True)

        return self

    def get_targets_scaler(
        self,
        dataset_idx: int,
    ) -> TransformerMixin:
        return self.datasets_params.iloc[dataset_idx]["scaler"]

    def _check_if_has_predictions(self):
        assert self.predictions is not None and self.predictions.shape[0] > 0,\
            "Before plotting experiments must be runned."

    def _get_models_predictions(
        self,
        dataset_idx: int,
        models_ids: List[int]
    ) -> pd.DataFrame:
        self._check_if_has_predictions()

        df = None
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

    def _rescale_true_vals_and_preds(
        self,
        scaler_dataset_idx: int,
        true_vals: Union[pd.Series, pd.DataFrame],
        predictions_df: pd.DataFrame,
        target: Union[str, List[str]]
    ) -> Tuple[List[float], pd.DataFrame]:
        vals = self._scale_inverse(
            true_vals, scaler_dataset_idx, target)
        predictions = \
            predictions_df["predictions"].apply(
                lambda preds: self._scale_inverse(
                    preds, scaler_dataset_idx, target))
        return vals, predictions

    def _concat_dfs_with_gaps(
        self,
        dfs: Union[List[pd.Series], List[pd.DataFrame]],
    ) -> pd.DataFrame:
        dfs_with_gaps = []
        if isinstance(dfs[0], pd.DataFrame):
            for df in dfs:
                dfs_with_gaps += [
                    df, pd.DataFrame(data=[df.iloc[0]], index=[''])]
        else:
            for df in dfs:
                dfs_with_gaps += [
                    df,
                    pd.Series(data=[df.iloc[0]], index=[''], name=df.name)]
        return pd.concat(dfs_with_gaps)

    def ts_to_plotly_data(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        name: str,
        version: str = ""
    ) -> List[go.Scatter]:
        data = []
        if isinstance(ts, pd.Series):
            data += [go.Scatter(
                x=ts.index, y=ts, connectgaps=False,
                name=name + version)]
        elif isinstance(ts, pd.DataFrame):
            if len(ts.columns) > 1:
                data += [go.Scatter(
                        x=ts.index, y=ts[col], connectgaps=False,
                        name=name + f"-{col}" + version)
                    for col in ts.columns
                ]
            else:
                data += [go.Scatter(
                    x=ts.index, y=ts.iloc[:, 0], connectgaps=False,
                    name=name + version)]
        return data

    def _true_vals_and_preds_to_plotly_data(
        self,
        true_vals: Union[pd.Series, pd.DataFrame],
        predictions_df: pd.DataFrame,
        version: str = ""
    ) -> List[go.Scatter]:
        true_vals = self._split_ts_where_breaks(true_vals)
        true_vals = self._concat_dfs_with_gaps(true_vals)

        data = self.ts_to_plotly_data(true_vals, "true_values", version)
        for _, row in predictions_df.iterrows():
            model_name = self.models_params.iloc[row["model_id"]].name_
            preds = self._split_ts_where_breaks(row["predictions"])
            preds = self._concat_dfs_with_gaps(preds)
            data += self.ts_to_plotly_data(preds, model_name, version)
        return data

    def plot_preprocessed_dataset(
        self,
        dataset_idx: int,
        rescale: bool = False,
        file_path: str = None,
        start: int = None,
        end: int = None
    ):
        tsm = self.load_time_series_module(dataset_idx)

        ts_with_gaps = []
        train_end = tsm.train_range[1]
        val_start, val_end = tsm.val_range
        test_start = tsm.test_range[0]

        train_end_add = 0
        val_start_add, val_end_add = 0, 0
        test_start_add = 0

        start_add = 0
        end_add = 0
        counter = 0
        if start is None:
            start = 0
        if end is None:
            end = tsm.test_range[1] + 1 + tsm.window_size

        for ts in tsm.sequences:
            ts_with_gaps += [ts]
            # adding gap
            ts_with_gaps += [pd.DataFrame(
                data=[ts.iloc[-1].to_dict()], index=[''])]

            # take into account gap
            counter += ts.shape[0]

            if train_end >= counter:
                train_end_add += 1

            if val_start >= counter:
                val_start_add += 1
            if val_end >= counter:
                val_end_add += 1

            if test_start >= counter:
                test_start_add += 1

            if start >= counter:
                start_add += 1
            if end > counter:
                end_add += 1

        train_end += train_end_add + tsm.window_size

        val_start += val_start_add
        val_end += val_end_add + tsm.window_size

        test_start += test_start_add

        end += end_add
        start += start_add

        # time series dataframe concatenation
        df = pd.concat(ts_with_gaps)
        df = df.iloc[start:end]

        if rescale:
            df[df.columns] = self._scale_inverse(df, dataset_idx)

        data = []
        for col in df.columns:
            data += [go.Scatter(
                x=df.index, y=df[col], name=col, connectgaps=False)]

        layout = go.Layout(
            title=self.datasets_params.iloc[dataset_idx].name_,
            yaxis=dict(title="values"),
            xaxis=dict(title='dates')
        )

        fig = go.Figure(data=data, layout=layout)

        # adding v-lines splitting train, val and test datasets
        start = df.index[start]
        end = df.index[end-1]
        self._add_dataset_type_vrect(
            fig, df.index[0], df.index[train_end], start, end,
            fillcolor="blue", opacity=0.3, layer="below", line_width=1,
            annotation_text="train")
        self._add_dataset_type_vrect(
            fig, df.index[val_start], df.index[val_end], start, end,
            fillcolor="yellow", opacity=0.3, layer="below", line_width=1,
            annotation_text="validation")
        self._add_dataset_type_vrect(
            fig, df.index[test_start], df.index[-1], start, end,
            fillcolor="red", opacity=0.3, layer="below", line_width=1,
            annotation_text="test")

        if file_path is not None:
            plot(fig, filename=file_path)
        else:
            fig.show()

    def _add_dataset_type_vrect(
        self,
        fig: go.Figure,
        x0: Union[datetime, int],
        x1: Union[datetime, int],
        start: Union[datetime, int],
        end: Union[datetime, int],
        **kwargs
    ):
        if x0 < start:
            x0 = start
        if x1 > end:
            x1 = end
        if x0 < x1:
            fig.add_vrect(x0=x0, x1=x1, **kwargs)

    def plot_predictions(
        self,
        dataset_idx: int,
        models_ids: List[int] = None,
        rescale: bool = False,
        file_path: str = None
    ):
        """Plots selected prediction made during experiment run and true values.

        If any step of experiment hasn't been made, raise AssertionError.

        Parameters
        ----------
        dataset_idx : int
            Index of parameters stored in *datasets_params*.
        models_ids : List[int], optional
            Group of index of parameters stored in *models_params*.
            If provided, plot only for selected models, if not, plot for
            all models form predictions. By default None.
        rescale : bool = False
            If True, rescale true values and predictions using scaler
            assigned to dataset.
        file_path : str, optional
            If type is string, chart will be saved to html file with provided
            path.
        """
        predictions_df = self._get_models_predictions(dataset_idx, models_ids)
        ds_params = self.datasets_params.iloc[dataset_idx]
        target = ds_params.target

        if rescale:
            self._rescale_true_vals_and_preds(
                dataset_idx, ds_params.true_values, predictions_df, target)

        data = self._true_vals_and_preds_to_plotly_data(
            ds_params.true_values, predictions_df)

        layout = go.Layout(
            title=ds_params.name_,
            yaxis=dict(title=target),
            xaxis=dict(title='dates')
        )

        fig = go.Figure(data=data, layout=layout)
        if file_path is not None:
            plot(fig, filename=file_path)
        else:
            fig.show()

    def _get_target_columns_ids(
        self,
        target_name: Union[str, List[str]],
        scaler: TransformerMixin
    ) -> List[int]:
        result = []
        if isinstance(target_name, str):
            idx = np.where(scaler.feature_names_in_ == target_name)[0][0]
            result = [idx]
        elif isinstance(target_name, list):
            for t in target_name:
                idx = np.where(scaler.feature_names_in_ == t)[0][0]
                result += [idx]
        return result

    def _scale_inverse(
        self,
        time_series: Union[pd.Series, pd.DataFrame],
        scaler_dataset_idx: int,
        target_name: Union[str, List[str]] = None
    ) -> List[float]:
        scaler = self.get_targets_scaler(scaler_dataset_idx)

        result = None
        if target_name is not None and isinstance(time_series, pd.Series):
            cols_ids = self._get_target_columns_ids(target_name, scaler)
            scaler_input = np.array(
                [time_series.tolist()] * scaler.n_features_in_).T
            result = scaler.inverse_transform(scaler_input)
            result = result.T[cols_ids]
        elif target_name is not None and isinstance(time_series, pd.DataFrame):
            cols_ids = self._get_target_columns_ids(target_name, scaler)
            mocked_column = [0] * time_series.shape[0]
            scaler_input = []

            for scaler_feature in scaler.feature_names_in_:
                if scaler_feature in time_series.columns:
                    scaler_input += [time_series[scaler_feature].tolist()]
                else:
                    scaler_input += [mocked_column]
            scaler_input = np.array(scaler_input).T

            result = scaler.inverse_transform(scaler_input)
            result = result.T[cols_ids]
        elif target_name is None and isinstance(time_series, pd.DataFrame):
            result = scaler.inverse_transform(time_series)

        return result

    def train_model(
        self,
        model_idx: int,
        dataset_idx: int = None,
        tsm: MultiTimeSeriesModule = None
    ) -> pl.LightningModule:
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
        pl.LightningModule
            Trained lightning module.
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

        pl_model = self.WrapperCls(
            model=m_params.cls_(**m_params.init_params),
            **asdict(self.learning_params),
            **self.wrapper_kwargs)

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
                        "learning_params": self.learning_params,
                        "WrapperCls": self.WrapperCls,
                        "wrapper_kwargs": self.wrapper_kwargs,
                        "trainer_params": self.trainer_params,
                        "checkpoint_params": self.checkpoint_params,
                        "early_stopping_params": self.early_stopping_params,
                        "loggers_params": self.loggers_params,
                        "LoggersClasses": self.LoggersClasses,
                        "predictions": self.predictions,
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
            learning_params=attrs["learning_params"],
            WrapperCls=attrs["WrapperCls"],
            wrapper_kwargs=attrs["wrapper_kwargs"],
            trainer_params=attrs["trainer_params"],
            checkpoint_params=attrs["checkpoint_params"],
            early_stopping_params=attrs["early_stopping_params"],
            loggers_params=attrs["loggers_params"],
            LoggersClasses=attrs["LoggersClasses"]
        )
        exp.predictions = attrs["predictions"]
        exp.exp_date = attrs["exp_date"]
        return exp


@dataclass
class ExperimentatorPlot:
    '''Data class representing experimentator plotting parameters

    *datasets_to_models* should be a List of dictionaries mapping every
    dataset to models which predictions you want to plot.
    '''
    experimentator: Experimentator
    datasets_to_models: Dict[int, List[int]] = field(default_factory=list)
    rescale: bool = True


def plot_aggregated_predictions(
    exps_params:
        Union[List[Experimentator], List[ExperimentatorPlot]],
    file_path: str = None
):
    """Plots selected prediction from list of experimentator.
    To point which predictions should be plotted, use ExperimentatorPlot.
    ExperimentatorPlot without set *datasets_to_models* cause all
    experimentator predictions plotting, the same as passing list of
    *Experimentator* instead of list of *ExperimentatorPlot*.

    Parameters
    ----------
    exps : Union[List[Experimentator], List[ExperimentatorPlot]]
        List of experimentators or list of experimentators plot params.
    file_path : str, optional
        If type is string, chart will be saved to html file with provided
        path.
    """
    data = []

    for exp_params in exps_params:
        exp, rescale, datasets_to_models = None, None, None
        if isinstance(exp_params, Experimentator):
            exp = exp_params
            rescale = True
            datasets_to_models = []
        else:
            exp = exp_params.experimentator
            datasets_to_models = exp_params.datasets_to_models
            rescale = exp_params.rescale

        if len(datasets_to_models) == 0:
            datasets_to_models = {
                ds_idx: exp.models_params.index.tolist()
                for ds_idx in exp.datasets_params.index.tolist()
            }
        for dataset_idx, models_ids in datasets_to_models.items():
            predictions_df = exp._get_models_predictions(
                dataset_idx, models_ids)
            ds_params = exp.datasets_params.iloc[dataset_idx]
            target = ds_params.target

            if rescale:
                exp._rescale_true_vals_and_preds(
                    dataset_idx, ds_params.true_values, predictions_df, target)

            data += exp._true_vals_and_preds_to_plotly_data(
                ds_params.true_values, predictions_df,
                version=f", exp: {exp.exp_date}, ds: {dataset_idx}")

    layout = go.Layout(
        title=ds_params.name_,
        yaxis=dict(title=target),
        xaxis=dict(title='dates')
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        fig.show()

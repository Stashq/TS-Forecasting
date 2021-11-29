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
from typing import List, Dict, Type, Tuple
import time
import pathlib
import pickle
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.base import TransformerMixin
from string import Template
import numpy as np

from predpy.wrapper import Predictor
from predpy.data_module import TimeSeriesModule
from predpy.preprocessing import load_and_preprocess
from predpy.trainer import get_trained_pl_model
from predpy.trainer import (
    TrainerParams, LoggerParams, EarlyStoppingParams, CheckpointParams,
    LearningParams)

from .experimentator_params import DatasetParams, ModelParams, PredictionRecord

dataset_setup_exception_msg = Template(
    "Error during setup $dataset_idx dataset named $dataset_name.")
training_exception_msg = Template(
    "Problem with training $model_idx model named $model_name on"
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
        wrapper_kwargs: Dict = {}
    ):
        """

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
        """
        self.set_experiment_params(models_params, datasets_params)

        # Wrapper params
        self.WrapperCls = WrapperCls
        self.learning_params = learning_params
        self.wrapper_kwargs = wrapper_kwargs

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

    def get_preprocessed_data(
        self,
        dataset_idx: int,
    ) -> TimeSeriesModule:
        """Create time series module.

        Takes parameters for dataset with provided index,
        load and preprocess data, then pass it to time series module.
        At the end setup created module.

        Additionaly if dataset parameters dictionary has scaler, fits it on
        training dataset.

        Parameters
        ----------
        dataset_idx : int
            Index of parameters stored in *datasets_params*.

        Returns
        -------
        TimeSeriesModule
            Data module created based on provided parameters.
        """
        ds_params = self.datasets_params.iloc[dataset_idx]
        df = load_and_preprocess(
            ds_params.path, ds_params.load_params,
            ds_params.drop_refill_pipeline, ds_params.preprocessing_pipeline,
            ds_params.scaler,
            training_proportion=ds_params.split_proportions[0])

        tsm = TimeSeriesModule(
            sequence=df,
            dataset_name=ds_params.name_,
            target=ds_params.target,
            split_proportions=ds_params.split_proportions,
            window_size=int(ds_params.window_size),
            batch_size=int(ds_params.batch_size),
            DatasetCls=ds_params.DatasetCls
        )
        tsm.setup()
        return tsm

    def experiment_step(
        self,
        trainer_params: TrainerParams,
        checkpoint_params: CheckpointParams,
        early_stopping_params: EarlyStoppingParams,
        logger_params,
        chp_path,
        tsm,
        predictions,
        ds_idx,
        m_idx
    ):
        pl.seed_everything(42)

        m_params = self.models_params.iloc[m_idx]

        # setting files paths and logger with model name
        logger_params.name = m_params.name_
        checkpoint_params.dirpath =\
            f"{chp_path}/{tsm.name_}/{m_params.name_}"

        # training model
        pl_model = get_trained_pl_model(
            model=m_params.cls_(**m_params.init_params),
            data_module=tsm, trainer_params=trainer_params,
            logger_params=logger_params,
            checkpoint_params=checkpoint_params,
            early_stopping_params=early_stopping_params,
            learning_params=self.learning_params,
            WrapperCls=self.WrapperCls,
            wrapper_kwargs=self.wrapper_kwargs
            )

        # collecting model prediction on test dataset
        preds = pl_model.get_dataset_predictions(tsm.test_dataloader())
        predictions.append(
            PredictionRecord(
                dataset_id=ds_idx, model_id=m_idx, predictions=preds
            ))

    def _deliver_exception(self, msg, exception, safe):
        if safe:
            print("\n\n==== Warning ====")
            print(msg)
            print(*exception.args)
            print("\n\n")
        else:
            exception.args += (msg,)
            raise exception

    def _data_setup(self, ds_idx: int):

        # data loading and transforming
        tsm = self.get_preprocessed_data(ds_idx)

        # saving true values
        self.datasets_params["true_values"].iat[ds_idx] =\
            tsm.test_dataloader().dataset.get_labels()
        return tsm

    def run_experiments(
        self,
        logs_path: str,
        trainer_params: TrainerParams,
        checkpoint_params: CheckpointParams,
        early_stopping_params: EarlyStoppingParams,
        skip_points: List[Tuple[int, int]] = [],
        experiments_path: str = None,
        safe: bool = True
    ) -> Experimentator:
        """Executes experiment.

        For every possible pair of provided dataset configuration and model
        setup run step of experiment where: model is trained, validated and
        predictions on test dataset are collected.
        Only steps included in *skip_points* will be omitted.

        Parameters
        ----------
        logs_path : str
            Path where logs will be saved.
        trainer_params : TrainerParams
            Lightning trainer parameters (can contain logger and additional
            callbacks).
        checkpoint_params : CheckpointParams
            Lightning *ModelCheckpoint* init arguments.
        early_stopping_params : EarlyStoppingParams
            Lightning *EarlyStopping* init arguments.
        skip_points : List[Tuple[int, int]], optional
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
        checkpoint_params.filename = self.exp_date

        # init variables
        logger_params = LoggerParams(logs_path, None, self.exp_date)
        chp_path = checkpoint_params.dirpath
        predictions = []

        # run experiment
        for ds_idx in self.datasets_params.index:

            # data setup and true values saving
            try:
                tsm = self._data_setup(ds_idx)
            except Exception as e:
                ds_name = self.datasets_params.iloc[ds_idx]["name_"]
                self._deliver_exception(
                    msg=dataset_setup_exception_msg.substitute(
                        dataset_idx=ds_idx, dataset_name=ds_name),
                    exception=e, safe=safe
                )
                continue

            for m_idx in self.models_params.index:

                if (ds_idx, m_idx) in skip_points:
                    # skipping experiment step
                    continue

                try:
                    self.experiment_step(
                        trainer_params,
                        checkpoint_params,
                        early_stopping_params,
                        logger_params,
                        chp_path,
                        tsm,
                        predictions,
                        ds_idx,
                        m_idx)
                except Exception as e:
                    ds_name = self.datasets_params.iloc[ds_idx]["name_"]
                    m_name = self.models_params.iloc[ds_idx]["name_"]
                    self._deliver_exception(
                        msg=training_exception_msg.substitute(
                            model_idx=m_idx, model_name=m_name,
                            dataset_idx=ds_idx, dataset_name=ds_name),
                        exception=e, safe=safe
                    )
                    continue

        # saving experiments outputs in experimentator instance
        self.predictions = pd.DataFrame(predictions)

        # saving experiments run to file
        if len(predictions) == 0:
            print("\n\n==== Warning ====")
            print("No predictions were made")
            print("\n\n")
        elif experiments_path is not None:
            self.save(experiments_path, safe=True)

        return self

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
                pickle.dump(self, file)
        except FileNotFoundError as file_not_found_error:
            if safe:
                print(f"{path} not found.")
            else:
                raise file_not_found_error

    def get_target_scaler(
        self,
        dataset_idx: int,
        # scaler: TransformerMixin
    ) -> TransformerMixin:
        return self.datasets_params.iloc[dataset_idx]["scaler"]
        # df = pd.read_csv(ds_params.path, **ds_params.load_params)
        # return fit_scaler(
        #     df[[ds_params.target]],
        #     ds_params.split_proportions[0],
        #     scaler
        # )
        """Creates scaler for target column of selected dataset.

        Fits scaler based on training data.

        Parameters
        ----------
        dataset_idx : int
            Index of parameters stored in *datasets_params*.
        scaler : TransformerMixin
            Scaler instance.

        Returns
        -------
        TransformerMixin
            Fitted scaler.
        """

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
        assert self.predictions is not None and self.predictions.shape[0] > 0,\
            "Before plotting experiments must be runned."

        df = None
        if models_ids is None:
            df = self.predictions.loc[
                self.predictions["dataset_id"] == dataset_idx
            ]
        else:
            df = self.predictions.loc[
                (self.predictions["dataset_id"] == dataset_idx) &
                (self.predictions["models_ids"].isin(models_ids))
            ]
        ds_params = self.datasets_params.iloc[dataset_idx]

        x = ds_params.true_values.index.tolist()
        true_vals = ds_params.true_values.tolist()
        target = ds_params.target

        if rescale:
            scaler = self.get_target_scaler(dataset_idx)

            true_vals = self._scale_inverse_preds(true_vals, scaler, target)
            df["predictions"] = \
                df[["predictions"]].apply(
                    lambda preds: self._scale_inverse_preds(
                        preds.tolist()[0], scaler, target),
                    axis=1)

        data = [go.Scatter(x=x, y=true_vals, name="True values")]

        for _, row in df.iterrows():
            model_name = self.models_params.iloc[row["model_id"]].name_
            data += [go.Scatter(x=x, y=row["predictions"],
                                name=model_name)]

        layout = go.Layout(
            title=ds_params.name_,
            yaxis=dict(title=target),
            xaxis=dict(title='dates')
        )

        fig = go.Figure(data=data, layout=layout)
        if isinstance(file_path, str):
            plot(fig, filename=file_path)
        else:
            fig.show()

    def _scale_inverse_preds(
        self,
        preds: List[float],
        scaler: TransformerMixin,
        target_name: str
    ) -> List[float]:
        target_col_idx = np.where(
            scaler.feature_names_in_ == target_name)[0][0]
        zipped_preds = np.array([[val]*scaler.n_features_in_ for val in preds])

        result = scaler.inverse_transform(zipped_preds)
        result = result.T[target_col_idx]
        return result

    def retrain_model(
        self,
        model_idx: int,
        dataset_idx: int,
        logs_path: str,
        trainer_params: TrainerParams,
        checkpoint_params: CheckpointParams,
        early_stopping_params: EarlyStoppingParams,
        learning_params: LearningParams = None
    ) -> pl.LightningModule:
        """Train single model on single dataset from experimentator data
        without saving predictions.

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
        if learning_params is None:
            learning_params = self.learning_params
        exp_date = time.strftime("%Y-%m-%d_%H:%M:%S")
        checkpoint_params.filename = exp_date

        m_params = self.models_params.iloc[model_idx]
        tsm = self.get_preprocessed_data(dataset_idx)
        logger_params = LoggerParams(logs_path, m_params.name_, exp_date)

        return get_trained_pl_model(
            model=m_params.cls_(**m_params.hyperparams),
            data_module=tsm, trainer_params=trainer_params,
            logger_params=logger_params,
            checkpoint_params=checkpoint_params,
            early_stopping_params=early_stopping_params,
            learning_params=learning_params,
            WrapperCls=self.WrapperCls,
            wrapper_kwargs=self.wrapper_kwargs
        )

    def change_dataset_path(
        self,
        dataset_idx: int,
        path: str,
        set_name: bool = True
    ):
        """Changes stored path to selected dataset.

        Additionaly changes name of dataset to name same as name of provided
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
        #  ################ SET NAME ###########################

    @staticmethod
    def load_experimentator(path: str) -> Experimentator:
        """Loads experimentator instance from file.

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
            return pickle.load(file)

"""Package for automation experiments, versioning and reproducting them,
also getting their desirable parts.\n

*experimentator_params* provides dataclasses representing experimentator fields
like datasets and models parameters, also predictions of trained models.\n

*experimentator* automates:
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
# flake8:noqa
from .experimentator import (
    Experimentator, plot_aggregated_predictions, load_experimentator,
    ExperimentatorPlot)
from .experimentator_params import (
    DatasetParams, ModelParams, PredictionRecord, LearningParams)

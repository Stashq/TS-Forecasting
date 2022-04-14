"""Package contains modules facilitating training params setting and training
process.\n

*training params* module provides dataclasses with default parameters setting.
Use them to create Experimentator instance from :py:mod:`experimentator`.
*training* module enable to initiate lightning trainer, also create and train
lightning model. Uses *training params* modules dataclasses.\n

In simple examples it is recommended to use pl.Trainer rather those modules.
However if many arguments have to be given or some parameters change during
experiment, this package could be useful. It is also used in
:py:mod:`experimentator` module.
"""
# flake8:noqa
from .training import (
    get_trained_pl_model, get_trainer)
from .training_params import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)

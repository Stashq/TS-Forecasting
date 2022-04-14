"""Package provides Lightning modules wrapping any time series predicting model
defined as pytroch module.

You can define your own wrapping module inheriting from one of those package
modules and pass them to :py:mod:`experimentator` as *WrappingCls*.

Warning:
To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
# flake8:noqa
from .base import ModelWrapper
from .predictor import Predictor
from .autoencoder import Autoencoder
from .vae import VAE
from .pae import PAE
from .pvae import PVAE

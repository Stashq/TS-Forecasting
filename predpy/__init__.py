"""Package makes time series experiments much easier.

Use :py:mod:`experimentator` module to automate:
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

...........................................................???????????????
How it is easier:
* to create dataset, even 2 types
* use lightning module and data module for time series if all is implemented
* write complicated functions including training if all parameters and training
functions are implemented 

But also experiments with this package are memory-efficient.

What you have to store to be able 
"""
# flake8:noqa

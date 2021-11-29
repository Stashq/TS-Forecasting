"""Package provides customized pytorch lightning data module for time series.\n

*TimeSeriesModule* splits single time series for training, validation and test
datasets which are created *dataset* modules. For more details read
its and *LightningDataModule* documentation.\n
"""
# flake8:noqa
from .time_series_module import TimeSeriesModule
from .multi_time_series_module import MultiTimeSeriesModule

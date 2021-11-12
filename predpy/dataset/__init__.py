"""Package provide pytorch dataset classes customized for time series.\n

*BaseTimeSeriesDataset* is abstract class from which rest inherit.\n

*TimeSeriesRecordsDataset* class stores data as records.
Every record is a tuple containing a sequence (model input data)
and a single target value following after sequence (predicted value).
Because of redundancy, class object size grows linearly proportional
to window size.\n

*SingleTimeSeriesDataset* solve this problem - its samples share same memory
so created object is much lighter than *TimeSeriesRecodsDataset* object.
It is strongly advised not to change class object during usage.
"""
# flake8:noqa
from .time_series_records_dataset import TimeSeriesRecordsDataset
from .single_time_series_dataset import SingleTimeSeriesDataset
from .time_series_dataset import TimeSeriesDataset

"""Package for preprocessing time series.

Main function **load_and_preprocess** loads time series from csv file,
converts it to dataframe, and preprocess it with functions defined in
*pipeline*. Every *pipeline* element consists from tuple of function
and its arguments restricting rules described in :py:mod:`load_and_preprocess`.

*set_index*, *scale* are example functions that can be passed to the pipeline.

Also functions creating new features like *moving_average* can be passed
to it.

**seq_to_records** is used to create dataset for *TimeSeriesRecordsDataset*
module :py:mod:`time_series_records_dataset`, where every record is a tuple
containing a sequence (model input data) and a single target value following
after sequence (predicted value).
"""
# flake8:noqa
from .load_and_preprocess import load_and_preprocess
from .preprocessing import (
    set_index, fit_scaler, scale, drop_if_is_in, use_dataframe_func,
    drop_if_index_is_in, drop_if_equals, drop_if_index_equals, loc, iloc)
from .feature_engineering import moving_average
from .seq_to_records import seq_to_records, Record

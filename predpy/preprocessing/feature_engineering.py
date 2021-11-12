"""Modue provide functions for feature engineering purposes.

They takes time series dataframe, calculate new features for selected
columns, join them to dataframe, additionaly remove records for which
features can not be calculated, and return dataframe with new columns.

Most of them are compatibile with *load_and_preprocess* function
from :py:mod:`load_and_preprocess` and can be passed to the pipeline
(conditions described in the module).
"""
import pandas as pd
from typing import List


def moving_average(
    time_series: pd.DataFrame,
    window_size: int,
    col_names: List[str],
    drop: bool = True
) -> pd.DataFrame:
    """Appends moving average columns to time series dataframe.\n\n

    For every indicated column from input dataframe produces moving average
    values calculated from sequences with window size length.\n
    Saves them into dataframe as "{column_name}_MA_{window_size}".\n
    By default drop first *window_size* records (those records has not
    assigned values in moving average columns).

    Parameters
    ----------
    time_series : pd.DataFrame
        Input time series.
    window_size : int
        Length of a sequences for moving average calculations.
    col_names : List[str]
        Time series dataframe columns provided the basis for a moving average.
    drop : bool, optional
        If True, first *window_size* records will be dropped. By default True.

    Returns
    -------
    pd.DataFrame
        Time series with joined new columns.
    """
    for col in col_names:
        time_series[f"{col}_MA_{window_size}"] =\
            time_series[col].rolling(window_size).mean()

    if drop:
        # Drop first records without assigned values.
        time_series = time_series.iloc[window_size:]
    return time_series

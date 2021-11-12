"""Modue provide functions for preprocessing purposes.

Most of them are compatibile with *load_and_preprocess* function
from :py:mod:`module_2` and can be passed to the pipeline (conditions
described in the module).
"""
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from typing import List, Union, Any, Literal


def set_index(
    time_series: pd.DataFrame,
    column_name: str,
    to_datetime: bool = True
) -> pd.DataFrame:
    """Set index of dataframe. Additionaly change it to datetime format.

    Parameters
    ----------
    time_series : pd.DataFrame
        Transformed time series.
    column_name : str
        Name of dataframe column to be set as index.
    to_datetime : bool, optional
        If True, change index to datetime format. By default True.

    Returns
    -------
    pd.DataFrame
        Transformed time series.
    """
    time_series = time_series.set_index(column_name)
    if to_datetime:
        time_series.index = pd.to_datetime(time_series.index)
    return time_series


def fit_scaler(
    time_series: pd.DataFrame,
    training_fraction: float,
    scaler: TransformerMixin = MinMaxScaler
) -> TransformerMixin:
    """Fits a scaler.

    For training purpose takes first *n* input values, where *n*
    is calculated based on input length and provided fraction.
    If fraction is negative or greater than 1, raises AssertionError.

    Parameters
    ----------
    time_series : pd.DataFrame
        Transformed time series.
    training_fraction : float
        Portion of data to train scaler.
        Should be non-negative and less or equal 1.
    scaler : TransformerMixin, optional
        Scaler instance, by default sklearn MinMaxScaler.

    Returns
    -------
    TransformerMixin
        Fitted scaler.
    """
    assert training_fraction >= 0, "Training fraction can't be negative."
    assert training_fraction <= 1, "Training fraction can't be greater than 1."
    margin = int(time_series.shape[0] * training_fraction)

    # traning scaler on training data
    return scaler.fit(time_series[:margin])


def scale(
    time_series: pd.DataFrame,
    training_fraction: float,
    scaler: TransformerMixin = MinMaxScaler
) -> pd.DataFrame:
    """Scales time series.

    Uses provided scaler class, by defaily MinMaxScaler.
    Scaler is trained with first *n* values from input data, where *n* is
    calculated based on input length and provided fraction.

    Parameters
    ----------
    time_series : pd.DataFrame
        Transformed time series.
    training_fraction : float
        Portion of data to train scaler.
        Should be non-negative and less or equal 1.
    scaler : TransformerMixin, optional
        Scaler class, by default sklearn MinMaxScaler.

    Returns
    -------
    pd.DataFrame
        Transformed time series.
    """
    scaler = fit_scaler(time_series, training_fraction, scaler)

    return pd.DataFrame(
        scaler.transform(time_series),
        index=time_series.index,
        columns=time_series.columns
    )


def to_datetime(
    time_series: pd.DataFrame,
    columns: Union[str, List[str]],
    format: str
) -> pd.DataFrame:
    """Convert dataframe columns to datetime format.

    Parameters
    ----------
    time_series : pd.DataFrame
        Transformed time series.
    columns : Union[str, List[str]]
        Column or columns names that have to be converted.
    format : str
        The strftime to parse time.

    Returns
    -------
    pd.DataFrame
        Transformed time series.
    """
    if isinstance(columns, str):
        time_series[columns] =\
            pd.to_datetime(time_series[columns], format=format)
    else:
        for col in columns:
            time_series[col] = pd.to_datetime(time_series[col], format=format)

    return time_series


def drop_if_is_in(
    time_series: pd.DataFrame,
    rejected_values: List[Any],
    columns: List[str] = None,
    how: Literal["any", "all"] = "any",
    axis: int = 1
) -> pd.DataFrame:
    assert isinstance(rejected_values, (tuple, list)),\
        "\"rejected_values\" should be a tuple or a list."
    assert len(rejected_values) > 0, "\"rejected_values\" is empty."
    assert isinstance(columns, (tuple, list)),\
        "\"columns\" should be a tuple or a list."

    if columns is None:
        columns = time_series.columns

    is_rejected_value = time_series[columns].isin(rejected_values)
    if how == "any":
        to_reject = is_rejected_value.any(axis=axis)
    elif how == "all":
        to_reject = is_rejected_value.all(axis=axis)
    else:
        raise ValueError("Unknown scope. Value should be \"any\" or \"all\".")

    time_series = time_series[~to_reject]

    return time_series


def use_dataframe_func(time_series, func_name, *args, **kwargs):
    func = getattr(time_series, func_name)
    time_series = func(*args, **kwargs)
    return time_series


def select_columns(time_series, columns):
    return time_series[columns]

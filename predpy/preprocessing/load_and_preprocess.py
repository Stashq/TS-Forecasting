"""Modue provide function **load_and_preprocess** function,
which allows to load and preprocess a data in the way defined in *pipeline*.

Creating *pipeline* you have to follow the rules:
* every element has to be a callable, tuple or list of preprocessing
functions and additional arguments,
* first element has to be a function, it has to take as first argument
preprocessed dataframe and after transformation, return only it; preprocessing
function can also create new features; in module :py:mod:`preprocessing` are
examples of those, you can aslo create you own preprocessing function,
* you can pass *args or **kwargs or both of them to transforming function,
just remember to pass it in this order.
"""
from typing import Callable, Dict, List, Tuple, Union
import pandas as pd
from tqdm import tqdm
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from string import Template


UNKNOWN_TARTEG_TYPE = Template(
    "Unknown type of target: $target_type. Allowed: str, list.")


def load_and_preprocess(
    dataset_path: str,
    load_params: Dict = {},
    drop_refill_pipeline: List[Union[Callable, Tuple, List]] = [],
    preprocessing_pipeline: List[Union[Callable, Tuple, List]] = [],
    detect_anomalies_pipeline: List[Union[Callable, Tuple, List]] = [],
    scaler: TransformerMixin = None,
    training_proportion: float = None,
    verbose: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TransformerMixin]]:
    """Loads data and preprocesses it with functions provided in in pipeline.\n

    Pipeline elements are tuple containing function and additional arguments.
    Functions should take transforming pandas dataframe as first argument,
    which will be passed between them automaticly.
    List of preprocessing commands. Pipeline has to be created
    with the following rules:
        * every element has to be a callable, tuple or list of preprocessing
        functions and additional arguments,
        * first parameter has to be a function, it has to take as first
        argument preprocessed dataframe and after transformation, return
        only it; preprocessing function can also create new features;
        in module :py:mod:`preprocessing` are examples of those, you can also
        create your own preprocessing function sticking to these rules,
        * you can pass *args or **kwargs or both of them to transforming
        function, just remember to pass it in this order.
        By default None.

    Function use three pipelines:
        * drop_refill_pipeline - functions dropping unwanted data and refilling
        missing values; runs first,
        * preprocessing_pipeline - functions preprocessing cleaned time series;
        runs second,
        * detect_anomalies_pipeline - functions returning fiters for anomalies
        (normal data <- True, anomaly <- False); pipeline if runned after
        scaling, create logical sum from filters and apply result on dataframe.
    Remember to pass functions to right pipeline.

    If *scaler* and *training_proportion* is defined,
    at the end of preprocessing scales result time series
    and returns it with preprocessed data.

    Parameters
    ----------
    dataset_path : str
        Path to time series file. Should be "csv" format.
    load_params : Dict, optional
        Additional parameters passed to pd.read_csv function loading dataset.
        By default {}.
    drop_refill_pipeline : List[Union[Callable, Tuple, List]], optional
        Pipeline removing unwanted data and refilling missing values.
    preprocessing_pipeline : List[Union[Callable, Tuple, List]], optional
        Pipeline preprocessing passed time series. Follows after
        *drop_refill_pipeline*.
    scaler: TransformerMixin, optional
        Scaler transforming data.
    training_proportion: float, optional
        Value from 0 to 1 defining fraction of training data
        for fitting scaler.
    verbose : bool, optional
        If True, show progress bar. By default False.

    Returns
    -------
    pd.DataFrame
        Preprocessed data from file.
    """
    df = pd.read_csv(dataset_path, **load_params)

    pipeline = drop_refill_pipeline + preprocessing_pipeline
    if verbose:
        iterator = tqdm(enumerate(pipeline))
    else:
        iterator = iter(enumerate(pipeline))

    for i, step in iterator:
        func, args, kwargs = _read_pipeline_step(step, i)
        if verbose:
            iterator.set_description(
                f"Preprocessing step: {func.__name__}")
        try:
            df = func(df, *args, **kwargs)
        except Exception as e:
            e.args += (f"Error occured in {i} pipeline step "
                       f"with function \"{func.__name__}\".",)
            raise e

    if scaler is not None:
        assert training_proportion is not None,\
            "Training proportion not defined."
        df = scale(df, training_proportion, scaler)

    df = find_anomalies(
        df=df, detect_anomalies_pipeline=detect_anomalies_pipeline,
        verbose=verbose)

    return df


def find_anomalies(
    df: pd.DataFrame,
    detect_anomalies_pipeline: List[Union[Callable, Tuple, List]],
    verbose: bool = False,
):
    """This pipeline is runned in different way than others.
    All functions are executed separetly, not in sequence.
    At the end dataframe is filtered by logical sum of all filters.

    Functions finds anomalies based on single time series,
    without including relations of them.
    """
    if len(detect_anomalies_pipeline) == 0:
        return df
    else:
        if verbose:
            iterator = tqdm(enumerate(detect_anomalies_pipeline))
        else:
            iterator = iter(enumerate(detect_anomalies_pipeline))

        final_filter = df.apply(lambda x: False, axis=1)
        for i, step in iterator:
            func, args, kwargs = _read_pipeline_step(step, i)
            if verbose:
                iterator.set_description(
                    f"Preprocessing step: {func.__name__}")
            try:
                filter = func(df, *args, **kwargs)
                final_filter = final_filter | filter
            except Exception as e:
                e.args += (f"Error occured in {i} pipeline step "
                           f"with function \"{func.__name__}\".",)
                raise e
        return df[final_filter]


def _read_pipeline_step(
    step: Union[Callable, Tuple, List],
    step_idx: int
) -> Tuple[Callable, Tuple, Dict]:
    """Validates and converts pipeline element to function, *args
    and **kwargs.\n

    Raises sexceptions if element doesn't follow the rules:
    * element has to be a callable, tuple or list of preprocessing
    functions and additional arguments,
    * first parameter has to be a function,
    * you can pass *args or **kwargs or both of them to transforming
    function, just remember to pass it in this order.

    Parameters
    ----------
    step : Union[Callable, Tuple, List]
        Element of pipeline.
    step_idx : int
        Index of step in pipeline.

    Returns
    -------
    Tuple[Callable, Tuple, Dict]
        Preprocessing function, *args and **kwargs.
    """
    func, args, kwargs = None, (), {}

    if callable(step):
        func = step
    elif isinstance(step, (tuple, list)):
        assert len(step) > 0, f"Empty {step_idx} pipeline step."

        func = _get_func(step, step_idx)
        args, kwargs = _get_args_kwargs(step)
    else:
        raise ValueError(
            f"Wrong type of {step_idx} pipeline step. "
            f"Expected callable, tuple or list, got {type(step)}.")

    return func, args, kwargs


def _get_args_kwargs(step: Union[Tuple, List]) -> Tuple[Tuple, Dict]:
    """Assing *args and **kwargs based on pipeline element parameters.\n

    If last parameter is a dict, will be treated as **kwargs.
    Parameters
    ----------
    step : Union[Tuple, List]
        Pipeline step consisting function and optional arguments.

    Returns
    -------
    Tuple[Tuple, Dict]
        *args and *kwargs of pipeline step function.
    """
    args, kwargs = (), {}
    if isinstance(step[-1], dict):
        kwargs = step[-1]
        args = tuple(step[1:-1])
    else:
        args = step[1:]
    return args, kwargs


def _get_func(
    step: Union[Callable, Tuple, List],
    step_idx: int
) -> Callable:
    """Checks if first parameter of pipeline element is callable.
    If is, returns it.

    Parameters
    ----------
    step : Union[Callable, Tuple, List]
        Element of pipeline.
    step_idx : int
        Index of step in pipeline.

    Returns
    -------
    Callable
        Preprocessing function.
    """
    assert callable(step[0]),\
        f"Pipeline first argument of {step_idx} step is not a function."
    return step[0]


def fit_scaler(
    time_series: pd.DataFrame,
    training_fraction: float,
    scaler: TransformerMixin = MinMaxScaler()
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

    scaler.fit(time_series[:margin])


def scale(
    time_series: pd.DataFrame,
    training_fraction: float,
    scaler: TransformerMixin = MinMaxScaler()
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, TransformerMixin]]:
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
    # scaler = fit_scaler(time_series, training_fraction, scaler)
    fit_scaler(time_series, training_fraction, scaler)
    df = pd.DataFrame(
        scaler.transform(time_series),
        index=time_series.index,
        columns=time_series.columns
    )
    return df

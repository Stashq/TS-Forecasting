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


def load_and_preprocess(
    dataset_path: str,
    load_params: Dict = {},
    drop_refill_pipeline: List[Union[Callable, Tuple, List]] = None,
    preprocessing_pipeline: List[Union[Callable, Tuple, List]] = None,
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

    Function use two pipelines:
        * drop_refill_pipeline - functions dropping unwanted data and refilling
        missing values; runs first,
        * preprocessing_pipeline - functions preprocessing cleaned time series;
        runs second.
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
        df = scale(df, training_proportion, scaler, return_scaler=True)
    return df


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


def _params_for_len_2(
    step: Tuple,
    step_idx: int
) -> Tuple[Tuple, Dict]:
    """Reads and validate parameters from pipeline element that length
    is equal 2.\n

    Checks if second argument is tuple or dict. If not, ValueError is raised.
    If yes, set *args or **kwargs  depending on what is the type of second
    parameter.
    One of them is empty, but to keep the convention and, both are returned.

    Parameters
    ----------
    step : Union[Callable, Tuple, List]
        Element of pipeline.
    step_idx : int
        Index of step in pipeline.

    Returns
    -------
    Tuple[Tuple, Dict]
        *Args and  *kwargs for preprocessing function.
    """
    args, kwargs = (), {}
    if isinstance(step[1], tuple):
        args = step[1]
    elif isinstance(step[1], dict):
        kwargs = step[1]
    else:
        raise ValueError(
            f"Wrong second argument type of {step_idx} pipeline steps. "
            f"Expected tuple or dict, got {type(step[1])}.")
    return args, kwargs


def _params_for_len_3(
    step: Tuple,
    step_idx: int
) -> Tuple[Tuple, Dict]:
    """Reads and validate parameters from pipeline element that length
    is equal 3.\n

    Checks if second argument is a tuple and third is a dict.
    If yes, returns them. If not, raises AssertionError.

    Parameters
    ----------
    step : Union[Callable, Tuple, List]
        Element of pipeline.
    step_idx : int
        Index of step in pipeline.

    Returns
    -------
    Tuple[Tuple, Dict]
        *Args and  *kwargs for preprocessing function.
    """
    assert isinstance(step[1], tuple),\
        f"Second argument of {step_idx} pipeline step "\
        "is not a tuple."
    assert isinstance(step[2], dict),\
        f"Third argument of {step_idx} pipeline step "\
        "is not a dictionary."
    return step


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

    # traning scaler on training data
    scaler.fit(time_series[:margin])
    # return scaler.fit(time_series[:margin])


def scale(
    time_series: pd.DataFrame,
    training_fraction: float,
    scaler: TransformerMixin = MinMaxScaler(),
    return_scaler: bool = False
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
    # if return_scaler:
    #     return df, scaler
    # else:
    return df

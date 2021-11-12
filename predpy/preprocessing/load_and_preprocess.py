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


def load_and_preprocess(
    dataset_path: str,
    pipeline: List[Union[Callable, Tuple, List]],
    load_params: Dict = {},
    verbose: bool = False
) -> pd.DataFrame:
    """Loads data and preprocesses it with functions provided in in pipeline.\n

    Pipeline elements are tuple containing function and additional arguments.
    Functions should take transforming pandas dataframe as first argument,
    which will be passed between them automaticly.
    Rest of params should be passed in dictionaries sticking to the arguments
    names.

    Parameters
    ----------
    dataset_path : str
        Path to time series file. Should be "csv" format.
    pipeline : List[Union[Callable, Tuple, List]]
        List of preprocessing comands. Pipeline has to be created
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
    load_params : Dict, optional
        Additional parameters passed to pd.read_csv function loading dataset.
        By default {}.
    verbose : bool, optional
        If True, show progress bar. By default False.

    Returns
    -------
    pd.DataFrame
        Preprocessed data from file.
    """
    df = pd.read_csv(dataset_path, **load_params)
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


# def _read_pipeline_step(
#     step: Union[Callable, Tuple, List],
#     step_idx: int
# ) -> Tuple[Callable, Tuple, Dict]:
#     """Validates and converts pipeline element to function, *args
#     and **kwargs.\n

#     Raises sexceptions if element doesn't follow the rules:
#     * element has to be a callable, tuple or list of preprocessing
#     functions and additional arguments,
#     * first parameter has to be a function,
#     * you can pass *args or **kwargs or both of them to transforming
#     function, just remember to pass it in this order.

#     Parameters
#     ----------
#     step : Union[Callable, Tuple, List]
#         Element of pipeline.
#     step_idx : int
#         Index of step in pipeline.

#     Returns
#     -------
#     Tuple[Callable, Tuple, Dict]
#         Preprocessing function, *args and **kwargs.
#     """
#     func, args, kwargs = None, (), {}

#     if callable(step):
#         func = step
#     elif isinstance(step, (tuple, list)):
#         assert len(step) > 0, f"Empty {step_idx} pipeline step."

#         func = _get_func(step, step_idx)
#         if len(step) == 2:
#             args, kwargs = _params_for_len_2(step, step_idx)
#         elif len(step) == 3:
#             args, kwargs = _params_for_len_3(step, step_idx)
#         else:
#             raise ValueError(
#                 f"Too many arguments in pipeline {step_idx} step.")
#     else:
#         raise ValueError(
#             f"Wrong type of {step_idx} pipeline step. "
#             f"Expected callable, tuple or list, got {type(step)}.")

#     return func, args, kwargs


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

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Callable, Union, List
from ipywidgets import interact
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.auto import tqdm
from sklearn.ensemble import IsolationForest


def _to_series(
    ts: Union[pd.Series, pd.DataFrame],
    target: str = None
):
    if isinstance(ts, pd.Series):
        return ts
    elif isinstance(target, str) and isinstance(ts, pd.DataFrame):
        return ts[target]
    elif target is not None and not isinstance(target, str):
        raise ValueError(f"Target should be \"str\", not {type(target)}")
    elif target is None and isinstance(ts, pd.DataFrame):
        raise ValueError("Can not pass dataframe without passing target.")
    else:
        raise ValueError("ts type or target type not allowed.")


def dtw_distance(s, t):
    n, m = len(s), len(t)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(n+1):
        for j in range(m+1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s[i-1] - t[j-1])
            # take last min from a square box
            last_min = np.min(
                [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix[n, m]


def ts_distance_matrix(
    ts: Union[pd.Series, pd.DataFrame],
    window_size: int,
    target: str = None,
    verbose: bool = False
) -> np.ndarray:
    ts = _to_series(ts, target)
    n_rec = ts.shape[0] - window_size + 1
    dist_matrix = np.zeros((n_rec, n_rec))
    if verbose:
        for i in tqdm(range(1, n_rec)):
            for j in tqdm(range(i)):
                dist_matrix[i, j] = dtw_distance(
                    ts[i:i+window_size], ts[j:j+window_size])
    else:
        for i in range(1, n_rec):
            for j in range(i):
                dist_matrix[i, j] = dtw_distance(
                    ts[i:i+window_size], ts[j:j+window_size])
    return dist_matrix


def collective_isolation_forest(
    ts: Union[pd.Series, pd.DataFrame],
    window_size: int,
    target: str = None,
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, IsolationForest]]:
    ts = _to_series(ts, target)
    X = []
    ts.rolling(window_size).apply(lambda x: X.append(x.values) or 0)
    X = np.array(X)
    model = IsolationForest(random_state=0).fit(X)
    scores = model.score_samples(X)
    if return_model:
        return scores, model
    return scores


def get_isoforest_filter(
    ts: Union[pd.Series, pd.DataFrame],
    scores_threshold: float,
    window_size: int = None,
    target: str = None,
    scores: np.ndarray = None
):
    ts = _to_series(ts, target)
    if scores is None:
        scores = collective_isolation_forest(ts, window_size=window_size)
    elif window_size is None and scores is None:
        raise ValueError("Window size and scores cannot be both None.")
    result = (scores <= scores_threshold)

    diff = ts.shape[0] - len(result)
    result = np.concatenate([
        np.array([True]*diff), result])
    return result


def get_residuals(
    ts: Union[pd.Series, pd.DataFrame],
    model: str = "additive",
    period: int = None,
    target: str = None,
    **seasonal_decompose_kwargs
):
    ts = _to_series(ts, target)
    decomposed_ts = seasonal_decompose(
        ts, model=model, period=period, **seasonal_decompose_kwargs)
    return decomposed_ts.resid


def get_residual_filter(
    ts: Union[pd.Series, pd.DataFrame],
    residual_threshold: float,
    residuals: np.ndarray = None,
    target: str = None,
    **seasonal_decompose_kwargs
):
    ts = _to_series(ts, target)
    if residuals is None:
        residuals = get_residuals(ts, **seasonal_decompose_kwargs)
    return (residuals <= residual_threshold)


def get_dataframe_filter(
    df: pd.DataFrame,
    threshold: float,
    target: Union[str, List[str]],
    filter_fun: Callable,
    **filter_fun_kwargs
):
    result = None
    if isinstance(target, str):
        result = filter_fun(df[target], threshold, **filter_fun_kwargs)
    else:
        result = filter_fun(df[target[0]], threshold, **filter_fun_kwargs)
        for t in target[1:]:
            result = result | filter_fun(df[t], threshold, **filter_fun_kwargs)
    return result


def get_variance_filter(
    ts: Union[pd.Series, pd.DataFrame],
    window_size: int,
    log_variance_limits: Tuple[float],
    target: str = None
):
    ts = _to_series(ts, target)
    var = ts.rolling(window=window_size, center=True).var()
    lower = np.exp(log_variance_limits[0])
    upper = np.exp(log_variance_limits[1])
    result = (var >= lower) & (var <= upper)

    result[var.isna()] = True
    return result


def plot_interact_filtering(
    ts: Union[pd.Series, pd.DataFrame],
    widgets: Dict,
    filter_f: Callable,
    title: str = "Filtering values",
    target: str = None,
    **f_kwargs
):
    ts = _to_series(ts, target)
    fig, axes = plt.subplots(2, figsize=(9, 6))
    fig.suptitle(title)

    filter_ = filter_f(
        ts, *[w.value for name, w in widgets.items()], **f_kwargs)
    axes[0].plot(ts.index, ts)
    line2, = axes[0].plot(
        ts[~filter_].index,
        ts[~filter_],
        color='red', marker='o', linestyle='dashed',
        linewidth=0, markersize=3)
    line3, = axes[1].plot(
        ts[filter_].index,
        ts[filter_])

    axes[0].set_title("Time series and unwanted data")
    axes[1].set_title("Cleared time series")
    axes[1].set_xlabel(f"{ts[filter_].shape[0]} points")
    axes[1].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_ylim())

    def update(**widgets):
        filter_ = filter_f(ts, **widgets, **f_kwargs)
        line2.set_data(
            ts[~filter_].index,
            ts[~filter_])
        line3.set_data(
            ts[filter_].index,
            ts[filter_])
        axes[1].set_xlabel(f"{ts[filter_].shape[0]} points")
        fig.canvas.draw_idle()

    interact(update, **widgets)

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Callable, Union
from ipywidgets import interact
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from tqdm.auto import tqdm
from sklearn.ensemble import IsolationForest


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


def ts_distance_matrix(ts: pd.Series, window_size: int, verbose: bool = False):
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
    ts: pd.Series,
    window_size: int,
    return_model: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, IsolationForest]]:
    X = []
    ts.rolling(window_size).apply(lambda x: X.append(x.values) or 0)
    X = np.array(X)
    model = IsolationForest(random_state=0).fit(X)
    scores = model.score_samples(X)
    if return_model:
        return scores, model
    return scores


def get_residuals(
    ts=pd.Series, model="additive", period=None, **sd_kwargs
):
    decomposed_ts = seasonal_decompose(
        ts, model=model, period=period, **sd_kwargs)
    return decomposed_ts.resid


def get_residual_filter(
    df: pd.DataFrame,
    residual_threshold: float,
    residuals: pd.Series,
):
    return (residuals <= residual_threshold)


def get_variance_filter(
    df: pd.DataFrame,
    window_size: int,
    log_variance_limits: Tuple[float]
):
    var = df.rolling(window=window_size, center=True).var()
    var = var.dropna()
    lower = np.exp(log_variance_limits[0])
    upper = np.exp(log_variance_limits[1])
    result = (var >= lower) & (var <= upper)
    return result


def get_isoforest_filter(
    df: pd.DataFrame,
    scores_threshold: float,
    scores: pd.Series,
):
    result = (scores <= scores_threshold)
    if df.shape[0] > len(result):
        diff = df.shape[0] - len(result)
        result = np.concatenate([
            np.array([True]*diff), result])
    return result


def plot_interact_filtering(
    df: pd.DataFrame,
    target: str,
    widgets: Dict,
    filter_f: Callable,
    title: str = "Filtering values",
    **f_kwargs
):
    fig, axes = plt.subplots(2, figsize=(12, 6))
    fig.suptitle(title)
    axes[0].set_title("Time series and unwanted data")
    axes[1].set_title("Cleared time series")
    axes[1].set_ylim(axes[0].get_ylim())

    filter_ = filter_f(
        df, *[w.value for name, w in widgets.items()], **f_kwargs)
    axes[0].plot(df.index, df[target])
    line2, = axes[0].plot(
        df[~filter_].index,
        df[~filter_][target],
        color='red', marker='o', linestyle='dashed',
        linewidth=0, markersize=3)
    line3, = axes[1].plot(
        df[filter_].index,
        df[filter_][target])
    axes[1].set_xlabel(f"{df[filter_].shape[0]} points")

    def update(**widgets):
        filter_ = filter_f(df, **widgets, **f_kwargs)
        line2.set_data(
            df[~filter_].index,
            df[~filter_][target])
        line3.set_data(
            df[filter_].index,
            df[filter_][target])
        axes[1].set_xlabel(f"{df[filter_].shape[0]} points")
        fig.canvas.draw_idle()

    interact(update, **widgets)

import os

import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
import math


os.chdir('/home/stachu/Projects/Anomaly_detection/TSAD')
sns.set_style()


def _adjust_subplot_params(
    n_cols: int, n_rows: int = None, figsize: Tuple[int] = None,
    features_cols: List[int] = None, n_features: int = None
):
    if features_cols is None:
        features_cols = list(range(n_features))
    if n_rows is not None:
        assert n_rows * n_cols >= len(features_cols)
    else:
        if len(features_cols) < n_cols:
            n_rows, n_cols = 1, len(features_cols)
        else:
            n_rows = math.ceil(len(features_cols)/2)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 2)
    return n_rows, n_cols, figsize, features_cols


def _select_ax(axs, n_rows, n_cols, i, title: str = None):
    ax_row = int(i / n_cols)
    ax_col = i % n_cols
    if n_rows > 1 and n_cols > 1:
        ax = axs[ax_row, ax_col]
    elif n_rows > 1 and n_cols == 1:
        ax = axs[ax_row]
    elif n_rows == 1 and n_cols > 1:
        ax = axs[ax_col]
    else:
        ax = axs
    if title is not None:
        ax.set_title(title)
    return ax


def plot_scores(
    scores: np.ndarray, features_cols: List[int] = None,
    n_rows: int = None, n_cols: int = 2,
    classes: List[Literal[0, 1]] = None,
    figsize=None
):
    n_rows, n_cols, figsize, features_cols = _adjust_subplot_params(
        n_cols, n_rows, figsize, features_cols, scores.shape[1])

    data = scores[:, features_cols]
    df = pd.DataFrame(
        data, columns=['score_%d' % i for i in features_cols])
    if classes is not None:
        df['classes'] = classes
        hue = 'classes'
    else:
        hue = None

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, col_id in enumerate(features_cols):
        ax = _select_ax(axs, n_rows, n_cols, i)
        sns.scatterplot(
            data=df,
            x=df.index,
            y='score_%d' % col_id,
            hue=hue,
            ax=ax
        )
    return fig


def plot_kde(
    scores1: np.ndarray,
    features_cols: List[int] = None,
    scores2: np.ndarray = None,
    n_rows: int = None, n_cols: int = 2,
    figsize=None, scores_names=['0', '1']
):
    n_rows, n_cols, figsize, features_cols = _adjust_subplot_params(
        n_cols, n_rows, figsize, features_cols, scores1.shape[1])

    # defining data and classes
    if scores2 is not None:
        data = np.concatenate(
            [scores1[:, features_cols], scores2[:, features_cols]])
        classes = [scores_names[0]] * len(scores1)\
            + [scores_names[1]] * len(scores2)
    else:
        data, classes = scores1[:, features_cols], None

    # saving data and classes to DataFrame
    df = pd.DataFrame(
        data, columns=['score_%d' % i for i in features_cols])
    if classes is not None:
        df['classes'] = classes
        hue = 'classes'
    else:
        hue = None

    # plotting selected features
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, col_id in enumerate(features_cols):
        ax = _select_ax(axs, n_rows, n_cols, i)
        sns.kdeplot(
            data=df,
            x='score_%d' % col_id,
            hue=hue,
            common_norm=True,
            ax=ax
        )
        ax.set_title('score_%d' % col_id)
    return fig


def plot_scores_and_bands(
    scores: np.ndarray,
    bounds: List[pd.DataFrame],
    features_cols: List[int] = None,
    n_rows: int = None, n_cols: int = 2,
    classes: List[Literal[0, 1]] = None,
    figsize=None
):
    n_rows, n_cols, figsize, features_cols = _adjust_subplot_params(
        n_cols, n_rows, figsize, features_cols, scores.shape[1])

    data = scores[:, features_cols]
    df = pd.DataFrame(
        data, columns=['score_%d' % i for i in features_cols])
    if classes is not None:
        df['classes'] = classes
        hue = 'classes'
    else:
        hue = None

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, col_id in enumerate(features_cols):
        ax = _select_ax(axs, n_rows, n_cols, i)
        sns.scatterplot(
            data=df,
            x=df.index,
            y='score_%d' % col_id,
            hue=hue,
            ax=ax
        )
        for b in bounds:
            ax.plot(b[col_id], label=b.name)
    return fig


def plot_dataset(
    ds: Dict[str, np.ndarray],
    anoms_vrects: List[Tuple[int]] = [],
    pred_anoms_vrects: List[Tuple[int]] = [],
    features_cols: List[int] = None,
    figsize: Tuple[int] = None,
    min_id: int = None, max_id: int = None,
    hlines: Dict[str, float] = {}
):
    n_points, n_features = list(ds.values())[0].shape[:2]
    is_df = type(list(ds.values())[0]) in [pd.DataFrame, pd.Series]
    n_rows, n_cols, figsize, features_cols = _adjust_subplot_params(
        2, None, figsize, features_cols, n_features)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i, col_id in enumerate(features_cols):
        ax = _select_ax(axs, n_rows, n_cols, i)
        for ds_name, ds_vals in ds.items():
            if is_df:
                ax.plot(ds_vals.iloc[min_id:max_id, col_id], label=ds_name)
            else:
                ax.plot(ds_vals[min_id:max_id, col_id], label=ds_name)
        for start, end in anoms_vrects:
            _add_vrect(
                ax, start, end, color='red', min_id=min_id, max_id=max_id)
        for start, end in pred_anoms_vrects:
            _add_vrect(
                ax, start, end, color='blue', min_id=min_id, max_id=max_id)
        for hl_name, hl_val in hlines.items():
            plt.hlines(
                y=hl_val, xmin=0, xmax=n_points, color='green',
                linestyles='-', lw=2, label=hl_name)
    return fig


def _add_vrect(
    ax, start: int, end: int, color: str = 'red',
    min_id: int = None, max_id: int = None
):
    if start > end:
        start, end = end, start
    if (min_id is not None and start < min_id and end < min_id)\
            or (max_id is not None and start >= max_id and end >= max_id):
        pass
    else:
        if min_id is not None and start < min_id:
            start = min_id
        if max_id is not None and end >= max_id:
            end = max_id
        ax.axvspan(start, end, alpha=0.2, color=color)

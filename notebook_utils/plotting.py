import os

import numpy as np
import pandas as pd
from typing import List, Literal, Tuple
# from predpy.plotter import plot_anomalies
import seaborn as sns
import matplotlib.pyplot as plt
import math


os.chdir('/home/stachu/Projects/Anomaly_detection/TSAD')
sns.set_style()


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
    if features_cols is None:
        features_cols = list(range(scores.shape[1]))
    if n_rows is not None:
        assert n_rows * n_cols >= len(features_cols)
    else:
        if len(features_cols) < n_cols:
            n_rows, n_cols = 1, len(features_cols)
        else:
            n_rows = math.ceil(len(features_cols)/2)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 2)

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
    if features_cols is None:
        features_cols = list(range(scores1.shape[1]))
    if n_rows is not None:
        assert n_rows * n_cols >= len(features_cols)
    else:
        if len(features_cols) < n_cols:
            n_rows, n_cols = 1, len(features_cols)
        else:
            n_rows = math.ceil(len(features_cols)/2)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 2)

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
    if features_cols is None:
        features_cols = list(range(scores.shape[1]))
    if n_rows is not None:
        assert n_rows * n_cols >= len(features_cols)
    else:
        if len(features_cols) < n_cols:
            n_rows, n_cols = 1, len(features_cols)
        else:
            n_rows = math.ceil(len(features_cols)/2)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 2)

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
    train_ds: np.ndarray, test_ds: np.ndarray,
    anoms_vrects: List[Tuple[int]]
):
    # train_np = train_dl.dataset.sequences[0].to_numpy()
    # test_np = test_dl.dataset.sequences[0].to_numpy()
    n_features = train_ds.shape[1]
    n_cols = 2
    n_rows = int(n_features/n_cols) + int(n_features % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 9, n_rows * 2))
    for i in range(n_features):
        col = i % n_cols
        row = int(i/n_cols)
        axs[row, col].plot(train_ds[:, i], label='train')
        axs[row, col].plot(test_ds[:, i], label='test')

        for start, end in anoms_vrects:
            axs[row, col].axvspan(start, end, alpha=0.2, color='red')
    return fig

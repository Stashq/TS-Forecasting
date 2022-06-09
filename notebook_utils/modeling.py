import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, fbeta_score
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import scipy

from predpy.wrapper import Reconstructor


nd = scipy.stats.norm(loc=0, scale=1)
pos_in_distribution = nd.ppf(0.95)

DEFAULT_WDD_W_F = nd.pdf(pos_in_distribution)
DEFAULT_WDD_MA_F = nd.pdf(pos_in_distribution)


def get_a_scores(model, dataloader) -> np.ndarray:
    a_scores = []
    for batch in tqdm(dataloader):
        x = batch['sequence']
        a_s = model.anomaly_score(
            x, scale=False)
        a_scores += [a_s]
    a_scores = np.concatenate(a_scores)
    if len(a_scores.shape) == 1:
        a_scores = a_scores.reshape(-1, 1)
    return a_scores


def get_model_a_scores_one_per_point(
    model, dataloader: DataLoader, ws: int,
    **a_scorer_kwargs
) -> np.ndarray:
    """Batch size must equal 1"""
    a_scores = []
    len_ = dataloader.dataset.sequences[0].shape[0]
    last_n = len_ % ws
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i % ws == 0:
                x = batch['sequence']
                a_s = model.anomaly_score(x, **a_scorer_kwargs)
                a_scores += [a_s[0]]
            if i == len(dataloader) - 1 and last_n > 0:
                x = batch['sequence']
                a_s = model.anomaly_score(x, **a_scorer_kwargs)
                a_scores += [a_s[0, -last_n:]]
    a_scores = np.concatenate(a_scores, axis=0)
    if len(a_scores.shape) == 1:
        a_scores = a_scores.reshape(-1, 1)
    return a_scores


def get_model_recon_one_per_point(
    model: Reconstructor, dataloader: DataLoader, ws: int
) -> List:
    """Batch size must equal 1"""
    recon = []
    len_ = dataloader.dataset.sequences[0].shape[0]
    last_n = len_ % ws
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i % ws == 0:
                x = batch['sequence']
                res = model.predict(x)
                recon += [res[0]]
            if i == len(dataloader) - 1 and last_n > 0:
                x = batch['sequence']
                res = model.predict(x)
                recon += [res[0, -last_n:]]
    recon = torch.concat(recon, dim=0).numpy()
    return recon


def predict(
    point_scores: np.ndarray, th: float, ws: int,
    return_point_cls: bool = True
) -> np.ndarray:
    pred_point_cls = np.zeros(len(point_scores))
    p = np.where(np.any(point_scores > th, axis=1))[0]
    pred_point_cls[p] = 1

    if return_point_cls:
        return pred_point_cls
    else:
        pred_rec_cls = adjust_point_cls_with_window(
            pred_point_cls, ws, return_point_cls=False)
        return pred_rec_cls


def adjust_point_cls_with_window(
    point_cls: np.ndarray, ws: int,
    return_point_cls: bool = True
) -> np.ndarray:
    """Translating point classes to record classes, then
    if "return_point_cls" is True, projecting classes back on points.

    Any record consisting anomaly point
    is treated as all points in sequence
    are anomalies."""
    s = pd.Series(point_cls)
    res = s.rolling(2*ws - 1, center=True).max()
    res[:ws] = res[:ws].index.to_series().apply(
        lambda idx: s[0:idx + ws].max()
    )
    if return_point_cls:
        res[-ws:] = res[-ws:].index.to_series().apply(
            lambda idx: s[idx - ws:].max()
        )
    else:
        res = res.dropna().iloc[1:]

    return res.to_numpy()


def get_rec_fbeta_score_conf_mat(
    point_scores: np.ndarray, point_cls: np.ndarray,
    th: float, ws: int, beta: float = 1
) -> Tuple[float, List[List[int]]]:
    pred_cls = predict(
        point_scores=point_scores, th=th, ws=ws)
    true_cls = pd.Series(point_cls).rolling(
        ws).max().dropna().to_numpy()

    f1_score = fbeta_score(true_cls, pred_cls, beta=beta)
    cm = confusion_matrix(true_cls, pred_cls)
    return f1_score, cm


def th_ws_experiment(
    series_index: pd.Index, point_scores: np.ndarray,
    point_cls: List[Literal[0, 1]], ths: List[float],
    wss: List[int], t_max: int = None, w_f: float = None,
    ma_f: float = None, betas: List[float] = [1.0]
) -> pd.DataFrame:
    if type(betas) in [int, float]:
        betas = [betas]
    threshold_stats = defaultdict(lambda: [])

    for ws in tqdm(wss):
        true_cls = adjust_point_cls_with_window(
            point_cls, ws, return_point_cls=False)
        for th in ths:
            exp_step(
                threshold_stats=threshold_stats, series_index=series_index,
                point_scores=point_scores, true_cls=true_cls,
                th=th, ws=ws, t_max=t_max, w_f=w_f, ma_f=ma_f,
                betas=betas
            )
    return pd.DataFrame(threshold_stats)


def stats_experiment(
    series_index: pd.Index, point_scores_list: List[np.ndarray],
    point_cls: List[Literal[0, 1]], ths_list: List[List[float]],
    ws_list: List[int], t_max: int = None, w_f: float = None,
    ma_f: float = None, betas: List[float] = [1.0],
) -> pd.DataFrame:
    if type(betas) in [int, float]:
        betas = [betas]
    threshold_stats = defaultdict(lambda: [])
    assert len(point_scores_list) == len(ths_list) == len(ws_list),\
        'Length of "point_scores" (%d), "ths" (%d), "wss" (%d) not same'\
        % (len(point_scores_list), len(ths_list), len(ws_list))

    n_stats = len(point_scores_list)
    for i in tqdm(range(n_stats)):
        point_scores = point_scores_list[i]
        ths = ths_list[i]
        ws = ws_list[i]
        true_cls = adjust_point_cls_with_window(
            point_cls, ws, return_point_cls=False)

        for th in ths:
            exp_step(
                threshold_stats=threshold_stats, series_index=series_index,
                point_scores=point_scores, true_cls=true_cls,
                th=th, ws=ws, t_max=t_max, w_f=w_f, ma_f=ma_f,
                betas=betas
            )
    return pd.DataFrame(threshold_stats)


def exp_step(
    threshold_stats: Dict, series_index: pd.Index,
    point_scores: np.ndarray, true_cls: np.ndarray,
    th: float, ws: int, t_max: int = None,
    w_f: float = None, ma_f: float = None,
    betas: List[float] = [1.0]
):
    pred_cls = predict(
        point_scores=point_scores, th=th, ws=ws,
        return_point_cls=False)

    threshold_stats['ws'] += [ws]
    threshold_stats['th'] += [th]

    if t_max is not None:
        wdd = calculate_rec_wdd(
            series_index[-len(true_cls):],
            pred_rec_cls=pred_cls, true_rec_cls=true_cls,
            t_max=t_max, w_f=w_f, ma_f=ma_f)
        threshold_stats['wdd'] += [wdd]
    else:
        threshold_stats['wdd'] += [None]
    for beta in betas:
        fb_s = fbeta_score(true_cls, pred_cls, beta=beta)
        threshold_stats[f'f{beta}-score'] += [fb_s]
    cm = confusion_matrix(true_cls, pred_cls)

    threshold_stats['tn'] += [cm[0, 0]]
    threshold_stats['fp'] += [cm[0, 1]]
    threshold_stats['fn'] += [cm[1, 0]]
    threshold_stats['tp'] += [cm[1, 1]]

    threshold_stats['preds_rec_cls'] += [pred_cls]
    return threshold_stats


def calculate_rec_wdd(
    series_index, pred_rec_cls: List[int],
    true_rec_cls: List[int], t_max: int,
    w_f: float = DEFAULT_WDD_W_F,
    ma_f: float = DEFAULT_WDD_MA_F
) -> float:
    """Calculate WDD score from article
    'Evaluation metrics for anomaly detection algorithms in time-series'
    based on gaussian distribution function.
    Compares distance between records in time series.

    Args:
        pred_rec_cls (List[int]): predicted records classes.
        true_rec_cls (List[int]): true records classes.
        t_max (int): maximum distance between paired
            predicted and true anomaly position.
        w_f (float): false anomaly detenction penalty.
        ma_f (float, optional): missed anomaly penalty. Defaults to 0.

    Returns:
        float: wdd score.
    """
    if w_f is None:
        w_f = DEFAULT_WDD_W_F
    if ma_f is None:
        ma_f = DEFAULT_WDD_MA_F

    cls_df = pd.DataFrame(zip(
        true_rec_cls, pred_rec_cls
    ), index=series_index, columns=['true_cls', 'pred_cls'])

    def score(row):
        # calculate w
        if row['true_cls'] == 1:
            idx = row.name
            frame = cls_df.loc[idx-t_max:idx+t_max]
            preds = frame[frame['pred_cls'] == 1]
            # missed anomaly penalty
            if preds.shape[0] == 0:
                return -ma_f
            else:
                diff = abs(preds.index - idx).min()
                return nd.pdf(diff/t_max)
        # find FA
        elif row['pred_cls'] == 1:
            idx = row.name
            frame = cls_df.loc[idx-t_max:idx+t_max]
            preds = frame[frame['true_cls'] == 1]
            # detected anomaly (DA)
            if preds.shape[0] > 0:
                return 0
            # false anomaly penalty
            else:
                return -w_f
        else:
            return 0
    scores = cls_df.apply(score, axis=1)
    wdd = scores.sum()

    return wdd


def recalculate_wdd(th_df, s_index, point_cls, t_max):
    for ws in th_df['ws'].unique():
        true_cls = adjust_point_cls_with_window(
            point_cls, ws, return_point_cls=False)
        ws_th_index = th_df[th_df['ws'] == ws].index
        ws_th_df = th_df.loc[ws_th_index]
        th_df.loc[ws_th_index, 'wdd'] = ws_th_df.apply(
            lambda x: calculate_rec_wdd(
                s_index[-len(true_cls):], x['preds_rec_cls'],
                true_cls, t_max=t_max),
            axis=1
        )

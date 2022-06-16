import numpy as np
import pandas as pd
from typing import Dict, List, Literal, Tuple, Union
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, fbeta_score
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import scipy

from predpy.wrapper import Reconstructor
from predpy.experimentator import Experimentator


nd = scipy.stats.norm(loc=0, scale=1)
pos_in_distribution = nd.ppf(0.95)

DEFAULT_WDD_W_F = nd.pdf(pos_in_distribution)
DEFAULT_WDD_MA_F = nd.pdf(pos_in_distribution)


def get_a_scores(
    model, dataloader, scale: bool = False,
    use_tqdm: bool = True, **a_scorer_kwargs
) -> np.ndarray:
    a_scores = []
    if use_tqdm:
        iterator = tqdm(dataloader)
    else:
        iterator = dataloader
    for batch in iterator:
        x = batch['sequence']
        a_s = model.anomaly_score(
            x, scale=scale, **a_scorer_kwargs)
        a_scores += [a_s]
    a_scores = np.concatenate(a_scores)
    if len(a_scores.shape) == 1:
        a_scores = a_scores.reshape(-1, 1)
    return a_scores


def get_a_scores_one_per_point(
    model, dataloader: DataLoader, ws: int,
    **a_scorer_kwargs
) -> np.ndarray:
    """Batch size must equal 1"""
    a_scores = []
    len_ = dataloader.dataset.n_points
    last_n = len_ % ws
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i % ws == 0:
                x = batch['sequence']
                a_s = model.anomaly_score(x, **a_scorer_kwargs)
                a_scores += [a_s[0]]
            elif (i == len(dataloader) - 1):
                x = batch['sequence']
                a_s = model.anomaly_score(x, **a_scorer_kwargs)
                a_scores += [a_s[0][-last_n:]]
    a_scores = np.concatenate(a_scores, axis=0)
    if len(a_scores.shape) == 1:
        a_scores = a_scores.reshape(-1, 1)
    return a_scores


def get_recon_one_per_point(
    model: Reconstructor, dataloader: DataLoader, ws: int
) -> List:
    """Batch size must equal 1"""
    recon = []
    len_ = dataloader.dataset.n_points
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
    res[:ws-1] = res[:ws-1].index.to_series().apply(
        lambda idx: s[0:idx + ws].max()
    )
    if return_point_cls:
        res[-ws+1:] = res[-ws+1:].index.to_series().apply(
            lambda idx: s[idx - ws:].max()
        )
    else:
        res = res.dropna()

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
    series_index: pd.Index, scores_list: List[np.ndarray],
    point_cls: List[Literal[0, 1]], ths_list: List[List[float]],
    ws_list: List[int], model_ws: int = None,
    t_max: int = None, w_f: float = None, ma_f: float = None,
    betas: List[float] = [1.0],
) -> pd.DataFrame:
    # assert scores_are_points and model_ws is not None,\
    #     '"model_ws" cannot be None if "scores_are_points" is True.'
    if type(betas) in [int, float]:
        betas = [betas]
    threshold_stats = defaultdict(lambda: [])
    assert len(scores_list) == len(ths_list) == len(ws_list),\
        'Length of "point_scores" (%d), "ths" (%d), "wss" (%d) not same'\
        % (len(scores_list), len(ths_list), len(ws_list))

    if model_ws is None:
        model_ws = 0
    else:
        model_ws -= 1
    n_stats = len(scores_list)
    for i in tqdm(range(n_stats)):
        point_scores = scores_list[i]
        ths = ths_list[i]
        ws = ws_list[i]
        true_cls = adjust_point_cls_with_window(
            point_cls, ws + model_ws, return_point_cls=False)

        for th in ths:
            exp_step(
                threshold_stats=threshold_stats, series_index=series_index,
                scores=point_scores, true_cls=true_cls,
                th=th, ws=ws, t_max=t_max, w_f=w_f, ma_f=ma_f,
                betas=betas
            )
    return pd.DataFrame(threshold_stats)


def exp_step(
    threshold_stats: Dict, series_index: pd.Index,
    scores: np.ndarray, true_cls: np.ndarray,
    th: float, ws: int,
    t_max: int = None, w_f: float = None, ma_f: float = None,
    betas: List[float] = [1.0]
):
    pred_cls = predict(
        point_scores=scores, th=th, ws=ws,
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
    cm = confusion_matrix(true_cls, pred_cls, labels=[0, 1])

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


def a_score_exp(
    train_dl: DataLoader, test_dl: DataLoader,
    exp: Experimentator, scale: bool,
    ds_id: int = 0, models_ids: List = None,
    calculate_training_a_score: bool = True
) -> Dict:
    res = {}
    if models_ids is None:
        models_ids = exp.models_params.index
    else:
        assert all(item in exp.models_params.index for item in models_ids)
    for m_id in models_ids:
        m_name = exp.models_params.loc[m_id]['name_']
        ds_name = exp.datasets_params.loc[ds_id]['name_']
        model = exp.load_pl_model(m_id, f'checkpoints/{ds_name}/{m_name}')

        train_a_scores = np.zeros((1, 1))
        if calculate_training_a_score:
            if scale:
                train_a_scores = model.fit_scores_scaler(
                    train_dl, use_tqdm=True)
            else:
                train_a_scores = get_a_scores(
                    model=model, dataloader=train_dl, use_tqdm=True)

        test_a_scores = get_a_scores(
            model, test_dl, scale=scale, use_tqdm=True)
        res[m_name] = [train_a_scores, test_a_scores]
    return res


def exctract_a_scores(
    exps_a_scores: Union[Dict, List[Dict]],
    train_len: int = None, test_len: int = None
):
    if not (isinstance(exps_a_scores, List) or
            isinstance(exps_a_scores, Tuple)):
        exps_a_scores = [exps_a_scores]

    train_a_scores = {}
    test_a_scores = {}
    for e_a_s in exps_a_scores:
        for m_name, scores in e_a_s.items():
            train_a_s = np.copy(scores[0])
            test_a_s = np.copy(scores[1])
            if train_len is not None and len(train_a_s) < train_len:
                n_padding = train_len - len(train_a_s)
                train_a_s = np.concatenate(
                    [np.zeros((n_padding, 1)), train_a_s])
            if test_len is not None and len(test_a_s) < test_len:
                # old = scores[1]
                n_padding = test_len - len(test_a_s)
                test_a_s = np.concatenate(
                    [np.zeros((n_padding, 1)), test_a_s])
            train_a_scores[m_name] = train_a_s
            test_a_scores[m_name] = test_a_s
    return train_a_scores, test_a_scores

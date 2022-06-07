import numpy as np
import pandas as pd
from typing import List, Literal, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, fbeta_score
from collections import defaultdict
import torch
from torch.utils.data import DataLoader

from predpy.wrapper import Reconstructor


def get_a_scores(model, dataloader):
    a_scores = []
    for batch in tqdm(dataloader):
        x = batch['sequence']
        a_score = model.anomaly_score(
            x, scale=False)
        a_scores += [a_score]
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
    return_rec_cls: bool = True
) -> np.ndarray:
    pred_point_cls = np.zeros(len(point_scores))
    p = np.where(np.any(point_scores > th, axis=1))[0]
    pred_point_cls[p] = 1

    if not return_rec_cls:
        return pred_point_cls
    # pred_cls = pd.Series(pred_point_cls).rolling(
    #     ws).max().dropna().to_numpy()
    pred_cls = adjust_point_cls_with_window(pred_point_cls, ws)
    return pred_cls


def adjust_point_cls_with_window(point_cls: np.ndarray, ws: int):
    s = pd.Series(point_cls)
    res = s.rolling(2*ws + 1, center=True).max()
    res[:ws] = res[:ws].index.to_series().apply(
        lambda idx: s[0:idx + ws].max()
    )
    res[-ws:] = res[-ws:].index.to_series().apply(
        lambda idx: s[idx - ws:].max()
    )
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
    point_scores: np.ndarray, point_cls: List[Literal[0, 1]],
    ths: List[float], wss: List[int], betas: List[float] = [1.0]
) -> pd.DataFrame:
    if type(betas) in [int, float]:
        betas = [betas]
    threshold_stats = defaultdict(lambda: [])

    for ws in tqdm(wss):
        # true_cls = pd.Series(point_cls).rolling(
        #     ws).max().dropna().to_numpy()
        true_cls = adjust_point_cls_with_window(
            point_cls, ws)
        for th in ths:
            pred_cls = predict(
                point_scores=point_scores, th=th, ws=ws)

            for beta in betas:
                fb_s = fbeta_score(true_cls, pred_cls, beta=beta)
                threshold_stats[f'f{beta}-score'] += [fb_s]
            cm = confusion_matrix(true_cls, pred_cls)
            threshold_stats['ws'] += [ws]
            threshold_stats['th'] += [th]
            threshold_stats['cm'] += [cm]
    return pd.DataFrame(threshold_stats)

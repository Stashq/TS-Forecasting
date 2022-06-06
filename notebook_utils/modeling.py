import numpy as np
import pandas as pd
from typing import List, Literal, Tuple
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, fbeta_score
from collections import defaultdict


def get_a_scores(model, dataloader):
    a_scores = []
    for batch in tqdm(dataloader):
        x = batch['sequence']
        a_score = model.anomaly_score(
            x, scale=False)
        a_scores += [a_score]
    return a_scores


def predict(
    point_scores: np.ndarray, th: float, ws: int,
    return_rec_cls: bool = True
) -> np.ndarray:
    pred_point_cls = np.zeros(len(point_scores))
    p = np.where(np.any(point_scores > th, axis=1))[0]
    pred_point_cls[p] = 1

    if not return_rec_cls:
        return pred_point_cls
    pred_cls = pd.Series(pred_point_cls).rolling(
        ws).max().dropna().to_numpy()
    return pred_cls


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
    ths: List[float], wss: List[int], beta: float = 1.0
) -> pd.DataFrame:
    threshold_stats = defaultdict(lambda: [])
    for ws in tqdm(wss):
        true_cls = pd.Series(point_cls).rolling(
            ws).max().dropna().to_numpy()
        for th in ths:
            pred_cls = predict(
                point_scores=point_scores, th=th, ws=ws)

            fb_s = fbeta_score(true_cls, pred_cls, beta=beta)
            cm = confusion_matrix(true_cls, pred_cls)

            threshold_stats['ws'] += [ws]
            threshold_stats['th'] += [th]
            threshold_stats[f'f{beta}-score'] += [fb_s]
            threshold_stats['cm'] += [cm]
    return pd.DataFrame(threshold_stats)

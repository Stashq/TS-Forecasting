# flake8: noqa

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torch.utils.data import DataLoader
import torch
from typing import List, Literal, Dict, Union, Tuple
from pathlib import Path
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score
import re

from predpy.dataset import MultiTimeSeriesDataset
from predpy.experimentator import (
    DatasetParams, ModelParams,
    Experimentator, load_experimentator, load_last_experimentator)
from predpy.plotter import plot_anomalies, get_ids_ranges, get_cls_ids_ranges
from anomaly_detection import (
    AnomalyDetector, fit_run_detection, exp_fit_run_detection,
    get_dataset, get_dataset_names, load_anom_scores, MovingStdAD)

from notebook_utils.modeling import (
    predict, get_a_scores, get_rec_fbeta_score_conf_mat,
    get_a_scores_one_per_point, get_recon_one_per_point,
    adjust_point_cls_with_window,
    th_ws_experiment, stats_experiment,
    calculate_rec_wdd, recalculate_wdd
)
from notebook_utils.plotting import (
    plot_scores, plot_kde, plot_dataset, plot_scores_and_bands
)
from notebook_utils.save_load import (
    save_th_exp, load_th_exp
)
from notebook_utils.ts_stats import (
    get_bollinger, get_std, get_diff
)

def a_score_exp(
    train_dl: DataLoader, test_dl: DataLoader,
    exp: Experimentator, scale: bool, models_ids: List = None
) -> Dict:
    res = {}
    if models_ids is None:
        models_ids = exp.models_params.index
    else:
        assert all(item in exp.models_params.index for item in models_ids)
    for m_id in tqdm(models_ids):
        m_name = exp.models_params.loc[m_id]['name_']
        model = exp.load_pl_model(m_id, f'checkpoints/{ds_name}/{m_name}')
        if scale:
            train_a_scores = model.fit_scores_scaler(
                train_dl, use_tqdm=False)
        else:
            train_a_scores = get_a_scores(
                train_dl, use_tqdm=False)

        test_a_scores = get_a_scores(
            model, test_dl, scale=scale, use_tqdm=False)
        res[m_name] = [train_a_scores, test_a_scores]
    return res

def exctract_a_scores(
    exps_a_scores: Union[Dict, List[Dict]],
    train_len: int = None, test_len: int = None):
    if not (isinstance(exps_a_scores, List) or isinstance(exps_a_scores, Tuple)):
        exps_a_scores = [exps_a_scores]
    
    train_a_scores = {}
    test_a_scores = {}
    for e_a_s in exps_a_scores:
        for m_name, scores in e_a_s.items():
            if train_len is not None and len(scores[0]) < train_len:
                n_padding = train_len - len(scores[0])
                scores[0] = np.concatenate([np.zeros((n_padding, 1)), scores[0]])
            if test_len is not None and len(scores[1]) < test_len:
                # old = scores[1]
                n_padding = test_len - len(scores[1])
                scores[1] = np.concatenate([np.zeros((n_padding, 1)), scores[1]])
            print(test_len, len(scores[1]))
            train_a_scores[m_name] = scores[0]
            test_a_scores[m_name] = scores[1]
    return train_a_scores, test_a_scores



topic, colleciton_name, ds_name = 'Handmade', 'Sin', 'artificial_1'
window_size = 200
# scores_dirpath = f'notebook_a_scores/{colleciton_name}/{ds_name}/{m_name}/{exp_date}/'


train_ds = get_dataset(
    f'data/{topic}/{colleciton_name}/train/{ds_name}.csv',
    window_size=window_size)
test_ds = get_dataset(
    f'data/{topic}/{colleciton_name}/test/{ds_name}.csv',
    window_size=window_size)

train_dl = DataLoader(train_ds, batch_size=500)
test_dl = DataLoader(test_ds, batch_size=500)
test_index = test_ds.sequences[0].index

test_point_cls_path = f'data/{topic}/{colleciton_name}/test_label/{ds_name}.csv'
test_point_cls = pd.read_csv(
    test_point_cls_path, header=None)\
    .iloc[:, 0].to_numpy()

n_features = train_ds.n_points

exp2 = load_experimentator('./saved_experiments/2022-06-12_01:00:49.pkl')
exp2.models_params['name_']

exp_a_scores = a_score_exp(
    train_dl, test_dl, exp2, scale=True, models_ids=[1])


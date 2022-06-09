# flake8: noqa

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from torch.utils.data import DataLoader
import torch
from typing import List, Literal
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
    get_dataset, get_dataset_names, load_anom_scores)
from notebook_utils.modeling import (
    predict, get_rec_fbeta_score_conf_mat,
    get_a_scores, get_model_recon_one_per_point,
    adjust_point_cls_with_window,
    th_ws_experiment, stats_experiment
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


# os.chdir('/home/stachu/Projects/Anomaly_detection/TSAD')
sns.set_style()

# Loading
exp = load_experimentator('./saved_experiments/2022-05-30_22:57:20.pkl')
model_id = 0


m_name = 'ConvMVR_ws200_nk10_ks3_es50'
topic, colleciton_name, ds_name = 'Industry', 'ServerMachineDataset', 'machine-1-1'
exp_date = exp.exp_date
model = exp.load_pl_model(model_id, f'checkpoints/machine-1-1/{m_name}')
window_size = model.model.params['window_size']
scores_dirpath = f'notebook_a_scores/{colleciton_name}/{ds_name}/{m_name}/{exp_date}/'


train_ds = get_dataset(
    f'data/{topic}/{colleciton_name}/train/{ds_name}.csv',
    window_size=window_size)
test_ds = get_dataset(
    f'data/{topic}/{colleciton_name}/test/{ds_name}.csv',
    window_size=window_size)

train_dl = DataLoader(train_ds, batch_size=500)
test_dl = DataLoader(test_ds, batch_size=500)
test_index = test_ds.sequences[0].index

test_cls_path = f'saved_scores_preds/{colleciton_name}/{ds_name}/record_classes/{window_size}.csv'
test_cls = pd.read_csv(
    test_cls_path, header=None)\
    .iloc[:, 0].to_numpy()
test_point_cls_path = f'data/{topic}/{colleciton_name}/test_label/{ds_name}.csv'
test_point_cls = pd.read_csv(
    test_point_cls_path, header=None)\
    .iloc[:, 0].to_numpy()

n_features = train_ds.sequences[0].shape[1]


x_hat1 = np.load(scores_dirpath + 'x_hat1.npy', allow_pickle=True)
x_hat1_err_points = np.load(scores_dirpath + 'x_hat1_err_points.npy', allow_pickle=True)
x_hat1_err_points_st = np.load(scores_dirpath + 'x_hat1_err_points_st.npy', allow_pickle=True)


# searching for best threshold in rec err
betas = [0.5, 1]
wss = [200]  # np.arange(100, 501, 100)
ths = np.linspace(0.25, 1.25, 50)
th_df = th_ws_experiment(
    series_index=test_index, point_scores=x_hat1_err_points,
    point_cls=test_point_cls, t_max=500,
    ths=ths, wss=wss, betas=betas)


# searching for best threshold in std stats
ws_list = [100, 200, 300, 400]
bounds = [
    get_diff(get_std(x_hat1_err_points, ws))
    for ws in ws_list
]

betas = [0.5, 1]
ths = np.linspace(0.005, 0.1, 40)

std_th_df = stats_experiment(
    series_index=test_index, t_max=500,
    point_scores_list=bounds, point_cls=test_point_cls,
    ths_list=[ths] * len(ws_list), ws_list=ws_list, betas=betas)

save_th_exp(std_th_df, scores_dirpath + 'std_th_exp.csv')
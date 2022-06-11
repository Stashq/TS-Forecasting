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
import scipy
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



    # exp = load_experimentator('./saved_experiments/2022-06-05_16:27:49.pkl')
    # model_id = 0


    # m_name = 'AnomTrans_l3_d512_lambda3'
    # topic, colleciton_name, ds_name = 'Industry', 'ServerMachineDataset', 'machine-1-1'
    # exp_date = exp.exp_date
    # model = exp.load_pl_model(model_id, f'checkpoints/machine-1-1/{m_name}')
    # window_size = model.model.params['window_size']
    # scores_dirpath = f'notebook_a_scores/{colleciton_name}/{ds_name}/{m_name}/{exp_date}/'

    # train_ds = get_dataset(
    #     f'data/{topic}/{colleciton_name}/train/{ds_name}.csv',
    #     window_size=window_size)
    # test_ds = get_dataset(
    #     f'data/{topic}/{colleciton_name}/test/{ds_name}.csv',
    #     window_size=window_size)

    # train_dl = DataLoader(train_ds, batch_size=500)
    # test_dl = DataLoader(test_ds, batch_size=500)
    # test_index = test_ds.sequences[0].index

    # test_cls_path = f'saved_scores_preds/{colleciton_name}/{ds_name}/record_classes/{window_size}.csv'
    # test_cls = pd.read_csv(
    #     test_cls_path, header=None)\
    #     .iloc[:, 0].to_numpy()
    # test_point_cls_path = f'data/{topic}/{colleciton_name}/test_label/{ds_name}.csv'
    # test_point_cls = pd.read_csv(
    #     test_point_cls_path, header=None)\
    #     .iloc[:, 0].to_numpy()

def load_necessities(
    exp: Experimentator, m_id: int, m_name: str,
    ds_topic: str, ds_collection: str, ds_name: str
):
    model = exp.load_pl_model(m_id, f'checkpoints/{ds_name}/{m_name}')
    window_size = model.model.params['window_size']

    train_df = pd.read_csv(
        f'data/{ds_topic}/{ds_collection}/train/{ds_name}.csv', header=None)
    test_df = pd.read_csv(
        f'data/{ds_topic}/{ds_collection}/test/{ds_name}.csv', header=None)
    train_df.columns = train_df.columns.astype(int)
    test_df.columns = test_df.columns.astype(int)

    test_point_cls = pd.read_csv(
        f'data/{ds_topic}/{ds_collection}/test_label/{ds_name}.csv', header=None
    ).iloc[:, 0].to_numpy()

    return model, window_size, train_df, test_df, test_point_cls


def find_th(
    exp: Experimentator, m_id: int, m_name: str,
    ds_topic: str, ds_collection: str, ds_name: str,
    ws_list: List[int] = range(200, 1001, 100),
    art_anom_id: int = 4400, ppf_val: float = 0.995
):
    model, m_ws, train_df, test_df, test_point_cls = load_necessities(
        exp=exp, m_id=m_id, m_name=m_name, ds_topic=ds_topic,
        ds_collection=ds_collection, ds_name=ds_name
    )
    test_index = test_df.index

    # test reconstruction from training dataset start index
    recon_start_id = int(train_df.shape[0] * 0.8)
    art = insert_anomaly(
        ts_df=train_df.iloc[recon_start_id:],
        art_anom_id=art_anom_id, ppf_val=ppf_val)

    art_bounds = create_a_score_moving_std_diff(
        model=model, ts_df=art, m_ws=m_ws, ws_list=ws_list)
    test_ds_bounds = create_a_score_moving_std_diff(
        model=model, ts_df=test_df, m_ws=m_ws, ws_list=ws_list)

    th_art_test_best = {
        ws: float(art_bounds[i].iloc[art_anom_id+1])
        for i, ws in enumerate(ws_list)
    }

    gen_ths = [(ws, th) for ws, th in th_art_test_best.items()]
    wss, ths = list(zip(*gen_ths))
    ths_list = [[th] for th in ths]
    art_th_exp = stats_experiment(
        test_index, test_ds_bounds, ths_list=ths_list, ws_list=wss,
        point_cls=test_point_cls, betas=[0.5, 1.0])

    save_th_exp(
        art_th_exp, f'saved_scores_preds/{ds_collection}/{ds_name}/auto_th_find_exp.csv')
    return art_th_exp


def create_a_score_moving_std_diff(
    model, ts_df: pd.DataFrame, m_ws: int, ws_list: List[int]
):
    dl = DataLoader(
        MultiTimeSeriesDataset([ts_df], m_ws, target=ts_df.columns.tolist()),
        batch_size=1
    )
    a_scores = get_a_scores_one_per_point(
        model, dl, m_ws, return_only_a_score=True
    )
    msds = [
        get_diff(get_std(a_scores, ws))
        for ws in ws_list
    ]
    return msds


def insert_anomaly(
    ts_df: pd.DataFrame, art_anom_id: int, ppf_val: float = 0.995
) -> pd.DataFrame:
    """Creating artificial dataset
    from part of training dataset to test model reconstruction permormance
    by adding and subtracting 
    standard deviation multiplied by arbitrary z_score
    from mean in selected point"""
    assert art_anom_id < ts_df.shape[0]
    ts_df_std = ts_df.std()
    ts_df_mean = ts_df.mean()
    art = ts_df.copy()

    z_f = scipy.stats.norm.ppf(ppf_val)
    art.iloc[art_anom_id] = ts_df_mean + z_f * ts_df_std
    art.iloc[art_anom_id+1] = ts_df_mean - z_f * ts_df_std
    return art


def find_ths_for_collection(
    exp: Experimentator, model_id: int,
    model_name: str, ds_topic: str, ds_collection: str,
    ds_names: List[str], ppf_val: float = 0.995,
    verbose: bool = True, safe: bool = False
):
    res_df = []

    for ds_name in ds_names:
        try:
            if verbose:
                print('Finding threshold for dataset %s.' % ds_name)
            art_th_exp = find_th(
                exp=exp, m_id=model_id, m_name=model_name, ds_topic=ds_topic,
                ds_collection=ds_collection, ds_name=ds_name,
                ws_list=range(200, 1001, 100), art_anom_id=4400,
                ppf_val=ppf_val
            )
            art_th_exp['ds_name'] = ds_name
            res_df += [art_th_exp]

            if verbose:
                print(art_th_exp[['ws', 'th', 'f0.5-score', 'f1.0-score']])
        except Exception as e:
            if safe:
                print('Problem with dataset %s.' % ds_name)
                print(e)
            else:
                raise e

    res_df = pd.concat(res_df)
    save_th_exp(res_df, f'saved_scores_preds/{ds_collection}/collection_th_finding_exp.csv')


if __name__ == '__main__':
    # exp = load_experimentator('./saved_experiments/2022-06-05_16:27:49.pkl')
    exp = load_experimentator('./saved_experiments/2022-06-05_17:34:24.pkl')

    model_id = 0
    model_name = 'AnomTrans_l3_d512_lambda3'
    ds_topic, ds_collection = 'Industry', 'ServerMachineDataset'
    ds_names = os.listdir(f'./data/{ds_topic}/{ds_collection}/test')
    ds_names = [ds_n[:-4] for ds_n in ds_names]

    # art_th_exp = find_th(
    #     exp=exp, m_id=model_id, m_name=model_name, ds_topic=ds_topic,
    #     ds_collection=ds_collection, ds_name='machine-2-2',
    #     ws_list=range(200, 1001, 100), art_anom_id=4400
    # )
    find_ths_for_collection(
        exp=exp, model_id=model_id, model_name=model_name,
        ds_topic=ds_topic, ds_collection=ds_collection,
        ds_names=ds_names, ppf_val=0.999,
        verbose=True
    )

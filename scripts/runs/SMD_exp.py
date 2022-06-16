# flake8: noqa

import os
import csv

from sklearn.metrics import fbeta_score

from predpy.dataset import MultiTimeSeriesDataset
from predpy.data_module import MultiTimeSeriesModule
from predpy.wrapper import Autoencoder, Predictor, VAE
from predpy.experimentator import (
    DatasetParams, ModelParams,
    Experimentator, load_experimentator, load_last_experimentator)
from predpy.plotter import (
    plot_exp_predictions, get_cls_ids_ranges
)
from predpy.preprocessing import set_index
from predpy.preprocessing import moving_average
from predpy.preprocessing import (
    load_and_preprocess, set_index, moving_average, drop_if_is_in,
    use_dataframe_func, loc, iloc, get_isoforest_filter, get_variance_filter)
from predpy.trainer import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)
from predpy.experimentator import LearningParams
from tsad.noiser import apply_noise_on_dataframes, white_noise
from tsad.anomaly_detector import PredictionAnomalyDetector, ReconstructionAnomalyDetector
from models import LSTMAE, LSTMVAE, LSTMAEWrapper
from literature.anom_trans import AnomalyTransformer, ATWrapper
from literature.velc import VELC, VELCWrapper
from literature.dagmm import DAGMM, DAGMMWrapper
from literature.tadgan import TADGAN, TADGANWrapper
from anomaly_detection import AnomalyDetector
from models.ideas import LSTMMVR, ConvMVR, MVRWrapper
from models import ConvAE, MultipleConvAE, ConvAEWrapper
from anomaly_detection import (
    AnomalyDetector, fit_run_detection, exp_fit_run_detection,
    get_dataset, get_dataset_names, get_train_test_ds)

from notebook_utils.modeling import (
    predict, get_a_scores, get_rec_fbeta_score_conf_mat,
    get_a_scores_one_per_point, get_recon_one_per_point,
    adjust_point_cls_with_window,
    th_ws_experiment, stats_experiment,
    calculate_rec_wdd, recalculate_wdd,
    exctract_a_scores, a_score_exp
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

from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
# from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import nn
from typing import List, Dict, Literal
from predpy.plotter import plot_anomalies
from pathlib import Path

# =============================================================================

window_size = 200
batch_size = 64

c_in = 38
c_out = 38
topic = "Industry"
collection_name = "ServerMachineDataset"
# dataset_name1 = "machine-1-1"
# dataset_name2 = "machine-1-2"
# dataset_name3 = "machine-1-3"
# ds_names = os.listdir(f'./data/{topic}/{collection_name}/train')
# ds_names = ["machine-1-1.csv"]  # , "machine-1-2.csv", "machine-1-3.csv"]
# ds_names = [
#     f'machine-{id_}.csv'
#     for id_ in ['1-1', '1-7', '2-1', '2-3', '2-6', '3-6', '3-10']]
ds_names = [
    f'machine-{id_}.csv'
    for id_ in ['1-1', '2-1', '3-10']]

# c_in = 1
# c_out = 1
# topic = "Handmade"
# collection_name = "Sin"
# dataset_name = "artificial_1"

load_params = {
    "header": None, "names": [str(i) for i in range(c_in)]
}

drop_refill_pipeline = []
preprocessing_pipeline = [
    (use_dataframe_func, "astype", "float"),
]
detect_anomalies_pipeline = []

datasets_params = [
    DatasetParams(
        path="/home/stachu/Projects/Anomaly_detection/TSAD/data/%s/%s/train/%s.csv" % (topic, collection_name, ds_name[:-4]),
        load_params=load_params,
        target=[str(i) for i in range(c_in)],
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=batch_size,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        detect_anomalies_pipeline=detect_anomalies_pipeline,
        scaler=StandardScaler())
    for ds_name in ds_names
]

models_params = [
    ModelParams(
        name_=f"LSTMAE_h50_z50_l2", cls_=LSTMAE,
        init_params=dict(
            c_in=c_in, h_size=50, n_layers=2, z_size=50),
        WrapperCls=LSTMAEWrapper
    ),
    ModelParams(
        name_=f"LSTMAE_h100_z50_l2", cls_=LSTMAE,
        init_params=dict(
            c_in=c_in, h_size=100, n_layers=2, z_size=50),
        WrapperCls=LSTMAEWrapper
    ),
    ModelParams(
        name_=f'ConvAE_ws{window_size}_nk20_ks3_es50', cls_=ConvAE,
        init_params=dict(
            window_size=window_size, c_in=c_in, n_kernels=20,
            kernel_size=3, emb_size=50),
        WrapperCls=ConvAEWrapper
    ),
    ModelParams(
        name_="VELC_h100_l2_z50_N50_th0.0", cls_=VELC,
        init_params=dict(
            c_in=c_in, window_size=window_size, h_size=100, n_layers=2, z_size=50,
            h1_size=1024, h2_size=512,
            N_constraint=50, threshold=0.0),
        WrapperCls=VELCWrapper
    ),
    ModelParams(
        name_="TadGAN_h100_l2_z50_g1d1_warmup0_(literature)", cls_=TADGAN,
        init_params=dict(
            window_size=window_size, c_in=c_in, h_size=100, n_layers=2, z_size=50),
        WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
            gen_dis_train_loops=(1, 1), warmup_epochs=0, alpha=0.5)
    ),
    # ModelParams(
    #     name_="AnomTrans_l2_d10_lambda10", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=10, n_layers=2,
    #         lambda_=10),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l3_d512_lambda10", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=512, n_layers=3,
    #         lambda_=10),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="TadGAN_h50_l1_z10_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=50, n_layers=1, z_size=10),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # ModelParams(
    #     name_="TadGAN_h50_l1_z50_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=50, n_layers=1, z_size=50),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # # h_size = 100
    # ModelParams(
    #     name_="TadGAN_h100_l1_z5_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=1, z_size=5),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # ModelParams(
    #     name_="TadGAN_h100_l1_z10_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=1, z_size=10),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # ModelParams(
    #     name_="TadGAN_h100_l1_z50_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=1, z_size=50),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # # h_size = 100, n_layers = 2
    # ModelParams(
    #     name_="TadGAN_h100_l2_z5_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=2, z_size=5),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # ModelParams(
    #     name_="TadGAN_h100_l2_z10_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=2, z_size=10),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # ModelParams(
    #     name_="TadGAN_h100_l2_z50_g1d1_warm0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=100, n_layers=2, z_size=50),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)
    # ),
    # # n_layers = 2
    # ModelParams(
    #     name_="AnomTrans_l2_d2_l2", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=2, n_layers=2,
    #         lambda_=0.5),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l2_d5_l2", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=5, n_layers=2,
    #         lambda_=0.5),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l3_d512_lambda3", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=512, n_layers=3,
    #         lambda_=3),
    #     WrapperCls=ATWrapper),

    # # ModelParams(
    # #     name_=f"LSTMMVR_w{window_size}_h50_z10_l1", cls_=LSTMMVR,
    # #     init_params=dict(
    # #         window_size=window_size, c_in=c_in, h_size=50, z_size=10,
    # #         n_layers=1),
    # #     WrapperCls=MVRWrapper
    # # ),
    # ModelParams(
    #     name_=f'ConvMVR_ws{window_size}_nk10_ks3_es50', cls_=ConvMVR,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50, lambda_=0.3),
    #     WrapperCls=MVRWrapper
    # ),
    # ModelParams(
    #     name_=f'AE_ws{window_size}_nk10_ks3_es50', cls_=ConvAE,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50),
    #     WrapperCls=ConvAEWrapper
    # ),
    # ModelParams(
    #     name_=f'MultipleConvAE_ws{window_size}_nk10_ks3_es50', cls_=MultipleConvAE,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50),
    #     WrapperCls=ConvAEWrapper
    # ),
    # ModelParams(
    #     name_=f'ConvMVR_ws{window_size}_nk10_ks3_es50', cls_=ConvMVR,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50, lambda_=0.5),
    #     WrapperCls=MVRWrapper
    # ),
    # ModelParams(
    #     name_=f'ConvMVR_ws{window_size}_nk10_ks3_es50', cls_=ConvMVR,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50, lambda_=0.7),
    #     WrapperCls=MVRWrapper
    # ),
]


chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=15, gpus=1, auto_lr_find=True)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=3, min_delta=1e-3, verbose=True)

exp = Experimentator(
    models_params=models_params,
    datasets_params=datasets_params,
    trainer_params=tr_p,
    checkpoint_params=chp_p,
    early_stopping_params=es_p,
    LoggersClasses=[TensorBoardLogger],
    loggers_params=[LoggerParams(save_dir="./lightning_logs")]
)

# exp = load_experimentator('./saved_experiments/2022-06-14_15:36:53.pkl')
# exp = load_experimentator('./saved_experiments/2022-06-14_23:58:28.pkl')

# exp = load_last_experimentator('./saved_experiments')
# exp.run_experiments(
#     experiments_path="./saved_experiments",
#     safe=True, continue_run=True
# )
# for i in exp.datasets_params.index:
#     ds_name = exp.datasets_params.loc[i, 'name_']
#     plot_exp_predictions(
#         exp, dataset_idx=i,
#         file_path='./pages/%s/%s/%s/%s.html' % (topic, collection_name, ds_name, str(exp.exp_date))
#     )


def find_th(
    exp: Experimentator,
    ws_list: List[int], ths: List[float],
    save_scores: bool = False, load_scores: bool = False,
    ds_names: List[str] = None, m_ids: List[int] = None,
    verbose: bool = True, scale: bool = True,
    calculate_training_a_score: bool = True,
    recalculate_ths: bool = False
):
    model_train_date = exp.exp_date
    if ds_names is None:
        ds_ids = exp.datasets_params.index.tolist()
    else:
        ds_p = exp.datasets_params
        ds_ids = ds_p[ds_p['name_'].isin(ds_names)].index.tolist()
    if m_ids is None:
        m_ids = exp.models_params.index.tolist()

    for ds_id in ds_ids:
        window_size = exp.datasets_params.loc[ds_id]['window_size']
        ts_scaler = exp.datasets_params.loc[ds_id]['scaler']
        topic, collection_name, ds_name =\
            get_dataset_names(
                exp.datasets_params.loc[ds_id]['path'])
        print('Running experiment with ' + ds_name)

        # loading data
        train_ds, test_ds = get_train_test_ds(
            topic=topic, collection_name=collection_name, ds_name=ds_name,
            window_size=window_size, ts_scaler=ts_scaler, fit_scaler=True)

        train_dl = DataLoader(train_ds, batch_size=500)
        test_dl = DataLoader(test_ds, batch_size=500)
        test_index = test_ds.sequences[0].index

        test_point_cls_path = f'data/{topic}/{collection_name}/test_label/{ds_name}.csv'
        test_point_cls = pd.read_csv(
            test_point_cls_path, header=None)\
            .iloc[:, 0].to_numpy()
        test_rec_cls = adjust_point_cls_with_window(
            test_point_cls, window_size, return_point_cls=False)

        ds_results = {}

        for m_id in m_ids:
            m_name = exp.models_params.loc[m_id]['name_']
            try:
                model = exp.load_pl_model(m_id, f'checkpoints/{ds_name}/{m_name}')
            except:
                # print(f'Cannot find model {m_name} for dataset {ds_name}.')
                pass
                continue
            print('Model: ' + m_name)

            train_a_scores = None
            dirpath = f'notebook_a_scores/{collection_name}/{ds_name}/{m_name}/{model_train_date}/'
            if load_scores and os.path.exists(dirpath + 'test_a_scores.npy'):
                test_a_scores = np.load(dirpath + 'test_a_scores.npy', allow_pickle=True)
            else:
                # ONLY FOR ANOMALY TRANSFORMER
                if isinstance(model.model, AnomalyTransformer):
                    train_dl = DataLoader(train_ds, batch_size=1)
                    test_dl = DataLoader(test_ds, batch_size=1)
                if calculate_training_a_score:
                    print('Calculating a scores for train dataset...')
                    if scale:
                        train_a_scores = model.fit_scores_scaler(
                            train_dl, use_tqdm=True)
                    else:
                        train_a_scores = get_a_scores(
                            model=model, dataloader=train_dl, use_tqdm=True)

                print('Calculating a scores for test dataset...')
                test_a_scores = get_a_scores(
                    model, test_dl, scale=scale, use_tqdm=True)
                # saving a_scores
                if save_scores:
                    os.makedirs(dirpath, exist_ok=True)
                    if calculate_training_a_score:
                        train_a_scores.dump(dirpath + 'train_a_scores.npy')
                    test_a_scores.dump(dirpath + 'test_a_scores.npy')
                    print('Scores saved.')

                # GETTING BACK DATALOADERS
                if isinstance(model.model, AnomalyTransformer):
                    train_dl = DataLoader(train_ds, batch_size=500)
                    test_dl = DataLoader(test_ds, batch_size=500)

            sth_exp_path = f'notebook_a_scores/{collection_name}/{ds_name}/{m_name}/{model_train_date}/std_th_exp.csv'
            if not os.path.exists(sth_exp_path) or recalculate_ths:
                print('Finding threshold for model ' + m_name)
                # fitting with LR
                lr_model = LogisticRegression().fit(
                    test_a_scores, test_rec_cls)
                pred_cls = lr_model.predict(test_a_scores)
                pred_cls = adjust_point_cls_with_window(
                    pred_cls, window_size, return_point_cls=True)
                f1 = fbeta_score(test_rec_cls, pred_cls, beta=1)
                f0_5 = fbeta_score(test_rec_cls, pred_cls, beta=0.5)
                ds_results[m_name] = [f1, f0_5]

                bounds = [
                    get_diff(get_std(test_a_scores, ws))
                    for ws in ws_list
                ]
                std_th_df = stats_experiment(
                    series_index=test_index, t_max=None, model_ws=window_size,
                    scores_list=bounds, point_cls=test_point_cls, ths_list=[ths] * len(ws_list),
                    ws_list=ws_list, betas=[1.0, 0.5])
                if verbose:
                    df = std_th_df.loc[std_th_df.groupby(['ws'])['f1.0-score'].idxmax()]
                    df = df.drop(columns=['preds_rec_cls', 'wdd'])
                    print(df)

                # saving threshold finding experiment results
                os.makedirs(os.path.dirname(sth_exp_path), exist_ok=True)
                save_th_exp(std_th_df, sth_exp_path)

        if len(ds_results) > 0:
            # saving prediction scores
            path = f'notebook_a_scores/{collection_name}/{ds_name}/lr_prediction_results/{model_train_date}.csv'
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df = pd.DataFrame.from_dict(
                ds_results, orient='index', columns=['f1.0-score', 'f0.5-score']
            )
            df.to_csv(path)
            if verbose:
                print('Logistic regression detection relusts:')
                print(df)
        print('\n')


print('EXPERIMENT: 2022-06-14_15:36:53')
exp = load_experimentator('./saved_experiments/2022-06-14_15:36:53.pkl')
find_th(
    exp, ws_list=[50, 100, 200, 300, 400, 500],
    ths=np.linspace(5e-6, 2e-2, 1000),
    load_scores=True, save_scores=True,
    ds_names=['machine-1-1', 'machine-2-1', 'machine-3-10'],
    m_ids=None, scale=True, calculate_training_a_score=True,
    recalculate_ths=True
)
print('EXPERIMENT: 2022-06-14_23:58:28')
exp2 = load_experimentator('./saved_experiments/2022-06-14_23:58:28.pkl')
find_th(
    exp, ws_list=[50, 100, 200, 300, 400, 500],
    ths=np.linspace(5e-6, 2e-2, 1000),
    load_scores=True, save_scores=True,
    ds_names=['machine-1-1', 'machine-2-1', 'machine-3-10'],
    m_ids=None, scale=True, calculate_training_a_score=True,
    recalculate_ths=True
)
print('EXPERIMENT: 2022-06-15_05:11:43')
exp3 = load_experimentator('./saved_experiments/2022-06-15_05:11:43.pkl')
find_th(
    exp, ws_list=[50, 100, 200, 300, 400, 500],
    ths=np.linspace(5e-6, 2e-2, 1000),
    load_scores=True, save_scores=True,
    ds_names=['machine-1-1', 'machine-2-1', 'machine-3-10'],
    m_ids=None, scale=True, calculate_training_a_score=True,
    recalculate_ths=True
)

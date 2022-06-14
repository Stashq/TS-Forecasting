# flake8: noqa

import os
import sys
sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import MultiTimeSeriesDataset
from predpy.data_module import MultiTimeSeriesModule
from predpy.wrapper import Autoencoder, Predictor, VAE
from predpy.experimentator import (
    DatasetParams, ModelParams,
    Experimentator, load_experimentator, load_last_experimentator)
from predpy.plotter import (
    plot_exp_predictions
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
from models import LSTMAE, LSTMVAE
from literature.anom_trans import AnomalyTransformer, ATWrapper
from literature.velc import VELC, VELCWrapper
from literature.dagmm import DAGMM, DAGMMWrapper
from literature.tadgan import TADGAN, TADGANWrapper
from anomaly_detection import AnomalyDetector
from anomaly_detection.data_loading import get_dataset
from models.ideas import LSTMMVR, ConvMVR, MVRWrapper
from models import ConvAE, ConvAEWrapper, LSTMAE, LSTMAEWrapper


from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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

# c_in = 38
# c_out = 38
# topic = "Industry"
# collection_name = "ServerMachineDataset"
# dataset_name = "machine-1-1"

c_in = 1
c_out = 1
topic = "Handmade"
collection_name = "Sin"
dataset_name = "artificial_1"

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
        path="/home/stachu/Projects/Anomaly_detection/TSAD/data/%s/%s/train/%s.csv" % (topic, collection_name, dataset_name),
        load_params=load_params,
        target=[str(i) for i in range(c_in)],
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=batch_size,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        detect_anomalies_pipeline=detect_anomalies_pipeline,
        scaler=StandardScaler()),
]

models_params = [
    # ModelParams(
    #     name_=f"LSTMAE_h50_z10_l1", cls_=LSTMAE,
    #     init_params=dict(
    #         c_in=c_in, h_size=50, n_layers=1, z_size=10),
    #     WrapperCls=LSTMAEWrapper
    # ),
    # ModelParams(
    #     name_=f'ConvAE_ws{window_size}_nk10_ks3_es50', cls_=ConvAE,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, n_kernels=10,
    #         kernel_size=3, emb_size=50),
    #     WrapperCls=ConvAEWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N10_th0.0", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=10, threshold=0.0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N20_th0.0", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=20, threshold=0.0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N30_th0.0", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=30, threshold=0.0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N10_th0.025", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=10, threshold=0.025),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N20_th0.025", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=20, threshold=0.025),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h50_l1_z10_N30_th0.025", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=50, n_layers=1, z_size=10,
    #         N_constraint=30, threshold=0.025),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="TadGAN_h50_l1_z10_g1d1_warmup0", cls_=TADGAN,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, h_size=50, n_layers=1, z_size=10),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=0)),
    # ModelParams(
    #     name_="AnomTrans_l3_d512_lambda3", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         window_size=window_size, c_in=c_in, d_model=512, n_layers=3,
    #         lambda_=3),
    #     WrapperCls=ATWrapper),
    ModelParams(
        name_="AnomTrans_l1_d3_lambda10", cls_=AnomalyTransformer,
        init_params=dict(
            window_size=window_size, c_in=c_in, d_model=3, n_layers=1,
            lambda_=10),
        WrapperCls=ATWrapper),
]


chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=40, gpus=1, auto_lr_find=False)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=10, min_delta=1e-2, verbose=True)

exp = Experimentator(
    models_params=models_params,
    datasets_params=datasets_params,
    trainer_params=tr_p,
    checkpoint_params=chp_p,
    early_stopping_params=es_p,
    LoggersClasses=[TensorBoardLogger],
    loggers_params=[LoggerParams(save_dir="./lightning_logs")]
)

# exp = load_experimentator('./saved_experiments/2022-05-27_22:36:05.pkl')
# exp = load_last_experimentator('./saved_experiments')
exp.run_experiments(
    experiments_path="./saved_experiments",
    safe=False, continue_run=True
)
plot_exp_predictions(
    exp, dataset_idx=0,
    file_path='./pages/%s/%s/%s/%s.html' % (topic, collection_name, dataset_name, str(exp.exp_date))
)



# # exp = load_experimentator('./saved_experiments/2022-05-25_20:31:39.pkl')
# exp = load_last_experimentator('./saved_experiments')

# m_id, ds_id = 0, 0
# # plot_exp_predictions(
# #     exp, dataset_idx=ds_id,
# #     # file_path='./pages/Handmade/%s/%s.html' % (dataset_name, str(exp.exp_date))
# # )


# ds_id = 0
# min_points = 3
# test_cls_path = './data/%s/%s/test_label/%s.csv' % (topic, collection_name, dataset_name)
# class_weight = {0: 0.7, 1: 0.3}

# dataset = get_dataset(
#     path='./data/%s/%s/test/%s.csv' % (topic, collection_name, dataset_name),
#     window_size=window_size, ts_scaler=exp.get_targets_scaler(ds_id))
# data_classes = pd.read_csv(
#     test_cls_path, header=None)\
#     .iloc[:, 0].to_list()
# # rec_classes = None
# rec_classes = dataset.get_recs_cls_by_data_cls(
#     data_classes, min_points=min_points)
# n_models = exp.models_params.shape[0]

# for m_id in range(0, n_models):
#     model_name = exp.models_params.iloc[m_id]['name_']
#     try:
#         model = exp.load_pl_model(
#             m_id, os.path.join('checkpoints', dataset_name, model_name))
#         model.fit_run_detection(
#             window_size=window_size,
#             test_path='./data/%s/%s/test/%s.csv' % (topic, collection_name, dataset_name),
#             rec_classes=rec_classes,
#             test_cls_path=test_cls_path,
#             min_points=min_points, scale_scores=True,  # class_weight=class_weight,
#             ts_scaler=exp.get_targets_scaler(ds_id),
#             # load_scores_path = './saved_scores_preds/%s/%s/%s/anom_scores.csv' % (collection_name, dataset_name, model_name),
#             save_scores_path= './saved_scores_preds/%s/%s/%s/anom_scores.csv' % (collection_name, dataset_name, model_name),
#             # load_preds_path = './saved_scores_preds/%s/%s/%s/preds.csv' % (collection_name, dataset_name, model_name),
#             save_preds_path = './saved_scores_preds/%s/%s/%s/preds.csv' % (collection_name, dataset_name, model_name),
#             plot=True,  # start_plot_pos=15000, end_plot_pos=21000,
#             save_html_path='./pages/%s/%s/%s.html' % (collection_name, dataset_name, model_name),
#             f_score_beta=0.5,
#             wdd_t_max=window_size/2,
#             wdd_w_f=0.002,
#             wdd_ma_f=0.01
#         )
#     except Exception as e:
#         raise e
        # print('Problem with fit_run_detection on model "%s".' % model_name)

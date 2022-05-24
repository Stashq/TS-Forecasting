# flake8: noqa

import sys
sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import MultiTimeSeriesDataset
from predpy.data_module import MultiTimeSeriesModule
from predpy.wrapper import Autoencoder, Predictor, VAE
from predpy.experimentator import (
    DatasetParams, ModelParams,
    Experimentator, load_experimentator)
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
from literature.anomaly_detector_base import AnomalyDetector
from models.ideas import LSTMMVR, ConvMVR, MVRWrapper

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

window_size = 100
batch_size = 64

load_params = {
    "header": None, "names": [str(i) for i in range(38)]
}

drop_refill_pipeline = []
preprocessing_pipeline = [
    (use_dataframe_func, "astype", "float"),
]
detect_anomalies_pipeline = []

dataset_name = "machine-1-1"

datasets_params = [
    DatasetParams(
        path="/home/stachu/Projects/Anomaly_detection/TSAD/data/Industry/ServerMachineDataset/train/%s.csv" % dataset_name,
        load_params=load_params,
        target=[str(i) for i in range(38)],
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=batch_size,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        detect_anomalies_pipeline=detect_anomalies_pipeline,
        scaler=StandardScaler()),
    # DatasetParams(
    #     path="/home/stachu/Projects/Anomaly_detection/TSAD/data/Industry/ServerMachineDataset/train/machine-1-2.csv",
    #     load_params=load_params,
    #     target=[str(i) for i in range(38)],
    #     split_proportions=[0.8, 0.1, 0.1],
    #     window_size=window_size,
    #     batch_size=batch_size,
    #     drop_refill_pipeline=drop_refill_pipeline,
    #     preprocessing_pipeline=preprocessing_pipeline,
    #     detect_anomalies_pipeline=detect_anomalies_pipeline,
    #     scaler=StandardScaler()),
    # DatasetParams(
    #     path="/home/stachu/Projects/Anomaly_detection/TSAD/data/Industry/ServerMachineDataset/train/machine-1-3.csv",
    #     load_params=load_params,
    #     target=[str(i) for i in range(38)],
    #     split_proportions=[0.8, 0.1, 0.1],
    #     window_size=window_size,
    #     batch_size=batch_size,
    #     drop_refill_pipeline=drop_refill_pipeline,
    #     preprocessing_pipeline=preprocessing_pipeline,
    #     detect_anomalies_pipeline=detect_anomalies_pipeline,
    #     scaler=StandardScaler()),
]

c_in = 38
c_out = 38


models_params = [
    ModelParams(
        name_="ConvMVR", cls_=ConvMVR,
        init_params=dict(
            window_size=window_size, c_in=c_in, n_kernels=10,
            kernel_size=3, emb_size=50, z_glob_size=0),
        WrapperCls=MVRWrapper
    ),
    # ModelParams(
    #     name_="LSTMMVR", cls_=LSTMMVR,
    #     init_params=dict(
    #         c_in=c_in, h_size=100, z_size=50, z_glob_size=0),
    #     WrapperCls=MVRWrapper
    # ),
    # ModelParams(
    #     name_="LSTMAE_h1000_l2_z800", cls_=LSTMAE,
    #     init_params=dict(
    #         c_in=c_in, h_size=1000, n_layers=2, z_size=800),
    #     WrapperCls=Autoencoder
    # ),
    # ModelParams(
    #     name_="VELC_h200_l2_z300", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=200, n_layers=2, z_size=300,
    #         N_constraint=20, threshold=0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h200_l2_z400", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=200, n_layers=2, z_size=400,
    #         N_constraint=20, threshold=0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h200_l3_z400", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=200, n_layers=3, z_size=400,
    #         N_constraint=20, threshold=0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="VELC_h300_l3_z400", cls_=VELC,
    #     init_params=dict(
    #         c_in=c_in, window_size=window_size, h_size=300, n_layers=3, z_size=400,
    #         N_constraint=20, threshold=0),
    #     WrapperCls=VELCWrapper
    # ),
    # ModelParams(
    #     name_="TadGAN_h200_l2_z100_gen1_dis1_warm2", cls_=TADGAN,
    #     init_params=dict(
    #         c_in=c_in, h_size=200, n_layers=2, z_size=100),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=2)
    # ),
    # ModelParams(
    #     name_="TadGAN_h200_l2_z100_gen1_dis1_warm5", cls_=TADGAN,
    #     init_params=dict(
    #         c_in=c_in, h_size=200, n_layers=2, z_size=100),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(1, 1), warmup_epochs=5)
    # ),
    # ModelParams(
    #     name_="TadGAN_h200_l2_z400_gen3_dis1_warm5", cls_=TADGAN,
    #     init_params=dict(
    #         c_in=c_in, h_size=200, n_layers=2, z_size=400),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(3, 1), warmup_epochs=5)
    # ),
    # ModelParams(
    #     name_="TadGAN_h300_l3_z400_gen3_dis1_warm5", cls_=TADGAN,
    #     init_params=dict(
    #         c_in=c_in, h_size=300, n_layers=3, z_size=400),
    #     WrapperCls=TADGANWrapper, wrapper_kwargs=dict(
    #         gen_dis_train_loops=(3, 1), warmup_epochs=5)
    # ),
    # ModelParams(
    #     name_="AnomTrans_l2", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         N=window_size, d_model=c_in, layers=2, lambda_=0.5),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l3", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         N=window_size, d_model=c_in, layers=3, lambda_=0.5),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l4", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         N=window_size, d_model=c_in, layers=3, lambda_=0.5),
    #     WrapperCls=ATWrapper),
    # ModelParams(
    #     name_="AnomTrans_l5", cls_=AnomalyTransformer,
    #     init_params=dict(
    #         N=window_size, d_model=c_in, layers=3, lambda_=0.5),
    #     WrapperCls=ATWrapper),
]

chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=50, gpus=1, auto_lr_find=False)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=4, min_delta=3e-3, verbose=True)

exp = Experimentator(
    models_params=models_params,
    datasets_params=datasets_params,
    trainer_params=tr_p,
    checkpoint_params=chp_p,
    early_stopping_params=es_p,
    LoggersClasses=[TensorBoardLogger],
    loggers_params=[LoggerParams(save_dir="./lightning_logs")]
)

# exp.run_experiments(
#     experiments_path="./saved_experiments",
#     safe=False
# )
exp = load_experimentator('./saved_experiments/2022-05-24_19:33:35.pkl')

# exp = load_experimentator(
#     "./saved_experiments/2022-05-23_01:20:39.pkl"
# )
# exp.run_experiments(
#     experiments_path="./saved_experiments",
#     safe=True, continue_run=True)

# exp = load_experimentator(
#     "./saved_experiments/2022-05-21_00:57:43.pkl"
# )

# plot_exp_predictions(
#     exp, dataset_idx=0,
#     file_path='./pages/ServerMachineDataset/%s/%s.html' % (dataset_name, str(exp.exp_date)))

# plot_exp_predictions(
#     exp, dataset_idx=1,
#     file_path='./pages/ServerMachineDataset/machine-1-3/%s.html' % str(exp.exp_date))

# plot_exp_predictions(
#     exp, dataset_idx=2,
#     file_path='./pages/ServerMachineDataset/machine-1-3/%s.html' % str(exp.exp_date))

# velc_model = exp.load_pl_model(0, './checkpoints/machine-1-1/VELC/')
# tsm = exp.load_time_series_module(0)

# ts = pd.concat(tsm.val_dataloader().dataset.sequences)
# preds = pd.read_csv('./n_preds.csv')
# preds.index = ts.index[1:]
# plot_anomalies(ts, preds, [(23500, 24000)], [(24500, 25500)], None, True)


def fit_run_detection(
    model: AnomalyDetector, test_path, test_cls_path, min_points: int,
    plot: bool, scale_scores: bool, save_html_path=None,
    class_weight: Dict[Literal[0, 1], float] = {0: 0.5, 1: 0.5},
    load_scores_path: Path = None,
    save_scores_path: Path = None,
    load_preds_path: Path = None,
    save_preds_path: Path = None,
    ts_scaler: TransformerMixin = None,
    start_plot_pos: int = None,
    end_plot_pos: int = None
):
    """test_path file should contain columns of features without header in first line,
    test_cls file has to be csv with one column filled with values 0 (normal data) 1 (anomaly),
    min_points is minimal points required in record sequence to be anomaly,
    scale_scores should be True only if model requires scaling anomaly scores"""
    df = pd.read_csv(
        test_path, names=list(range(38)), header=None
    )
    dataset = MultiTimeSeriesDataset(
        sequences=[df],
        window_size=window_size,
        target=df.columns.tolist()
    )
    if load_scores_path is None:
        data_classes = pd.read_csv(
            test_cls_path, header=None)\
            .iloc[:, 0].to_list()
        rec_classes = dataset.get_recs_cls_by_data_cls(
            data_classes, min_points=min_points)
    else:
        rec_classes = []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    model.fit_detector(
        dataloader=dataloader,
        classes=np.array(rec_classes),
        plot=plot, scale_scores=scale_scores,
        save_html_path=save_html_path,
        class_weight=class_weight,
        load_scores_path=load_scores_path,
        save_scores_path=save_scores_path,
        load_preds_path=load_preds_path,
        save_preds_path=save_preds_path,
        ts_scaler=ts_scaler,
        start_plot_pos=start_plot_pos,
        end_plot_pos=end_plot_pos
    )


# fit_run_detection(
#     model=exp.load_pl_model(0, './checkpoints/machine-1-1/ConvMVR'),
#     test_path='./data/Industry/ServerMachineDataset/test/machine-1-1.csv',
#     test_cls_path='./data/Industry/ServerMachineDataset/test_label/machine-1-1.csv',
#     min_points=5, scale_scores=True, class_weight = {0: 0.1, 1: 0.9},
#     ts_scaler=exp.get_targets_scaler(0),
#     # load_scores_path = './anom_scores.csv',
#     save_scores_path= './anom_scores.csv',
#     # load_preds_path = './preds.csv',
#     save_preds_path = './preds.csv',
#     plot=True,# start_plot_pos=15000, end_plot_pos=21000,
#     save_html_path='./pages/ConvMSR_detection.html'
# )


def create_dataset(
    len_: int, lambdas: List[float], amplitudes: List[float], shift: float, noise_sig: float
):
    assert len(lambdas) == len(amplitudes), '"lambdas" and "amplitudes" have to have same lenght.'
    t = np.arange(len_)
    y = np.full(len_, shift)
    for lam, amp in zip(lambdas, amplitudes):
        wave = amp * np.sin(lam * t)
        y = y + wave
    y = y + noise_sig * np.random.randn(len_)
    return y

len_ = 1000
y = create_dataset(len_=len_, lambdas=[2, 0.1], amplitudes=[2, 100], shift=300, noise_sig=0.7)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.arange(len_), y)
fig.show()
# velc_model.fit_detector(
#     tsm.val_dataloader(), tsm.test_dataloader(),  # load_path='./tmp.csv',
#     plot=True, class_weight={0: 0.5, 1: 0.5}, scale_scores=True)

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
from tsad.noiser import apply_noise_on_dataframes, white_noise
from tsad.anomaly_detector import PredictionAnomalyDetector, ReconstructionAnomalyDetector
from models import LSTMAE, LSTMVAE
from literature.AnomTrans import AnomalyTransformer, ATWrapper

from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# =============================================================================

window_size = 100

load_params = {
    "header": None, "names": [str(i) for i in range(38)]
}

drop_refill_pipeline = []
preprocessing_pipeline = [
    (use_dataframe_func, "astype", "float"),
]
detect_anomalies_pipeline = []

datasets_params = [
    DatasetParams(
        path="/home/stachu/Projects/Anomaly_detection/TSAD/data/Industry/ServerMachineDataset/train/machine-1-1.csv",
        load_params=load_params,
        target=[str(i) for i in range(38)],
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=64,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        detect_anomalies_pipeline=detect_anomalies_pipeline,
        scaler=StandardScaler()),
]

c_in = 38
c_out = 38

models_params = [
    # ModelParams(
    #     name_="LSTMVAE_h200_l1", cls_=LSTMVAE,
    #     init_params=dict(
    #         c_in=c_in, h_size=200, n_layers=1),
    #     WrapperCls=VAE, wrapper_kwargs=dict(kld_weight=0.005)),
    ModelParams(
        name_="AnomTrans", cls_=AnomalyTransformer,
        init_params=dict(
            N=window_size, d_model=c_in, layers=2, lambda_=0.5),
        WrapperCls=ATWrapper),
]

chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=1, gpus=1, auto_lr_find=True)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=2, verbose=True)

exp = Experimentator(
    models_params=models_params,
    datasets_params=datasets_params,
    trainer_params=tr_p,
    checkpoint_params=chp_p,
    early_stopping_params=es_p,
    LoggersClasses=[TensorBoardLogger],
    loggers_params=[LoggerParams(save_dir="./lightning_logs")]
)

exp.run_experiments(experiments_path="./saved_experiments", safe=False)
# exp = load_experimentator(
#     "./saved_experiments/2021-12-29_01:31:42.pkl"
# )

plot_exp_predictions(exp, dataset_idx=0, models_ids=[0])

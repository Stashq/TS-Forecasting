# flake8: noqa
import sys

sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import MultiTimeSeriesDataset, MultiTimeSeriesDataloader
from predpy.data_module import MultiTimeSeriesModule
from predpy.wrapper import Autoencoder, Predictor, VAE, PAE
from predpy.experimentator import (
    DatasetParams, ModelParams,
    Experimentator, load_experimentator)
from predpy.preprocessing import set_index
from predpy.preprocessing import moving_average
from predpy.preprocessing import (
    load_and_preprocess, set_index, moving_average, drop_if_is_in,
    use_dataframe_func, loc, iloc, get_isoforest_filter, get_variance_filter)
from predpy.trainer import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)
from tsad.noiser import apply_noise_on_dataframes, white_noise
from tsad.anomaly_detector import (
    PredictionAnomalyDetector, ReconstructionAnomalyDetector,
    EmbeddingAnomalyDetector, ReconstructionDistributionAnomalyDetector)
from models import LSTMAE, LSTMVAE, LSTMPAE

from pytorch_lightning.loggers import TensorBoardLogger
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from prophet import Prophet


# =============================================================================

window_size = 366

load_params = {
    "sep": ';', "header": 0, "low_memory": False,
    "infer_datetime_format": True, "parse_dates": {'datetime': [0, 1]},
    "index_col": ['datetime']
}

columns = ["Global_active_power", "Voltage"]
drop_refill_pipeline = [
    (loc, {"columns": columns}),
    (drop_if_is_in, (["?", np.nan]), {"columns": columns}),
    # (iloc, {"rows_end": 1500}),
    # (iloc, {"rows_start": -20000}),
]
preprocessing_pipeline = [
    (use_dataframe_func, "astype", "float"),
]
detect_anomalies_pipeline = [
    # (get_isoforest_filter, dict(
    #     scores_threshold=-0.36, window_size=500, target="Global_active_power"))
    (get_variance_filter, dict(
        window_size=5000, log_variance_limits=(-7, 0),
        target="Global_active_power"))
]


datasets_params = [
    DatasetParams(
        path="/home/stachu/Projects/Anomaly_detection/Forecasting_models/data/Energy/household_power_consumption/household_power_consumption.csv",
        load_params=load_params,
        target="Global_active_power",
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=64,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
        detect_anomalies_pipeline=detect_anomalies_pipeline,
        scaler=MinMaxScaler()),
]

c_in = 2
c_out = 1

models_params = [
    # ModelParams(
    #     name_="TST_l3_fcDrop0.1", cls_=TST.TST,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "seq_len": window_size,
    #         "max_seq_len": window_size, "n_layers": 3, "fc_dropout": 0.1}),
    # ModelParams(
    #     name_="TST_l2_fcDrop0.1", cls_=TST.TST,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "seq_len": window_size,
    #         "max_seq_len": window_size, "n_layers": 2, "fc_dropout": 0.1}),
    # ModelParams(
    #     name_="TST_l2_fcDrop0.0", cls_=TST.TST,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "seq_len": window_size,
    #         "max_seq_len": window_size, "n_layers": 2, "fc_dropout": 0.0}),
    # ModelParams(
    #     name_="ResNet", cls_=ResNet.ResNet,
    #     init_params={"c_in": c_in, "c_out": c_out}),
    # ModelParams(
    #     name_="LSTM_h200_l1", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 1}),
    # ModelParams(
    #     name_="LSTM_h200_l2", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 2}),
    # ModelParams(
    #     name_="LSTM_h400_l4", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 400, "n_layers": 4}),
    # ModelParams(
    #     name_="LSTMAE_h200_l1", cls_=LSTMAE,
    #     init_params=dict(
    #         c_in=window_size, h_size=200, n_layers=1),
    #     WrapperCls=Autoencoder),
    # ModelParams(
    #     name_="LSTMVAE_h200_l1", cls_=LSTMVAE,
    #     init_params=dict(
    #         c_in=window_size, h_size=200, n_layers=1),
    #     WrapperCls=VAE, wrapper_kwargs=dict(kld_weight=1e-6)),
    ModelParams(
        name_="LSTMPAE_h200_l1", cls_=LSTMPAE,
        init_params=dict(
            c_in=window_size, h_size=200, n_layers=1),
        WrapperCls=PAE),
]

chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=1, gpus=1, auto_lr_find=True)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=2, verbose=True)

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger

# tmp = pl.Trainer(logger=TensorBoardLogger("./"))

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
#     experiments_path="./saved_experiments", safe=False, continue_run=False)

# pae
exp = load_experimentator(
    "./saved_experiments/2022-01-05_20:55:29.pkl")

# ae
# exp = load_experimentator(
#     "./saved_experiments/2022-01-04_23:45:30.pkl")

# vae
# exp = load_experimentator(
#     "./saved_experiments/2022-01-04_22:42:59.pkl")

# lstm example
# exp = load_experimentator(
#     "./saved_experiments/2022-01-04_00:20:01.pkl")

# exp.plot_predictions(0)

# exp.run_experiments(
#     experiments_path="./saved_experiments", safe=False, continue_run=True)

# TST, ResNet, LSTM
# exp = load_experimentator(
#     "./saved_experiments/2021-12-17_21:43:48.pkl")

# LSTM
# exp = load_experimentator(
#     "./saved_experiments/2021-12-16_17:19:21.pkl")

# LSTMAutoencoder
# exp = load_experimentator(
#     "./saved_experiments/2021-12-17_21:43:48.pkl")

# =============================================================================

# exp.plot_predictions(0, rescale=True)
# exp.plot_preprocessed_dataset(0, rescale=True)
# x = 0

# =============================================================================

tsm = exp.load_time_series_module(0)

# exp.plot_preprocessed_dataset(0, rescale=True, file_path="new_preprocessed.html")

# =============================================================================
# from predpy.preprocessing.statistic_anomalies_detection import *

# df = pd.concat(tsm.sequences)
# df = df.resample('1min').fillna("backfill")

# get_variance_filter(
#     df, window_size=500, log_variance_limits=(-10, -2),
#     target="Global_active_power")

# collective_isolation_forest(tsm.sequences[0].iloc[:1500, 0], 500)
# x = 0
# =============================================================================

# Prophet testing
# prophet = Prophet()
# df = pd.concat(tsm.sequences)[["Global_active_power"]]
# df = df.reset_index().rename(
#     columns={"datetime": "ds", "Global_active_power": "y"})
# prophet.fit(df)


# with open("./prophet.pkl", "rb") as file:
#     prophet = pickle.load(file)
# tmp = prophet.make_future_dataframe(periods=10)
# tmp = prophet.predict(tmp)

# =============================================================================

# from sklearn.ensemble import IsolationForest

# df = pd.concat(tsm.sequences)[["Global_active_power"]]
# gap = (df['Global_active_power'].values.reshape(-1,1))
# model_isoforest = IsolationForest()
# model_isoforest.fit(gap)
# scores = model_isoforest.score_samples(gap)
# df['anomaly_scores'] = model_isoforest.score_samples(gap)
# df['anomaly_classification'] = model_isoforest.predict(
#     df['Global_active_power'].values.reshape(-1,1))
# x = 0

# =============================================================================

normal_dfs = tsm.get_data_from_range(start=-10000, end=-3000, copy=True)
anomaly_dfs = tsm.get_data_from_range(start=-3000, copy=True)

apply_noise_on_dataframes(
    anomaly_dfs, make_noise=white_noise, negativity="abs", loc=1, scale=0.35)

# model = exp.load_pl_model(
#     model_idx=1,
#     dir_path="./checkpoints/household_power_consumption/LSTMAutoencoder_h400_l2"
# ).model

# ad = ReconstructionAnomalyDetector(
#     model, target_cols_ids=tsm.target_cols_ids())




model2 = exp.load_pl_model(
    model_idx=0,
    dir_path="./checkpoints/household_power_consumption/LSTMPAE_h200_l1"
)

# ad2 = ReconstructionAnomalyDetector(
# ad2 = EmbeddingAnomalyDetector(
ad2 = ReconstructionDistributionAnomalyDetector(
    model2, target_cols_ids=tsm.target_cols_ids())


ad2.fit(
    train_data=MultiTimeSeriesDataloader(
        normal_dfs, tsm.window_size, tsm.target,
        batch_size=tsm.batch_size),
    anomaly_data=MultiTimeSeriesDataloader(
        anomaly_dfs, tsm.window_size, tsm.target,
        batch_size=tsm.batch_size),
    normal_data=MultiTimeSeriesDataloader(
        normal_dfs, tsm.window_size, tsm.target,
        batch_size=tsm.batch_size),
    # class_weight=None,
    verbose=True, plot_time_series=True,
    # plot_embeddings=True,
)

# ad2.find_anomalies(tsm.test_dataloader())

# model = exp.load_pl_model(
#     model_idx=0,
#     dir_path="./checkpoints/household_power_consumption/LSTM_h400_l4"
# )

# ad = PredictionAnomalyDetector(model)

# ad.fit(
#     train_data=MultiTimeSeriesDataloader(
#         normal_dfs, tsm.window_size, tsm.target,
#         batch_size=tsm.batch_size),
#     anomaly_data=MultiTimeSeriesDataloader(
#         anomaly_dfs, tsm.window_size, tsm.target,
#         batch_size=tsm.batch_size),
#     normal_data=MultiTimeSeriesDataloader(
#         normal_dfs, tsm.window_size, tsm.target,
#         batch_size=tsm.batch_size),
#     class_weight=None, verbose=True, plot_time_series=True
# )

# ad.find_anomalies(tsm.test_dataloader())

x = 0

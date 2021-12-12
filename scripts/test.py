# flake8: noqa

import sys
sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import TimeSeriesRecordsDataset
from predpy.dataset import SingleTimeSeriesDataset, MultiTimeSeriesDataset
from predpy.data_module import MultiTimeSeriesModule
from predpy.experimentator import (
    DatasetParams, ModelParams, ExperimentatorPlot,
    Experimentator, load_experimentator, plot_aggregated_predictions)
from predpy.preprocessing import set_index
from predpy.preprocessing import moving_average
from predpy.preprocessing import (
    load_and_preprocess, set_index, moving_average, drop_if_is_in,
    use_dataframe_func, loc, iloc)
from predpy.trainer import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)
from tsad.noiser import apply_noise_on_dataframes, white_noise

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd
from torch.utils.data import DataLoader, Dataset


# # First experiment
# # ================

# datasets_params = [
#     DatasetParams(
#         path="../data/Meteorology/daily-min-temperatures.csv",
#         target="Temp",
#         split_proportions=[0.8, 0.1, 0.1],
#         window_size=366,
#         batch_size=64,
#         DatasetCls=SingleTimeSeriesDataset,
#         pipeline=[
#             (set_index, {"column_name": "Date"}),
#             (scale, {"training_fraction": 0.8, "scaler": MinMaxScaler()}),
#             (moving_average, {"window_size": 20, "col_names": ["Temp"]})
#         ])
# ]

# Second experiment
# ================

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
    (iloc, {"rows_end": 1500}),
    # (iloc, {"rows_start": -20000}),
]
preprocessing_pipeline = [
    (use_dataframe_func, "astype", "float"),
]
datasets_params = [
    DatasetParams(
        path="/home/stachu/Projects/Anomaly_detection/Forecasting_models/data/Energy/household_power_consumption/household_power_consumption.csv",
        load_params=load_params,
        target="Global_active_power",
        split_proportions=[0.8, 0.1, 0.1],
        window_size=window_size,
        batch_size=64,
        DatasetCls=MultiTimeSeriesDataset,
        drop_refill_pipeline=drop_refill_pipeline,
        preprocessing_pipeline=preprocessing_pipeline,
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
    ModelParams(
        name_="LSTM_h200_l1", cls_=RNN.LSTM,
        init_params={
            "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 1}),
    # ModelParams(
    #     name_="LSTM_h200_l2", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 2}),
    # ModelParams(
    #     name_="LSTM_h400_l1", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 400, "n_layers": 1}),
]
from pytorch_lightning.loggers import TensorBoardLogger

chp_p = CheckpointParams(
    dirpath="./checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=1, gpus=1, auto_lr_find=True,
    logger=TensorBoardLogger("./lightning_logs"))
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=2, verbose=True)

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger

# tmp = pl.Trainer(logger=TensorBoardLogger("./"))

# exp = Experimentator(
#     models_params=models_params,
#     datasets_params=datasets_params,
#     trainer_params=tr_p,
#     checkpoint_params=chp_p,
#     early_stopping_params=es_p
# )

# exp.run_experiments(experiments_path="./saved_experiments", safe=False)

exp = load_experimentator(
    "./saved_experiments/2021-12-11_16:34:50.pkl")

tsm = exp.load_time_series_module(0)
normal_dfs = tsm.get_data_from_range(start=-1000, end=-500, copy=True)
anomaly_dfs = tsm.get_data_from_range(start=-500, copy=True)

apply_noise_on_dataframes(
    anomaly_dfs, make_noise=white_noise, negativity="abs", loc=5, scale=1.0)

# from functools import wraps
# def anomaly_creation_wrapper(anomaly_creation):
#     @wraps
#     def create_anomaly(
#         tsm: MultiTimeSeriesModule,
#         range_: Union[
#             Tuple[int, int],
#             Literal["all", "train", "val", "test"]] = "all",
#         deal_with_negativity: str = None,
#         return_type: Union[
#             MultiTimeSeriesModule, DataLoader, Dataset,
#             List[pd.DataFrame]] = None,
#         *args, **kwargs
#     ):
#         seqs_dfs = copy_data_from_range(tsm, range_)
#         seqs_dfs = anomaly_creation(
#             seqs_dfs, deal_with_negativity, *ars, **kwargs)
#         result = seqs_to_type(seqs_dfs, type_=return_type)
#         return result
#     return create_anomaly


# =============================================================================

from tsad.anomaly_detector import PredictionAnomalyDetector

model = exp.load_pl_model(
    model_idx=0,
    dir_path="./checkpoints/household_power_consumption/LSTM_h200_l1")

ad = PredictionAnomalyDetector(model)

ad.fit(
    train_data=DataLoader(
        MultiTimeSeriesDataset(normal_dfs, tsm.window_size, tsm.target),
        batch_size=tsm.batch_size),
    anomaly_data=DataLoader(
        MultiTimeSeriesDataset(anomaly_dfs, tsm.window_size, tsm.target),
        batch_size=tsm.batch_size),
    normal_data=DataLoader(
        MultiTimeSeriesDataset(normal_dfs, tsm.window_size, tsm.target),
        batch_size=tsm.batch_size),
    class_weight=None, verbose=True, plot=True
)

ad.find_anomalies(tsm.test_dataloader())

x = 0

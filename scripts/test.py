import sys
from typing import Sequence
sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import TimeSeriesRecordsDataset
from predpy.dataset import SingleTimeSeriesDataset, MultiTimeSeriesDataset
from predpy.experimentator import Experimentator, DatasetParams, ModelParams
from predpy.preprocessing import set_index
from predpy.preprocessing import moving_average
from predpy.preprocessing import (
    load_and_preprocess, set_index, moving_average, drop_if_is_in,
    use_dataframe_func, loc, iloc)
from predpy.trainer import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd


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
    (iloc, {"rows_end": 400}),
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
        # DatasetCls=SingleTimeSeriesDataset,
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
    ModelParams(
        name_="ResNet", cls_=ResNet.ResNet,
        init_params={"c_in": c_in, "c_out": c_out}),
    # ModelParams(
    #     name_="LSTM_h200_l1", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 1}),
    # ModelParams(
    #     name_="LSTM_h200_l2", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 200, "n_layers": 2}),
    # ModelParams(
    #     name_="LSTM_h400_l1", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": c_in, "c_out": c_out, "hidden_size": 400, "n_layers": 1}),
]

# chp_p = CheckpointParams(
#     dirpath="./checkpoints", monitor='val_loss', verbose=True,
#     save_top_k=1)
# tr_p = TrainerParams(
#     max_epochs=1, gpus=1, auto_lr_find=True)
# es_p = EarlyStoppingParams(
#     monitor='val_loss', patience=2, verbose=True)

# exp = Experimentator(models_params, datasets_params)

# exp.run_experiments(
#     "./lightning_logs", tr_p, chp_p, es_p,
#     experiments_path="./saved_experiments", safe=False)

exp = Experimentator.load_experimentator(
    "saved_experiments/2021-12-04_17:06:28.pkl")

exp.plot_predictions(0, rescale=True)
x = 0

# ===========================================================================
# exp = Experimentator.load_experimentator(
#     "/home/stachu/Projects/Anomaly_detection/Forecasting_models/saved_experiments/2021-11-29_01:49:34.pkl")

# exp.change_dataset_path(0, "/home/stachu/Projects/Anomaly_detection/Forecasting_models/data/Energy/household_power_consumption/household_power_consumption.csv")

# exp.plot_predictions(0, scaler=MinMaxScaler(), file_path="predictions.html")

import sys
sys.path.append("/home/stachu/Projects/Anomaly_detection/Forecasting_models")

from predpy.dataset import TimeSeriesRecordsDataset
from predpy.dataset import SingleTimeSeriesDataset
from predpy.experimentator import Experimentator, DatasetParams, ModelParams
from predpy.preprocessing import set_index, scale
from predpy.preprocessing import moving_average
from predpy.preprocessing import (
    load_and_preprocess, set_index, scale, moving_average, drop_if_is_in,
    use_dataframe_func, select_columns)
from predpy.trainer import (
    CheckpointParams, TrainerParams, EarlyStoppingParams, LoggerParams)

import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tsai.models import TCN, ResNet, TST, RNN, TransformerModel, FCN
import pandas as pd


datasets_params = [
    DatasetParams(
        path="data/daily-min-temperatures.csv",
        target="Temp",
        split_proportions=[0.8, 0.1, 0.1],
        window_size=366,
        batch_size=64,
        DatasetCls=SingleTimeSeriesDataset,
        pipeline=[
            (set_index, {"column_name": "Date"}),
            (scale, {"training_fraction": 0.8, "scaler": MinMaxScaler()}),
            (moving_average, {"window_size": 20, "col_names": ["Temp"]})
        ])
]


# load_params = {
#     "sep": ';', "header": 0, "low_memory": False,
#     "infer_datetime_format": True, "parse_dates": {'datetime': [0, 1]},
#     "index_col": ['datetime']
# }
# pipeline = [
#     (drop_if_is_in, (["?", np.nan]), {"columns": ["Global_active_power"]}),
#     (use_dataframe_func, "astype", "float"),
#     (select_columns, ["Global_active_power"])
# ]
# datasets_params = [
#     DatasetParams(
#         path="./data/household_power_consumption.csv",
#         load_params=load_params,
#         target="Global_active_power",
#         split_proportions=[0.8, 0.1, 0.1],
#         window_size=366,
#         batch_size=64,
#         DatasetCls=SingleTimeSeriesDataset,
#         pipeline=pipeline)
# ]


# ALMOST SAME SECTION
# ===================

models_params = [
    ModelParams(
        name_="ResNet", cls_=ResNet.ResNet,
        init_params={"c_in": 2, "c_out": 1}),
    # ModelParams(
    #     name_="LSTM_h200_l1", cls_=RNN.LSTM,
    #     init_params={
    #         "c_in": 2, "c_out": 1, "hidden_size": 200, "n_layers": 1})
]

chp_p = CheckpointParams(
    dirpath="../checkpoints", monitor='val_loss', verbose=True,
    save_top_k=1)
tr_p = TrainerParams(
    max_epochs=1, gpus=1, auto_lr_find=True)
es_p = EarlyStoppingParams(
    monitor='val_loss', patience=3, verbose=True)

exp = Experimentator(models_params, datasets_params)

exp.run_experiments(
    "lightning_logs", tr_p, chp_p, es_p,
    experiments_path="saved_experiments", safe=False)

# print("true_vals: ", exp.datasets_params.iloc[0].true_values.shape[0])
# print("preds: ", exp.predictions.iloc[0]["predictions"].__len__())
exp.plot_predictions(0)

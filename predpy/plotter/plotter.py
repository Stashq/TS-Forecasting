import os
from typing import List, Tuple, Union
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn.base import TransformerMixin
import numpy as np
from datetime import timedelta, datetime
from predpy.experimentator import Experimentator
from predpy.data_module import MultiTimeSeriesModule
from predpy.plotter.experimentator_plot import ExperimentatorPlot
from predpy.wrapper import Reconstructor


PREDICTED_ANOMALIES_COLOR = '#9467bd'
TRUE_ANOMALIES_COLOR = '#d62728'
SERIES_COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]


def plot_exp_predictions(
    exp: Experimentator,
    dataset_idx: int,
    models_ids: List[int] = None,
    file_path: str = None
):
    d_params = exp.datasets_params.loc[dataset_idx]
    if models_ids is None:
        models_ids = exp.models_params.index
    ms_params = exp.models_params.loc[models_ids]
    models_names = ms_params["name_"].tolist()
    are_ae = ms_params["WrapperCls"].apply(
        lambda x: issubclass(x, Reconstructor)
    ).tolist()

    n_rows = exp.datasets_params.iloc[dataset_idx]['true_values'].shape[1]
    fig = make_subplots(rows=n_rows, cols=1)
    fig.update_layout(height=800 * n_rows, title_text=d_params.name_)

    plot_predictions(
        fig=fig,
        predictions=exp.get_models_predictions(dataset_idx, models_ids),
        true_vals=d_params.true_values,
        models_names=models_names, scaler=d_params.scaler,
        are_ae=are_ae, file_path=file_path)


def plot_predictions(
    fig,
    predictions: pd.DataFrame,
    true_vals: Union[pd.Series, pd.DataFrame],
    models_names: List[str],
    scaler: TransformerMixin = None,
    are_ae: List[bool] = None,
    file_path: str = None,
    prevent_plot: bool = False
) -> go.Figure:
    """Plots predictions made during experiment run and true values.

    If any step of experiment hasn't been made, raise AssertionError.

    Parameters
    ----------
    dataset_idx : int
        Index of parameters stored in *datasets_params*.
    models_ids : List[int], optional
        Group of index of parameters stored in *models_params*.
        If provided, plot only for selected models, if not, plot for
        all models form predictions. By default None.
    rescale : bool = False
        If True, rescale true values and predictions using scaler
        assigned to dataset.
    file_path : str, optional
        If type is string, chart will be saved to html file with provided
        path.
    """
    if scaler is not None:
        true_vals, predictions = _rescale_true_vals_and_preds(
            scaler=scaler, true_vals=true_vals, are_ae=are_ae,
            predictions_df=predictions)
    for target_i in range(true_vals.shape[1]):
        target_col = str(true_vals.columns[target_i])
        scatter_data = preds_and_true_vals_to_scatter_data(
            true_vals=true_vals, predictions=predictions, target=target_col,
            models_names=models_names, are_ae=are_ae)
        for trace in scatter_data:
            fig.add_trace(
                trace, row=target_i+1, col=1
            )
        fig.update_xaxes(title_text='date', row=target_i+1, col=1)
        fig.update_yaxes(title_text=target_col, row=target_i+1, col=1)

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot(fig, filename=file_path)
    if not prevent_plot:
        fig.show()
    return fig


def preds_and_true_vals_to_scatter_data(
    predictions: pd.DataFrame,
    true_vals: Union[pd.Series, pd.DataFrame],
    target: str,
    models_names: List[str],
    scaler: TransformerMixin = None,
    version: str = "",
    are_ae: List[bool] = None
) -> List[go.Scatter]:
    if scaler is not None:
        true_vals, predictions = _rescale_true_vals_and_preds(
            scaler=scaler, true_vals=true_vals, are_ae=are_ae,
            predictions_df=predictions, target=target)
    if are_ae is None:
        are_ae = [False]*len(predictions)

    data = ts_to_plotly_data(
        true_vals, name="true_vals", version=version, is_ae=False,
        target=target, color=SERIES_COLORS[0])

    # if passed dataframe of many models predictions
    if "predictions" in predictions.columns:
        for i, (_, row) in enumerate(predictions.iterrows()):
            data += ts_to_plotly_data(
                row["predictions"], name=models_names[i], version=version,
                is_ae=are_ae[i], target=target, color=SERIES_COLORS[i+1])
    # if passed dataframe of single model predictions
    else:
        data += ts_to_plotly_data(
            predictions, name=models_names[0], version=version,
            is_ae=are_ae[0], target=target, color=SERIES_COLORS[1])
    return data


def _rescale_true_vals_and_preds(
    scaler: TransformerMixin,
    true_vals: Union[pd.Series, pd.DataFrame],
    predictions_df: pd.DataFrame,
    are_ae: List[bool],
    target: Union[str, List[str]] = None,
) -> Tuple[List[float], pd.DataFrame]:
    vals = _scale_inverse(
        true_vals, scaler, target, is_ae=False)
    if "predictions" in predictions_df.columns:
        scaled_preds = []
        for i in range(predictions_df.shape[0]):
            preds = predictions_df.iloc[i]['predictions']
            scaled_preds +=\
                [_scale_inverse(
                    preds, scaler, target, is_ae=are_ae[i])]
        predictions = predictions_df[['dataset_id', 'model_id']]
        predictions['predictions'] = scaled_preds
    else:
        predictions = _scale_inverse(
            predictions_df, scaler, target, is_ae=are_ae[0])
    return vals, predictions


# ==================================
def ts_to_plotly_data(
    ts: Union[pd.Series, pd.DataFrame],
    name: str, color: str,
    version: str = "",
    set_gaps: bool = True,
    is_ae: bool = False,
    is_boundries: bool = False,
    target: str = None
) -> List[go.Scatter]:
    if set_gaps:
        ts = _set_gaps(ts)

    if is_boundries:
        data = _boundries_to_scatters(
            ts, name=name, version=version, color=color)
    elif isinstance(ts, pd.Series):
        data = _series_to_scatter(
            ts, name=name, version=version, color=color)
    elif isinstance(ts, pd.DataFrame) and is_ae:
        data = _reconstruction_quantiles_to_scatters(
            ts, name=name, version=version, col_name=target, color=color)
    elif isinstance(ts, pd.DataFrame) and is_ae is False:
        data = _preds_to_scatters(
            ts, name=name, version=version, target=target, color=color)
    return data


def _set_gaps(ts: Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]):
    ts = _split_ts_where_breaks(ts)
    ts = _concat_dfs_with_gaps(ts)
    return ts


def _boundries_to_scatters(
    boundries: pd.DataFrame, name: str = "Boundries", version: str = "",
    color: str = SERIES_COLORS[1]
) -> List[go.Scatter]:
    group_id = np.random.randint(9999999999, size=1)[0]
    columns = set([col[:-6] for col in boundries.columns])
    rgb = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    data = []
    for col in columns:
        data += [
            go.Scatter(
                x=boundries.index.tolist(),
                y=boundries[col + "_lower"],
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                legendgrouptitle_text=col + " boundries",
                legendgroup=str(group_id),
                name="lower"
            ),
            go.Scatter(
                x=boundries.index.tolist(),
                y=boundries[col + "_upper"],
                fill='tonexty',
                fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                legendgroup=str(group_id),
                name="upper"
            ),
        ]
    return data


def _series_to_scatter(
    ts: pd.Series, name: str, version: str = "", color: str = SERIES_COLORS[0]
) -> List[go.Scatter]:
    return [go.Scatter(
        x=ts.index, y=ts, connectgaps=False,
        name=name + version, line=dict(color=color))]


def _reconstruction_quantiles_to_scatters(
    ts: pd.Series, name: str, version: str = "", col_name: str = None,
    color: str = SERIES_COLORS[1]
) -> List[go.Scatter]:
    group_id = np.random.randint(9999999999, size=1)[0]
    if col_name is not None:
        columns = [col_name]
    else:
        columns = set([col[:-5] for col in ts.columns])
    rgb = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

    # quantile: 50%
    data = [
        go.Scatter(
            x=ts.index, y=ts[col + "_q050"], connectgaps=False,
            name=name + f"-{col}" + version,
            legendgroup=str(group_id),
            line=dict(
                color=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 1.0)'))
        for col in columns
    ]

    # quantiles: 25% - 75%
    for col in columns:
        data += [
            go.Scatter(
                x=ts.index.tolist(),
                y=ts[col + "_q025"],
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group_id),
                name="25%"
            ),
            go.Scatter(
                x=ts.index.tolist(),
                y=ts[col + "_q075"],
                fill='tonexty',
                fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group_id),
                name="75%"
            ),
        ]

    # quantiles: 0% - 100%
    for col in columns:
        data += [
            go.Scatter(
                x=ts.index.tolist(),
                y=ts[col + "_q000"],
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group_id),
                name="0%"
            ),
            go.Scatter(
                x=ts.index.tolist(),
                y=ts[col + "_q100"],
                fill='tonexty',
                fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=str(group_id),
                name="100%"
            ),
        ]
    return data


def _preds_to_scatters(
    ts: pd.Series, name: str, version: str = "", target: str = None,
    color: str = SERIES_COLORS[1]
) -> List[go.Scatter]:
    columns = ts.columns.tolist()
    if target is not None:
        columns = list(filter(lambda x: str(x) == str(target), columns))

    scatter_data = []
    for col in columns:
        scatter_data += [
            go.Scatter(
                x=ts.index, y=ts[col], connectgaps=False,
                name=name + f"-{col}" + version, line=dict(color=color))
        ]
    return scatter_data


def _concat_dfs_with_gaps(
    dfs: Union[List[pd.Series], List[pd.DataFrame]],
) -> pd.DataFrame:
    dfs_with_gaps = []
    if isinstance(dfs[0], pd.DataFrame):
        for df in dfs:
            dfs_with_gaps += [
                df, pd.DataFrame(data=[df.iloc[0]], index=[''])]
    else:
        for df in dfs:
            dfs_with_gaps += [
                df,
                pd.Series(data=[df.iloc[0]], index=[''], name=df.name)]
    return pd.concat(dfs_with_gaps)


def _split_ts_where_breaks(
    time_series: Union[pd.Series, pd.DataFrame],
    max_break: Union[int, timedelta] = None
) -> Union[pd.Series, pd.DataFrame]:
    index = time_series.index.to_series()
    diffs = index.diff()

    if max_break is None:
        max_break = diffs.mode()[0]
    elif isinstance(max_break, int):
        time_step = diffs.mode()[0]
        max_break *= time_step

    splits = diffs[diffs > max_break].index
    if splits.shape[0] == 0:
        splitted_time_series = [time_series]
    else:
        index = time_series.index
        splitted_time_series = [
            time_series.iloc[:index.get_loc(splits[0])]
        ]
        splitted_time_series += [
            time_series.iloc[
                index.get_loc(splits[i]):index.get_loc(splits[i+1])]
            for i in range(len(splits)-1)
        ]
        splitted_time_series += [
            time_series.iloc[
                index.get_loc(splits[-1]):]
        ]

    return splitted_time_series


# ==============================================================
def plot_preprocessed_dataset(
    tsm: MultiTimeSeriesModule,
    title: str = "Preprocessed dataset",
    scaler: TransformerMixin = None,
    file_path: str = None,
    start: int = None,
    end: int = None
):
    ts_with_gaps = []
    train_end = tsm.train_range[1]
    val_start, val_end = tsm.val_range
    test_start = tsm.test_range[0]

    train_end_add = 0
    val_start_add, val_end_add = 0, 0
    test_start_add = 0

    start_add = 0
    end_add = 0
    counter = 0
    if start is None:
        start = 0
    if end is None:
        end = tsm.test_range[1] + 1 + tsm.window_size

    for ts in tsm.sequences:
        ts_with_gaps += [ts]
        # adding gap
        ts_with_gaps += [pd.DataFrame(
            data=[ts.iloc[-1].to_dict()], index=[''])]

        # take into account gap
        counter += ts.shape[0]

        if train_end >= counter:
            train_end_add += 1

        if val_start >= counter:
            val_start_add += 1
        if val_end >= counter:
            val_end_add += 1

        if test_start >= counter:
            test_start_add += 1

        if start >= counter:
            start_add += 1
        if end > counter:
            end_add += 1

    train_end += train_end_add + tsm.window_size

    val_start += val_start_add
    val_end += val_end_add + tsm.window_size

    test_start += test_start_add

    end += end_add
    start += start_add

    # time series dataframe concatenation
    df = pd.concat(ts_with_gaps)
    df = df.iloc[start:end]

    if scaler is not None:
        df[df.columns] = _scale_inverse(df, scaler)

    data = []
    for col in df.columns:
        data += [go.Scatter(
            x=df.index, y=df[col], name=col, connectgaps=False)]

    layout = go.Layout(
        title=title,
        yaxis=dict(title="values"),
        xaxis=dict(title='dates')
    )

    fig = go.Figure(data=data, layout=layout)

    # adding v-lines splitting train, val and test datasets
    start = df.index[start]
    end = df.index[end-1]
    _add_vrects(
        fig, [(df.index[0], df.index[train_end])], start, end,
        fillcolor="blue", opacity=0.3, layer="below", line_width=1,
        annotation_text="train")
    _add_vrects(
        fig, [(df.index[val_start], df.index[val_end])], start, end,
        fillcolor="yellow", opacity=0.3, layer="below", line_width=1,
        annotation_text="validation")
    _add_vrects(
        fig, [(df.index[test_start], df.index[-1])], start, end,
        fillcolor="red", opacity=0.3, layer="below", line_width=1,
        annotation_text="test")

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot(fig, filename=file_path)
    else:
        fig.show()


def _scale_inverse(
    time_series: Union[pd.Series, pd.DataFrame],
    scaler: TransformerMixin,
    target_name: Union[str, List[str]] = None,
    is_ae: bool = False
):
    if target_name is not None and isinstance(time_series, pd.Series):
        cols_ids = _get_target_columns_ids(target_name, scaler)
        scaler_input = np.array(
            [time_series.tolist()] * scaler.n_features_in_).T
        result = scaler.inverse_transform(scaler_input)
        result = result.T[cols_ids]
    elif target_name is not None and isinstance(time_series, pd.DataFrame):
        cols_ids = _get_target_columns_ids(target_name, scaler)
        mocked_column = [0] * time_series.shape[0]
        scaler_input = []

        for scaler_feature in scaler.get_feature_names_out():
            if scaler_feature in time_series.columns:
                scaler_input += [time_series[scaler_feature].tolist()]
            else:
                scaler_input += [mocked_column]
        scaler_input = np.array(scaler_input).T

        # if is_ae:
        #     result = pd.DataFrame()
        #     for quantile in ['q000', 'q025', 'q050', 'q075', 'q100']:
        #         cols = [
        #             col for col in time_series.columns if quantile in col]
        #         result[cols] = scaler.inverse_transform(time_series[cols])
        # else:
        #     result = scaler.inverse_transform(time_series)
        result = scaler.inverse_transform(scaler_input)
        result = result.T[cols_ids]
    elif target_name is None and isinstance(time_series, pd.DataFrame):
        ids = time_series.index
        result = pd.DataFrame(columns=time_series.columns)
        if is_ae:
            for quantile in ['q000', 'q025', 'q050', 'q075', 'q100']:
                cols = [col for col in time_series.columns if quantile in col]
                result[cols] = scaler.inverse_transform(time_series[cols])
        else:
            result[time_series.columns] = scaler.inverse_transform(time_series)
        result.set_index(ids, inplace=True)

    return result


def _add_vrects(
    fig: go.Figure,
    intervals: List[Tuple[Union[datetime, int]]],
    start: Union[datetime, int],
    end: Union[datetime, int],
    **kwargs
):
    for inter in intervals:
        x0, x1 = inter
        if x0 < start:
            x0 = start
        if x1 > end:
            x1 = end
        if x0 < x1:
            fig.add_vrect(x0=x0, x1=x1, **kwargs)


def _get_target_columns_ids(
    target_name: Union[str, List[str]],
    scaler: TransformerMixin
) -> List[int]:
    result = []
    if isinstance(target_name, str):
        idx = np.where(scaler.get_feature_names_out() == target_name)[0][0]
        result = [idx]
    elif isinstance(target_name, list):
        for t in target_name:
            idx = np.where(scaler.get_feature_names_out() == t)[0][0]
            result += [idx]
    return result


# ==============================================================
def plot_aggregated_predictions(
    exps_params:
        Union[List[Experimentator], List[ExperimentatorPlot]],
    file_path: str = None
):
    """Plots selected prediction from list of experimentator.
    To point which predictions should be plotted, use ExperimentatorPlot.
    ExperimentatorPlot without set *datasets_to_models* cause all
    experimentator predictions plotting, the same as passing list of
    *Experimentator* instead of list of *ExperimentatorPlot*.

    Parameters
    ----------
    exps : Union[List[Experimentator], List[ExperimentatorPlot]]
        List of experimentators or list of experimentators plot params.
    file_path : str, optional
        If type is string, chart will be saved to html file with provided
        path.
    """
    data = []

    for exp_params in exps_params:
        exp, rescale, datasets_to_models = None, None, None
        if isinstance(exp_params, Experimentator):
            exp = exp_params
            rescale = True
            datasets_to_models = []
        else:
            exp = exp_params.experimentator
            datasets_to_models = exp_params.datasets_to_models
            rescale = exp_params.rescale

        if len(datasets_to_models) == 0:
            datasets_to_models = {
                ds_idx: exp.models_params.index.tolist()
                for ds_idx in exp.datasets_params.index.tolist()
            }
        for dataset_idx, models_ids in datasets_to_models.items():
            predictions_df = exp.get_models_predictions(
                dataset_idx, models_ids)
            ds_params = exp.datasets_params.iloc[dataset_idx]
            target = ds_params.target

            if rescale:
                scaler = ds_params.scaler

            m_params = exp.models_params.loc[models_ids]
            models_names = m_params["name_"].tolist()
            are_ae = m_params["WrapperCls"].apply(
                lambda x: issubclass(x, Reconstructor)
            ).tolist()
            data += preds_and_true_vals_to_scatter_data(
                true_vals=ds_params.true_values, predictions=predictions_df,
                models_names=models_names, target=target, are_ae=are_ae,
                version=f", exp: {exp.exp_date}, ds: {dataset_idx}",
                scaler=scaler
            )

    layout = go.Layout(
        title=ds_params.name_,
        yaxis=dict(title=target),
        xaxis=dict(title='dates')
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot(fig, filename=file_path)
    else:
        fig.show()


# ==============================================================

# def plot_anomalies(
#     time_series: Union[pd.Series, pd.DataFrame],
#     pred_anomalies: Union[pd.Series, pd.DataFrame] = None,
#     pred_anomalies_intervals: List[Tuple] = None,
#     true_anomalies: Union[pd.Series, pd.DataFrame] = None,
#     true_anomalies_intervals: Union[pd.Series, pd.DataFrame] = None,
#     predictions: pd.DataFrame = None,
#     detector_boundries: pd.DataFrame = None,
#     is_ae: bool = False,
#     title: str = "Finding anomalies",
#     file_path: str = None
# ):
#     data = ts_to_plotly_data(time_series, "True values")

#     if pred_anomalies is not None:
#         data += pandas_to_scatter(
#             vals=pred_anomalies, color=PREDICTED_ANOMALIES_COLOR,
#             label="predicted anomalies")
#     if predictions is not None:
#         data += ts_to_plotly_data(predictions, "Predictions", is_ae=is_ae)
#     if true_anomalies is not None:
#         data += pandas_to_scatter(
#             vals=pred_anomalies, color=TRUE_ANOMALIES_COLOR,
#             label="true anomalies"
#         )
#     if detector_boundries is not None:
#         data += ts_to_plotly_data(
#             detector_boundries, "Boundries", is_boundries=True)
#     layout = go.Layout(
#         title=title,
#         yaxis=dict(title='values'),
#         xaxis=dict(title='dates')
#     )

#     fig = go.Figure(data=data, layout=layout)
#     if pred_anomalies_intervals is not None:
#         _add_vrects(
#             fig, pred_anomalies_intervals,
#             start=time_series.index[0], end=time_series.index[-1],
#             fillcolor=PREDICTED_ANOMALIES_COLOR,
#             opacity=0.3, layer="below", line_width=1
#         )
#     if true_anomalies_intervals is not None:
#         _add_vrects(
#             fig, true_anomalies_intervals,
#             start=time_series.index[0], end=time_series.index[-1],
#             fillcolor=TRUE_ANOMALIES_COLOR,
#             opacity=0.3, layer="below", line_width=1
#         )
#     if file_path is not None:
#         plot(fig, filename=file_path)
#     else:
#         fig.show()

def plot_anomalies(
    time_series: Union[pd.Series, pd.DataFrame],
    predictions: pd.DataFrame = None,
    pred_anomalies_intervals: List[Tuple] = None,
    true_anomalies_intervals: List[Tuple] = None,
    scaler: TransformerMixin = None,
    is_ae: bool = True,
    title: str = "Finding anomalies",
    model_name: str = "Model",
    scores_df: pd.DataFrame = None,
    file_path: str = None
):
    n_rows = 1
    if predictions is not None:
        n_rows = time_series.shape[1]
    if scores_df is not None:
        n_rows += 1
    fig = make_subplots(rows=n_rows, cols=1)
    fig.update_layout(
        height=800 * n_rows,
        title_text=title)

    # plotting
    if predictions is not None:
        plot_predictions(
            fig, predictions=predictions, true_vals=time_series,
            models_names=[model_name], scaler=scaler,
            are_ae=[is_ae], prevent_plot=True)
    else:
        trace = ts_to_plotly_data(
            ts=time_series, name='', color=SERIES_COLORS[0]
        )[0]
        fig.add_trace(
            trace, row=1, col=1
        )
    if scores_df is not None:
        _add_detection_scores(
            fig=fig, scores_df=scores_df, row_id=n_rows, col_id=1)
    if pred_anomalies_intervals is not None:
        _add_vrects(
            fig, pred_anomalies_intervals,
            start=time_series.index[0], end=time_series.index[-1],
            fillcolor=PREDICTED_ANOMALIES_COLOR,
            opacity=0.3, layer="below", line_width=1
        )
    if true_anomalies_intervals is not None:
        _add_vrects(
            fig, true_anomalies_intervals,
            start=time_series.index[0], end=time_series.index[-1],
            fillcolor=TRUE_ANOMALIES_COLOR,
            opacity=0.3, layer="below", line_width=1
        )
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot(fig, filename=file_path)
    fig.show()


def get_cls_ids_ranges(
    series_index: pd.Index, classes: np.ndarray,
    max_break: Union[int, timedelta] = None,
    shift: int = None
) -> Union[pd.Series, pd.DataFrame]:
    series_index = series_index[-len(classes):]
    ids = series_index[np.where(classes == 1)[0]]

    ranges = get_ids_ranges(ids, max_break=max_break)
    if shift is not None:
        ranges = shift_ranges(ranges, shift)

    return ranges


def shift_ranges(ranges: List[Tuple[int]], shift: int):
    shifted_ranges = [
        (start + shift, end + shift)
        for start, end in ranges
    ]
    return shifted_ranges


def get_ids_ranges(
    ids: List,
    max_break: Union[int, timedelta] = None
) -> List[Tuple[int]]:
    if len(ids) == 0:
        return []
    ids = pd.Series(ids).sort_values()
    diffs = ids.diff()

    if max_break is None:
        max_break = diffs.mode()[0]
    elif isinstance(max_break, int):
        time_step = diffs.mode()[0]
        max_break *= time_step

    splits = diffs[diffs > max_break].index
    if splits.shape[0] == 0:
        splitted_time_series = [ids]
    else:
        index = ids.index
        splitted_time_series = [
            ids.iloc[:index.get_loc(splits[0])]
        ]
        splitted_time_series += [
            ids.iloc[
                index.get_loc(splits[i]):index.get_loc(splits[i+1])]
            for i in range(len(splits)-1)
        ]
        splitted_time_series += [
            ids.iloc[
                index.get_loc(splits[-1]):]
        ]

    ranges = [
        (ts.iloc[0], ts.iloc[-1])
        for ts in splitted_time_series
    ]

    return ranges


def _add_detection_scores(
    fig, scores_df: pd.DataFrame, row_id: int, col_id: int = 1
):
    scatter_data = []
    for score_col in scores_df.columns:
        group_id = str(np.random.randint(9999999999, size=1)[0])
        if score_col == 'class':
            continue
        scores_0 = scores_df[score_col][scores_df['class'] == 0]
        scores_1 = scores_df[score_col][scores_df['class'] == 1]
        scatter_data += [
            go.Scatter(
                x=scores_0.index.tolist(), y=scores_0.tolist(),
                name=str(score_col), mode='markers', legendgroup=group_id,
                legendgrouptitle_text=score_col, marker=dict(
                    color='green')),
            go.Scatter(
                x=scores_1.index.tolist(), y=scores_1.tolist(),
                name=str(score_col), mode='markers', legendgroup=group_id,
                marker=dict(
                    color='red')),
        ]
    for trace in scatter_data:
        fig.add_trace(
            trace, row=row_id, col=1
        )
    fig.update_xaxes(title_text='date', row=row_id, col=col_id)
    fig.update_yaxes(title_text='scores', row=row_id, col=col_id)
    return fig


# ==============================================================
def pandas_to_scatter(
    vals: Union[pd.DataFrame, pd.Series],
    color: str = '#d5c915',
    label: str = ""
) -> List[go.Scatter]:
    if isinstance(vals, pd.DataFrame):
        res = [
            go.Scatter(
                x=vals.index,
                y=vals[col],
                mode='markers', name=col + " " + label,
                marker=dict(
                    line=dict(width=5, color=color),
                    symbol='x-thin'))
            for col in vals.columns
        ]
    elif isinstance(vals, pd.Series):
        res = [
            go.Scatter(
                x=vals.index,
                y=vals,
                mode='markers', name=vals.name + " " + label,
                marker=dict(
                    line=dict(width=5, color=color),
                    symbol='x-thin'))
        ]
    else:
        raise ValueError(
            "Expected dataframe or series, got %s" % str(type(vals)))
    return res


# ==============================================================
def plot_3d_embeddings(
    embs_3d: np.ndarray,
    classes: List[int] = None,
    title: str = "Reconstructor embeddings",
    file_path: str = None
):
    if len(embs_3d.shape) != 2 or embs_3d.shape[1] != 3:
        raise ValueError(
            "Wrong embeddings shape. Expected (n, 3), got "
            + str(embs_3d.shape))
    classes = np.array(classes) == 0

    data = [
        go.Scatter3d(
            x=embs_3d[classes, 0].tolist(),
            y=embs_3d[classes, 1].tolist(),
            z=embs_3d[classes, 2].tolist(),
            mode='markers',
            name="normal",
            marker=dict(
                size=12,
                opacity=0.7
            )),
        go.Scatter3d(
            x=embs_3d[~classes, 0].tolist(),
            y=embs_3d[~classes, 1].tolist(),
            z=embs_3d[~classes, 2].tolist(),
            mode='markers',
            name="anomalies",
            marker=dict(
                size=12,
                opacity=0.7
            ))]

    layout = go.Layout(
        title=title,
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plot(fig, filename=file_path)
    else:
        fig.show()

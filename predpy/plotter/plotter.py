from typing import List, Tuple, Union
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn.base import TransformerMixin
import numpy as np
from datetime import timedelta, datetime
from predpy.experimentator import Experimentator
from predpy.data_module import MultiTimeSeriesModule
from .experimentator_plot import ExperimentatorPlot
from predpy.wrapper import Autoencoder


def _series_to_scatter(
    ts: pd.Series, name: str, version: str = ""
) -> List[go.Scatter]:
    return [go.Scatter(
        x=ts.index, y=ts, connectgaps=False,
        name=name + version)]


def _reconstruction_quantiles_to_scatters(
    ts: pd.Series, name: str, version: str = ""
) -> List[go.Scatter]:
    group_id = np.random.randint(9999999999, size=1)[0]
    columns = set([col[:-5] for col in ts.columns])
    rgb = np.random.randint(256, size=3)

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
    ts: pd.Series, name: str, version: str = ""
) -> List[go.Scatter]:
    return [
        go.Scatter(
            x=ts.index, y=ts[col], connectgaps=False,
            name=name + f"-{col}" + version)
        for col in ts.columns
    ]


def ts_to_plotly_data(
    ts: Union[pd.Series, pd.DataFrame],
    name: str,
    version: str = "",
    set_gaps: bool = True,
    is_ae: bool = False
) -> List[go.Scatter]:
    if set_gaps:
        ts = _set_gaps(ts)

    if isinstance(ts, pd.Series):
        data = _series_to_scatter(ts, name=name, version=version)
    elif isinstance(ts, pd.DataFrame) and is_ae:
        data = _reconstruction_quantiles_to_scatters(
            ts, name=name, version=version)
    elif isinstance(ts, pd.DataFrame) and is_ae is False:
        data = _preds_to_scatters(ts, name=name, version=version)
    return data


def _rescale_true_vals_and_preds(
    scaler: TransformerMixin,
    true_vals: Union[pd.Series, pd.DataFrame],
    predictions_df: pd.DataFrame,
    target: Union[str, List[str]]
) -> Tuple[List[float], pd.DataFrame]:
    vals = _scale_inverse(
        true_vals, scaler, target)
    predictions = \
        predictions_df["predictions"].apply(
            lambda preds: _scale_inverse(
                preds, scaler, target))
    return vals, predictions


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
    _add_dataset_type_vrect(
        fig, df.index[0], df.index[train_end], start, end,
        fillcolor="blue", opacity=0.3, layer="below", line_width=1,
        annotation_text="train")
    _add_dataset_type_vrect(
        fig, df.index[val_start], df.index[val_end], start, end,
        fillcolor="yellow", opacity=0.3, layer="below", line_width=1,
        annotation_text="validation")
    _add_dataset_type_vrect(
        fig, df.index[test_start], df.index[-1], start, end,
        fillcolor="red", opacity=0.3, layer="below", line_width=1,
        annotation_text="test")

    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        fig.show()


def _get_target_columns_ids(
    target_name: Union[str, List[str]],
    scaler: TransformerMixin
) -> List[int]:
    result = []
    if isinstance(target_name, str):
        idx = np.where(scaler.feature_names_in_ == target_name)[0][0]
        result = [idx]
    elif isinstance(target_name, list):
        for t in target_name:
            idx = np.where(scaler.feature_names_in_ == t)[0][0]
            result += [idx]
    return result


def _scale_inverse(
    time_series: Union[pd.Series, pd.DataFrame],
    scaler: TransformerMixin,
    target_name: Union[str, List[str]] = None
) -> List[float]:
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

        for scaler_feature in scaler.feature_names_in_:
            if scaler_feature in time_series.columns:
                scaler_input += [time_series[scaler_feature].tolist()]
            else:
                scaler_input += [mocked_column]
        scaler_input = np.array(scaler_input).T

        result = scaler.inverse_transform(scaler_input)
        result = result.T[cols_ids]
    elif target_name is None and isinstance(time_series, pd.DataFrame):
        result = scaler.inverse_transform(time_series)

    return result


def _add_dataset_type_vrect(
    fig: go.Figure,
    x0: Union[datetime, int],
    x1: Union[datetime, int],
    start: Union[datetime, int],
    end: Union[datetime, int],
    **kwargs
):
    if x0 < start:
        x0 = start
    if x1 > end:
        x1 = end
    if x0 < x1:
        fig.add_vrect(x0=x0, x1=x1, **kwargs)


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
        _rescale_true_vals_and_preds(
            scaler=scaler, true_vals=true_vals,
            predictions_df=predictions, target=target)
    if are_ae is None:
        are_ae = [False]*len(predictions)

    data = ts_to_plotly_data(
        true_vals, "true_vals", version, is_ae=False)

    for i, (_, row) in enumerate(predictions.iterrows()):
        data += ts_to_plotly_data(
            row["predictions"], models_names[i], version,
            is_ae=are_ae[i])
    return data


def _set_gaps(ts: Union[pd.Series, pd.DataFrame, List[pd.DataFrame]]):
    ts = _split_ts_where_breaks(ts)
    ts = _concat_dfs_with_gaps(ts)
    return ts


def plot_predictions(
    predictions: pd.DataFrame,
    true_vals: Union[pd.Series, pd.DataFrame],
    target: str,
    models_names: List[str],
    scaler: TransformerMixin = None,
    are_ae: List[bool] = None,
    title: str = "Predictions",
    file_path: str = None
):
    """Plots selected prediction made during experiment run and true values.

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
    data = preds_and_true_vals_to_scatter_data(
        true_vals=true_vals, predictions=predictions, target=target,
        models_names=models_names, scaler=scaler, are_ae=are_ae)
    layout = go.Layout(
        title=title,
        yaxis=dict(title=target),
        xaxis=dict(title='dates')
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        fig.show()


def plot_exp_predictions(
    exp: Experimentator,
    dataset_idx: int,
    models_ids: List[int] = None,
    file_path: str = None
):
    d_params = exp.datasets_params.loc[dataset_idx]
    ms_params = exp.models_params.loc[models_ids]
    models_names = ms_params["name_"].tolist()
    are_ae = ms_params["WrapperCls"].apply(
        lambda x: issubclass(x, Autoencoder)
    ).tolist()
    plot_predictions(
        predictions=exp.get_models_predictions(dataset_idx, models_ids),
        true_vals=d_params.true_vals, target=d_params.target,
        models_names=models_names, scaler=d_params.scaler,
        title=d_params.name_, are_ae=are_ae, file_path=file_path)


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
                lambda x: issubclass(x, Autoencoder)
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
        plot(fig, filename=file_path)
    else:
        fig.show()


def plot_anomalies(
    time_series: Union[pd.Series, pd.DataFrame],
    pred_anomalies: Union[pd.Series, pd.DataFrame],
    true_anomalies: Union[pd.Series, pd.DataFrame] = None,
    predictions: pd.DataFrame = None,
    # target: str = None,
    # scaler: TransformerMixin = None,
    is_ae: bool = False,
    title: str = "Finding anomalies",
    file_path: str = None
):
    data = ts_to_plotly_data(time_series, "True values")
    data += _anomalies_to_scatter(
        anomalies=pred_anomalies, color='#9467bd',
        label="predicted anomalies")

    if predictions is not None:
        data += ts_to_plotly_data(predictions, "Predictions", is_ae=is_ae)
    if true_anomalies is not None:
        data += _anomalies_to_scatter(
            anomalies=pred_anomalies, color='#d62728',
            label="true anomalies"
        )
    #     data += preds_and_true_vals_to_scatter_data(
    #         true_vals=time_series, predictions=predictions, target=target,
    #         models_names=[model_name], scaler=scaler, are_ae=[is_ae])
    # else:
    #     data += preds_and_true_vals_to_scatter_data(
    #         true_vals=time_series, predictions=predictions, target=target,
    #         models_names=[model_name], scaler=scaler, are_ae=[is_ae])
    layout = go.Layout(
        title=title,
        yaxis=dict(title='values'),
        xaxis=dict(title='dates')
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        fig.show()


def _anomalies_to_scatter(
    anomalies: Union[pd.DataFrame, pd.Series],
    color: str = '#9467bd',
    label: str = ""
) -> List[go.Scatter]:
    return [
        go.Scatter(
            x=anomalies.index,
            y=anomalies[col],
            mode='markers', name=col + " " + label,
            marker=dict(
                line=dict(width=5, color=color),
                symbol='x-thin'))
        for col in anomalies.columns
    ]


def plot_3d_embeddings(
    embs_3d: np.ndarray,
    classes: List[int] = None,
    title: str = "Autoencoder embeddings",
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
                # color="blue",
                # colorscale='Viridis',
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
                # color="red",
                # colorscale='Viridis',
                opacity=0.7
            ))]

    layout = go.Layout(
        title=title,
    )

    fig = go.Figure(data=data, layout=layout)
    if file_path is not None:
        plot(fig, filename=file_path)
    else:
        fig.show()
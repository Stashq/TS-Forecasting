from tsad.distributor import Distributor, Gaussian
from predpy.wrapper import ModelWrapper
from predpy.dataset import MultiTimeSeriesDataloader
from predpy.wrapper import Reconstructor
from predpy.plotter.plotter import plot_anomalies, plot_3d_embeddings

import torch
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Union, Tuple, Dict, List
from datetime import timedelta
from sklearn.linear_model import LogisticRegression
from string import Template
import plotly as pl
import plotly.graph_objs as go
# from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy.stats import norm
# import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, MultiTimeSeriesDataloader.")


class AnomalyDetector:
    def __init__(
        self,
        time_series_model: ModelWrapper,
        distributor: Distributor = Gaussian(),
    ):
        self.time_series_model = time_series_model
        self.time_series_model.eval()
        self.distributor = distributor
        self.thresholder = None

    @abstractmethod
    def forward(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        pass

    @abstractmethod
    def dataset_forward(
        self,
        data: MultiTimeSeriesDataloader,
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        pass

    def _any_forward(
        self,
        data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        verbose: bool = False,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        result = None
        if isinstance(data, torch.Tensor):
            result = self.forward(
                data["sequence"], data["label"],
                return_predictions=return_predictions)
        elif isinstance(data, MultiTimeSeriesDataloader):
            result = self.dataset_forward(
                data, verbose=verbose,
                return_predictions=return_predictions)
        else:
            raise ValueError(
                UNKNOWN_TYPE_MSG.substitute(data_type=type(data)))

        return result

    def fit_distributor(
        self,
        data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        verbose: bool = False,
        **kwargs
    ):
        result = self._any_forward(data, verbose=verbose)
        self.distributor.fit(result, verbose=verbose, **kwargs)

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        class_weight: Dict = {0: 0.5, 1: 0.5},  # {0: 0.05, 1: 0.95},
        verbose: bool = False,
        plot_distribution: bool = False,
        plot_time_series: bool = False,
        plot_embeddings: bool = False
    ):
        """Fits logistic regressor on distributor probabilities of normal
        and anomaly data.

        Parameters
        ----------
        normal_data : Union[torch.Tensor, MultiTimeSeriesDataloader]
            Data that is not anomaly.
        anomaly_data : Union[torch.Tensor, MultiTimeSeriesDataloader]
            Anomalies.
        class_weight : Dict, optional
            Normal data (0) and anomaly (1) weights,
            by default {0: 0.05, 1: 0.95}.
        """
        n_res, n_preds = self._any_forward(
            normal_data, verbose, return_predictions=True)
        a_res, a_preds = self._any_forward(
            anomaly_data, verbose, return_predictions=True)
        probs = np.concatenate([
                self.distributor.predict(n_res),
                self.distributor.predict(a_res)
            ], axis=0)
        classes = [0]*len(n_res) + [1]*len(a_res)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(probs, classes)
        pred_cls = self.thresholder.predict(probs)

        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        self._plot_fit_results(
            plot_time_series=plot_time_series,
            plot_embeddings=plot_embeddings,
            plot_distribution=False,
            n_res=n_res, a_res=a_res,
            classes=classes, pred_cls=pred_cls,
            n_data=normal_data, a_data=anomaly_data,
            model_n_ts_preds=n_preds,
            model_a_ts_preds=a_preds,
            n_embs=None, a_embs=None)

    def fit(
        self,
        train_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader] = None,
        class_weight: Dict = {0: 0.5, 1: 0.5},  # {0: 0.05, 1: 0.95},
        verbose: bool = True,
        dist_kwargs: Dict = {},
        plot_distribution: bool = False,
        plot_time_series: bool = False,
        plot_embeddings: bool = False
    ):
        self.fit_distributor(train_data, verbose=verbose, **dist_kwargs)
        self.fit_threshold(
            normal_data=normal_data, anomaly_data=anomaly_data,
            class_weight=class_weight, verbose=verbose,
            plot_distribution=plot_distribution,
            plot_time_series=plot_time_series,
            plot_embeddings=plot_embeddings)

    def find_anomalies(
        self,
        data: MultiTimeSeriesDataloader,
        classes: List[int] = None,
        return_indices: bool = False,
        verbose: bool = True,
        plot_distribution: bool = False,
        plot_time_series: bool = False,
        plot_embeddings: bool = False
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        if plot_time_series:
            res, model_preds = self.dataset_forward(
                data, verbose, return_predictions=True)
        else:
            res = self.dataset_forward(
                data, verbose, return_predictions=False)

        probs = self.distributor.predict(res)
        pred_cls = self.thresholder.predict(probs)

        self._plot_detection_results(
            plot_time_series=plot_time_series,
            plot_embeddings=plot_embeddings,
            plot_distribution=False,
            data=data,
            res=res,
            pred_cls=pred_cls,
            model_ts_preds=model_preds,
            embs=None)
        if return_indices is True:
            pred_cls = np.argwhere(pred_cls == 1)
        return pred_cls

    def plot_gaussian_result(
        self,
        vals: np.ndarray,
        classes: List[int] = None,
        title: str = None,
        file_path: str = None
    ):
        if self.distributor is not Gaussian:
            print("Can not plot with distributor other than Gaussian")
            return None
        n_dims = self.distributor._n_dims
        fig = make_subplots(rows=n_dims, cols=1)
        pdf = self.distributor.pdf(vals)

        dist_x = []
        threshold = []
        for dim in range(self.distributor._n_dims):
            mean = self.distributor._means[dim]
            var = self.distributor._vars[dim]
            dist_x += [np.linspace(0, mean + 2 * var, num=1000)]

            th =\
                -self.thresholder.intercept_[dim]\
                / self.thresholder.coef_[0, dim]
            threshold += [float(
                norm.ppf(th, mean, var))]
        pos = np.vstack(dist_x).T
        dist_y = self.distributor.pdf(pos)

        for dim in range(n_dims):
            if classes is not None:
                normal_ids = np.argwhere(np.array(classes) == 0).T[0]
                anomaly_ids = np.argwhere(np.array(classes) == 1).T[0]
                fig.add_trace(
                    go.Scatter(
                        name="Normal data",
                        x=vals[:, dim][normal_ids].tolist(),
                        y=pdf[:, dim][normal_ids].tolist(),
                        mode='markers',
                        line=dict(color="green")),
                    row=dim+1, col=1)
                fig.add_trace(
                    go.Scatter(
                        name="Anomaly",
                        x=vals[:, dim][anomaly_ids].tolist(),
                        y=pdf[:, dim][anomaly_ids].tolist(),
                        mode='markers',
                        line=dict(color="red")),
                    row=dim+1, col=1)
            else:
                fig.add_trace(
                    go.Scatter(
                        name="Data",
                        x=vals[:, dim][normal_ids].tolist(),
                        y=pdf[:, dim][normal_ids].tolist(),
                        mode='markers',
                        line=dict(color="green")),
                    row=dim+1, col=1)
            fig.add_trace(
                go.Scatter(
                    name="Distribution",
                    x=dist_x[0].tolist(),
                    y=dist_y[:, 0].tolist(),
                    line=dict(color="blue")),
                row=dim+1, col=1)
            fig.add_vline(
                x=threshold[dim])

        fig.update_layout(
            height=600*n_dims, width=1200,
            title_text=title)
        if file_path is not None:
            pl.offline.plot(fig, filename=file_path)
        else:
            fig.show()

    def _anomalies_ids_to_points_df(
        self,
        anom_ids: List[int],
        time_series: MultiTimeSeriesDataloader,
    ):
        if issubclass(type(self.time_series_model), Reconstructor):
            df = time_series.dataset.global_ids_to_data(
                anom_ids)
        else:
            df = time_series.dataset.get_labels().iloc[
                anom_ids]
        return df

    def _anomalies_to_intervals(
        self,
        anomalies: pd.DataFrame,
        time_series: MultiTimeSeriesDataloader,
    ):
        def get_previous(series: pd.Series, idx: Union[int, timedelta]):
            return series.iloc[series.index.get_loc(idx) - 1]
        step = time_series.dataset.sequences[0]\
            .index.to_series().diff().mode()[0]

        index = anomalies.index.to_series()
        diffs = index.diff()

        splits = diffs[diffs > step].index
        if len(anomalies) == 0:
            intervals = None
        elif splits.shape[0] == 0:
            intervals = [(index.iloc[0], index.iloc[-1])]
        else:
            intervals = [
                (index.iloc[0],
                 get_previous(index, splits[0]))]
            intervals += [
                (splits[i],
                 get_previous(index, splits[i+1]))
                for i in range(len(splits)-1)
            ]
            intervals += [(splits[-1], index.iloc[-1])]

        return intervals

    def plot_with_time_series(
        self,
        time_series: MultiTimeSeriesDataloader,
        pred_anomalies_ids: List[int],
        true_anomalies_ids: List[int] = None,
        model_preds: pd.DataFrame = None,
        detector_boundries: pd.DataFrame = None,
        anomalies_as_intervals: bool = False,
        title: str = "Finding anomalies",
        file_path: str = None
    ):
        pred_anom = self._anomalies_ids_to_points_df(
            anom_ids=pred_anomalies_ids, time_series=time_series)
        if anomalies_as_intervals:
            pred_anom_intervals = self._anomalies_to_intervals(
                anomalies=pred_anom, time_series=time_series)

        true_anom = None
        true_anom_intervals = None
        if true_anomalies_ids is not None:
            true_anom = time_series.dataset.global_ids_to_data(
                true_anomalies_ids)
            if anomalies_as_intervals:
                true_anom_intervals = self._anomalies_to_intervals(
                    anomalies=true_anom, time_series=time_series)

        ts = pd.concat(time_series.dataset.sequences)[
            time_series.dataset.target]
        plot_anomalies(
            time_series=ts,
            pred_anomalies=pred_anom,
            pred_anomalies_intervals=pred_anom_intervals,
            true_anomalies=true_anom,
            true_anomalies_intervals=true_anom_intervals,
            predictions=model_preds,
            detector_boundries=detector_boundries,
            is_ae=issubclass(type(self.time_series_model), Reconstructor),
            title=title, file_path=file_path
        )

    def plot_embeddings(
        self,
        embs: np.ndarray,
        classes: List[int] = None
    ):
        # if not issubclass(type(self.time_series_model), Reconstructor):
        #     print("Cannot plot embeddings for model\
        # other than Reconstructor.")
        pca = PCA(n_components=3)
        embs_3d = pca.fit_transform(embs)
        plot_3d_embeddings(embs_3d, classes)

    def _plot_fit_results(
        self,
        plot_time_series=False,
        plot_embeddings=False,
        plot_distribution=False,
        n_res=None, a_res=None,
        n_boundries=None, a_boundries=None,
        classes=None, pred_cls=None,
        n_data=None, a_data=None,
        model_n_ts_preds=None, model_a_ts_preds=None,
        n_embs=None, a_embs=None,
        anomalies_as_intervals=True
    ):
        if plot_time_series:
            self.plot_with_time_series(
                time_series=n_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[:len(n_res)] == 1).T[0],
                model_preds=model_n_ts_preds,
                detector_boundries=n_boundries,
                anomalies_as_intervals=anomalies_as_intervals,
                title="Detecting anomalies on normal data"
            )
            self.plot_with_time_series(
                time_series=a_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[len(n_res):] == 1).T[0],
                true_anomalies_ids=list(range(0, len(a_res))),
                model_preds=model_a_ts_preds,
                detector_boundries=a_boundries,
                anomalies_as_intervals=anomalies_as_intervals,
                title="Detecting anomalies on anomaly data"
            )
        if plot_embeddings:
            self.plot_embeddings(
                np.concatenate([n_embs, a_embs], axis=0), classes)
        if plot_distribution:
            self.plot_gaussian_result(
                vals=np.concatenate([
                    n_res, a_res]),
                classes=classes,
                title="Result of fitting")

    def _plot_detection_results(
        self,
        plot_time_series=False,
        plot_embeddings=False,
        plot_distribution=False,
        data=None,
        res=None,
        pred_cls=None,
        model_ts_preds=None,
        embs=None,
    ):
        if plot_time_series:
            self.plot_with_time_series(
                time_series=data, pred_anomalies_ids=pred_cls,
                model_preds=model_ts_preds, title="Predicted anomalies")
        if plot_embeddings:
            self.plot_embeddings(
                embs=embs, classes=pred_cls)
        if plot_distribution:
            self.plot_gaussian_result(
                vals=res, classes=pred_cls,
                title="Anomalies on distribution plot")

        # dane:
        # seria czasowa, wyniki wykrywania anomalii
        # dodatkowe:
        # seria anomalia / klasy rekordów serii czasowej, predykcje modelu
        # dalekie w realizacji:
        # zakres ufności na około predykcji modelu
        # na podstawie dystrybucji / modelu regresji
        # oraz wyliczonej przez model anomalii lub podanej przy inicjalizacji
        # wartości granicznej prawdopodobieństwa (threshold)

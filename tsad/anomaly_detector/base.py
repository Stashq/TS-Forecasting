from tsad.distributor import Distributor, Gaussian
from predpy.wrapper import TSModelWrapper
from predpy.dataset import MultiTimeSeriesDataloader
from predpy.wrapper import Autoencoder
from predpy.plotter.plotter import plot_anomalies

import torch
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import Union, Tuple, Dict, List
from sklearn.linear_model import LogisticRegression
from string import Template
import plotly as pl
import plotly.graph_objs as go
# from plotly.offline import plot
from plotly.subplots import make_subplots
from scipy.stats import norm
# import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, MultiTimeSeriesDataloader.")


class AnomalyDetector:
    def __init__(
        self,
        time_series_model: TSModelWrapper,
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
        verbose: bool = False
    ):
        result = self._any_forward(data, verbose)
        self.distributor.fit(result)

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        class_weight: Dict = {0: 0.05, 1: 0.95},
        verbose: bool = False,
        plot_distribution: bool = False,
        plot_time_series: bool = False
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
        cdf = np.concatenate(
            [self.distributor.cdf(n_res), self.distributor.cdf(a_res)],
            axis=0)
        classes = [0]*len(n_res) + [1]*len(a_res)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(cdf, classes)
        pred_cls = self.thresholder.predict(cdf)

        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        if plot_distribution:
            self.plot_gaussian_result(
                vals=np.concatenate([
                    n_res, a_res]),
                classes=classes,
                title="Result of fitting")
        if plot_time_series:
            self.plot_with_time_series(
                time_series=normal_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[:len(n_res)] == 1).T[0],
                model_preds=n_preds,
                title="Detecting anomalies on normal data"
            )
            self.plot_with_time_series(
                time_series=anomaly_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[len(n_res):] == 1).T[0],
                true_anomalies_ids=list(range(0, len(a_res))),
                model_preds=a_preds,
                title="Detecting anomalies on anomaly data"
            )

    def fit(
        self,
        train_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader] = None,
        class_weight: Dict = {0: 0.05, 1: 0.95},
        verbose: bool = True,
        plot_distribution: bool = False,
        plot_time_series: bool = False
    ):
        self.fit_distributor(train_data, verbose)
        self.fit_threshold(
            normal_data=normal_data, anomaly_data=anomaly_data,
            class_weight=class_weight, verbose=verbose,
            plot_distribution=plot_distribution,
            plot_time_series=plot_time_series)

    def find_anomalies(
        self,
        data: MultiTimeSeriesDataloader,
        classes_: List[int] = None,
        return_indices: bool = False,
        verbose: bool = True,
        plot_dist: bool = False,
        plot_time_series: bool = False
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        if plot_time_series:
            vals, model_preds = self.dataset_forward(
                data, verbose, return_predictions=True)
        else:
            vals = self.dataset_forward(
                data, verbose, return_predictions=False)

        cdf_res = self.distributor.cdf(vals)
        result = self.thresholder.predict(cdf_res)

        if plot_dist:
            self.plot_with_distribution(
                cdf_res, title="Anomalies on distribution plot")
        if plot_time_series:
            self.plot_with_time_series(
                model_preds, title="Anomalies on time series plot")
        if return_indices is True:
            result = np.argwhere(result == 1)
        return result

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

        # fig = ff.create_distplot(
        #     [preds[:, 0][normal_ids].tolist(),
        #      preds[:, dim][anomaly_ids].tolist()],
        #     pdf[:, 0].tolist(),
        #     bin_size=.02)
        # fig.show()

    def plot_with_time_series(
        self,
        time_series: MultiTimeSeriesDataloader,
        pred_anomalies_ids: List[int],
        true_anomalies_ids: List[int] = None,
        # model_preds: List[torch.Tensor] = None,
        model_preds: pd.DataFrame = None,
        title: str = "Finding anomalies",
        file_path: str = None
    ):
        if issubclass(type(self.time_series_model), Autoencoder):
            pred_anomalies = time_series.dataset.global_ids_to_data(
                pred_anomalies_ids)
        else:
            pred_anomalies = time_series.dataset.get_labels().iloc[
                pred_anomalies_ids]

        true_anomalies = None
        if true_anomalies_ids is not None:
            true_anomalies = time_series.dataset.global_ids_to_data(
                true_anomalies_ids)

        ts = pd.concat(time_series.dataset.sequences)[
            time_series.dataset.target]
        plot_anomalies(
            time_series=ts,
            pred_anomalies=pred_anomalies,
            true_anomalies=true_anomalies,
            predictions=model_preds,
            is_ae=issubclass(type(self.time_series_model), Autoencoder),
            title=title, file_path=file_path
        )
        # labels = time_series.dataset.get_labels()
        # dates = labels.index
        # target = labels.columns

        # true_series = time_series.dataset.sequences
        # if isinstance(true_series, list):
        #     true_series = pd.concat(true_series)

        # data = []
        # for col_name in target:
        #     data += [go.Scatter(
        #         x=true_series.index,
        #         y=true_series[col_name],
        #         connectgaps=False,
        #         name=col_name)]

        # for col_name in target:
        #     data += [go.Scatter(
        #         x=dates[pred_anomalies_ids],
        #         y=labels[col_name][pred_anomalies_ids],
        #         mode='markers', name="Predicted anomalies",
        #         marker=dict(
        #             line=dict(width=5, color='#9467bd'),
        #             symbol='x-thin')
        #     )]

        # if model_preds is not None:
        #     model_preds = np.array([
        #         pred.cpu().detach().tolist()
        #         for pred in model_preds
        #     ])
        #     for i, col_name in enumerate(target):
        #         data += [go.Scatter(
        #             x=dates, y=model_preds[:, i].tolist(), connectgaps=False,
        #             name=col_name + "_pred")]

        # if true_anomalies_ids is not None:
        #     for col_name in target:
        #         data += [go.Scatter(
        #             x=dates[true_anomalies_ids],
        #             y=labels[col_name][true_anomalies_ids],
        #             mode='markers', name="True anomalies",
        #             marker=dict(
        #                 line=dict(width=5, color='#d62728'),
        #                 symbol='x-thin')
        #         )]

        # layout = go.Layout(
        #     title=title,
        #     yaxis=dict(title="values"),
        #     xaxis=dict(title='dates')
        # )

        # fig = go.Figure(data=data, layout=layout)
        # if file_path is not None:
        #     plot(fig, filename=file_path)
        # else:
        #     fig.show()

        # dane:
        # seria czasowa, wyniki wykrywania anomalii
        # dodatkowe:
        # seria anomalia / klasy rekordów serii czasowej, predykcje modelu
        # dalekie w realizacji:
        # zakres ufności na około predykcji modelu
        # na podstawie dystrybucji / modelu regresji
        # oraz wyliczonej przez model anomalii lub podanej przy inicjalizacji
        # wartości granicznej prawdopodobieństwa (threshold)

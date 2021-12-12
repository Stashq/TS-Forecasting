from tsad.distributor import Distributor, GaussianDistributor

import torch
from tqdm.auto import tqdm
import numpy as np
from abc import abstractmethod
from typing import Type, Union, Tuple, Dict, List
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from string import Template
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly as pl
from scipy.stats import norm
# import plotly.figure_factory as ff


import matplotlib.pyplot as plt

from sklearn import linear_model
from scipy.special import expit
from sklearn.metrics import confusion_matrix

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, DataLoader.")


class AnomalyDetector:
    def __init__(
        self,
        time_series_model: torch.nn.Module,
        DistributorCls: Type[Distributor] = GaussianDistributor,
        **distributor_kwargs
    ):
        self.time_series_model = time_series_model
        self.time_series_model.eval()
        self.distributor = DistributorCls(**distributor_kwargs)
        self.thresholder = None

    @abstractmethod
    def forward(
        self,
        sequences: torch.Tensor,
        labels: torch.Tensor
    ) -> np.ndarray:
        pass

    # TODO: zmienić na szybszy!!
    def dataset_forward(
        self,
        data: Union[DataLoader, Dataset],
        verbose: bool = False
    ) -> np.ndarray:
        results = []
        iterator = None
        if verbose:
            iterator = tqdm(data, desc="Time series predictions")
        else:
            iterator = data

        for data in iterator:
            results += [self.forward(data["sequence"], data["label"])]
        results = np.concatenate(results, axis=0)
        return results

    def fit_distributor(
        self,
        data: Union[torch.Tensor, DataLoader, Dataset],
        verbose: bool = False
    ):
        result = self._any_forward(data, verbose)
        self.distributor.fit(result)

    def _any_forward(
        self,
        data: Union[torch.Tensor, DataLoader, Dataset],
        verbose: bool = False
    ) -> np.ndarray:
        result = None
        if isinstance(data, torch.Tensor):
            result = self.forward(data["sequence"], data["label"])
        elif isinstance(data, DataLoader) or isinstance(data, Dataset):
            result = self.dataset_forward(data, verbose=verbose)
        else:
            raise ValueError(
                UNKNOWN_TYPE_MSG.substitute(data_type=type(data)))
        return result

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, DataLoader, Dataset],
        anomaly_data: Union[torch.Tensor, DataLoader, Dataset],
        class_weight: Dict = {0: 0.05, 1: 0.95},
        verbose: bool = False,
        plot: bool = False
    ):
        """Fits logistic regressor on distributor probabilities of normal
        and anomaly data.

        Parameters
        ----------
        normal_data : Union[torch.Tensor, DataLoader, Dataset]
            Data that is not anomaly.
        anomaly_data : Union[torch.Tensor, DataLoader, Dataset]
            Anomalies.
        class_weight : Dict, optional
            Normal data (0) and anomaly (1) weights,
            by default {0: 0.05, 1: 0.95}.
        """
        normal_preds = self._any_forward(normal_data, verbose)
        anomaly_preds = self._any_forward(anomaly_data, verbose)
        cdf = np.concatenate([
            self.distributor.cdf(normal_preds),
            self.distributor.cdf(anomaly_preds)])
        classes = [0]*len(normal_preds) + [1]*len(anomaly_preds)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(cdf, classes)
        pred_cls = self.thresholder.predict(cdf)
        # plt.hist(cdf[:134], bins=20, label="normal")
        # plt.hist(cdf[134:], bins=20, label="anomaly")
        # plt.show()

        cm = confusion_matrix(classes, pred_cls)
        print(cm)
        # self.plot_logistic_regression(cdf, classes)
        if plot:
            self.plot_anomaly_detection(
                preds=np.concatenate([
                    normal_preds, anomaly_preds]),
                classes=classes,
                title="Result of fitting")

    def fit(
        self,
        train_data: Union[torch.Tensor, DataLoader, Dataset],
        anomaly_data: Union[torch.Tensor, DataLoader, Dataset],
        normal_data: Union[torch.Tensor, DataLoader, Dataset] = None,
        class_weight: Dict = {0: 0.05, 1: 0.95},
        verbose: bool = True,
        plot: bool = False
    ):
        # if normal_data is None:
        #     normal_data = np.array([
        #         seq.cpu().detach().numpy()
        #         for seq, _ in train_data.dataset
        #     ])
        self.fit_distributor(train_data, verbose)
        self.fit_threshold(
            normal_data, anomaly_data, class_weight, verbose, plot)

    def find_anomalies(
        self,
        data: Union[DataLoader, Dataset],
        return_indices: bool = True,
        verbose: bool = False,
        plot: bool = False
    ) -> Union[Tuple[np.ndarray], np.ndarray]:
        preds = self.dataset_forward(data, verbose)
        cdf = self.distributor.cdf(preds)
        result = self.thresholder.predict(cdf)
        if plot:
            self.plot_anomaly_detection(preds, title="Predictions")
        if return_indices is True:
            result = np.argwhere(result == 1)
        return result

    def plot_anomaly_detection(
        self,
        preds: np.ndarray,
        classes: List[int] = None,
        title: str = None,
        file_path: str = None
    ):
        n_dims = self.distributor._n_dims
        fig = make_subplots(rows=n_dims, cols=1)
        pdf = self.distributor.pdf(preds)

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
        x = 0
        # tmp_y = norm.pdf(dist_x[0], loc, scale)
        # fig.add_trace(
        #     go.Scatter(
        #         name="Normal distribution",
        #         x=dist_x[0].tolist(),
        #         y=tmp_y.tolist(),
        #         line=dict(color="pink")),
        #     row=dim+1, col=1)

        for dim in range(n_dims):
            if classes is not None:
                normal_ids = np.argwhere(np.array(classes) == 0).T[0]
                anomaly_ids = np.argwhere(np.array(classes) == 1).T[0]
                fig.add_trace(
                    go.Scatter(
                        name="Normal data",
                        x=preds[:, dim][normal_ids].tolist(),
                        y=pdf[:, dim][normal_ids].tolist(),
                        mode='markers',
                        line=dict(color="green")),
                    row=dim+1, col=1)
                fig.add_trace(
                    go.Scatter(
                        name="Anomaly",
                        x=preds[:, dim][anomaly_ids].tolist(),
                        y=pdf[:, dim][anomaly_ids].tolist(),
                        mode='markers',
                        line=dict(color="red")),
                    row=dim+1, col=1)
            else:
                fig.add_trace(
                    go.Scatter(
                        name="Data",
                        x=preds[:, dim][normal_ids].tolist(),
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
        x = 0

        # fig = ff.create_distplot(
        #     [preds[:, 0][normal_ids].tolist(),
        #      preds[:, dim][anomaly_ids].tolist()],
        #     pdf[:, 0].tolist(),
        #     bin_size=.02)
        # fig.show()

    def plot_logistic_regression(
        self,
        X: np.ndarray,
        y: List[int]
    ):
        # General a toy dataset:s it's just
        # a straight line with some Gaussian noise:
        # xmin, xmax = -5, 5
        # n_samples = 100
        # np.random.seed(0)
        # X = np.random.normal(size=n_samples)
        # y = (X > 0).astype(float)
        # X[X > 0] *= 4
        # X += 0.3 * np.random.normal(size=n_samples)

        # X = X[:, np.newaxis]

        # Fit the classifier
        # clf = linear_model.LogisticRegression(C=1e5)
        # clf.fit(X, y)
        clf = self.thresholder

        # and plot the result
        plt.figure(1, figsize=(4, 3))
        plt.clf()
        plt.scatter(X.ravel(), y, color="black", zorder=20)
        X_test = np.linspace(-5, 10, 300)

        loss = expit(X_test * clf.coef_ + clf.intercept_).ravel()
        plt.plot(X_test, loss, color="red", linewidth=3)

        ols = linear_model.LinearRegression()
        ols.fit(X, y)
        plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
        plt.axhline(0.5, color=".5")

        plt.ylabel("y")
        plt.xlabel("X")
        plt.xticks(range(-5, 10))
        plt.yticks([0, 0.5, 1])
        plt.ylim(-0.25, 1.25)
        plt.xlim(-4, 10)
        plt.legend(
            ("Logistic Regression Model", "Linear Regression Model"),
            loc="lower right",
            fontsize="small",
        )
        plt.tight_layout()
        plt.show()

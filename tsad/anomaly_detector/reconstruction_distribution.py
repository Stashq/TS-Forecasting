from .reconstruction import ReconstructionAnomalyDetector
from predpy.wrapper import ModelWrapper
import torch
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict
from tsad.distributor import Distributor, Gaussian
from tqdm.auto import tqdm
from predpy.dataset import MultiTimeSeriesDataloader
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm


class ReconstructionDistributionAnomalyDetector(ReconstructionAnomalyDetector):
    def __init__(
        self,
        time_series_model: ModelWrapper,
        distributor: Distributor = Gaussian(),
        target_cols_ids: List[int] = None,
        alpha: float = 0.9
    ):
        super().__init__(
            time_series_model, distributor,
            target_cols_ids=target_cols_ids)
        self.alpha = alpha

    def forward(
        self,
        records: torch.Tensor,
        return_predictions: bool = False
    ) -> Tuple[np.ndarray]:
        with torch.no_grad():
            seqs = self.get_seqences(records)
            x_mu, x_log_sig = self.time_series_model.predict(
                seqs, get_log_sig=True)
            errors = torch.abs(seqs - x_mu).cpu().detach().numpy()

        return errors, x_mu, x_log_sig

    def dataset_forward(
        self,
        dataloader: MultiTimeSeriesDataloader,
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        iterator = None
        if verbose:
            iterator = tqdm(dataloader, desc="Time series model running")
        else:
            iterator = dataloader

        x_mu, x_log_sig = [], []
        seqs = []
        for batch in iterator:
            batch_seqs = self.get_seqences(batch)
            seqs += [batch_seqs]
            with torch.no_grad():
                mu, log_sig = self.time_series_model.predict(
                    batch_seqs, get_x_log_sig=True)
                x_mu += mu
                x_log_sig += log_sig
        seqs = torch.cat(seqs, 0)
        x_mu = torch.cat(x_mu, 0)
        x_log_sig = torch.cat(x_log_sig, 0).numpy()
        if len(x_mu.shape) == 2:
            x_mu = x_mu.unsqueeze(dim=1)
        errors = torch.abs(seqs - x_mu).cpu().detach().numpy()

        if return_predictions:
            x_mu_df = self.time_series_model.preds_to_df(
                dataloader, x_mu.numpy())
            return errors, x_mu, x_log_sig, x_mu_df
        return errors, x_mu, x_log_sig

    def fit_distributor(
        self,
        data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        verbose: bool = False,
        **kwargs
    ):
        errors, x_mu, x_log_sig = self._any_forward(data, verbose)
        self.distributor.fit(errors, verbose=verbose, **kwargs)

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        class_weight: Dict = {0: 0.7, 1: 0.3},  # {0: 0.05, 1: 0.95},
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
        n_errs, n_mu, n_log_sig, n_preds = self._any_forward(
            normal_data, verbose, return_predictions=True)
        a_errs, a_mu, a_log_sig, a_preds = self._any_forward(
            anomaly_data, verbose, return_predictions=True)
        probs = np.concatenate([
                self.distributor.predict(n_errs),
                self.distributor.predict(a_errs)
            ], axis=0)
        classes = [0]*len(n_errs) + [1]*len(a_errs)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(probs, classes)
        # pred_cls = self.thresholder.predict(probs)
        pred_cls = np.zeros((len(classes),))
        pred_cls[:len(n_errs)] = self._predict_with_variance(
            normal_data, n_mu, n_log_sig, pred_cls[:len(n_errs)], verbose)
        pred_cls[len(n_errs):] = self._predict_with_variance(
            anomaly_data, a_mu, a_log_sig, pred_cls[len(n_errs):], verbose)

        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        if plot_distribution:
            self.plot_gaussian_result(
                vals=np.concatenate([
                    n_errs, a_errs]),
                classes=classes,
                title="Result of fitting")
        if plot_time_series:
            self.plot_with_time_series(
                time_series=normal_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[:len(n_errs)] == 1).T[0],
                model_preds=n_preds,
                title="Detecting anomalies on normal data"
            )
            self.plot_with_time_series(
                time_series=anomaly_data,
                pred_anomalies_ids=np.argwhere(
                    pred_cls[len(n_errs):] == 1).T[0],
                true_anomalies_ids=list(range(0, len(a_errs))),
                model_preds=a_preds,
                title="Detecting anomalies on anomaly data"
            )
        # if plot_embeddings:
        #     self.plot_embeddings(
        #         np.concatenate([n_errs, a_errs], axis=0), classes)

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

    def _predict_with_variance(
        self,
        data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        mu: np.ndarray,
        log_sig: np.ndarray,
        pred_cls: np.ndarray = None,
        verbose: bool = False
    ):
        if pred_cls is None:
            pred_cls = np.zeros((len(mu), 1))
        if verbose:
            iter_ = tqdm(data)
        else:
            iter_ = data

        counter = 0
        i = 0
        for batch in iter_:
            seqs = self.get_seqences(batch).numpy()
            for seq in seqs:
                if pred_cls[i] == 0:
                    low, high = norm.interval(
                        self.alpha, loc=mu[i], scale=np.exp(log_sig[i]))
                    res = np.logical_and(low <= seq, seq <= high).all().item()
                    if res is False:
                        pred_cls[i] = 1
                        counter += 1
                i += 1
        if verbose:
            print("Found %d new anomalies." % counter)
        return pred_cls

    def find_anomalies(
        self,
        data: MultiTimeSeriesDataloader,
        classes: List[int] = None,
        return_indices: bool = False,
        verbose: bool = True,
        plot_dist: bool = False,
        plot_time_series: bool = False,
        plot_embeddings: bool = False
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
        if plot_embeddings:
            self.plot_embeddings(
                vals, classes)
        if return_indices is True:
            result = np.argwhere(result == 1)
        return result

import numpy as np
from typing import Union, Tuple, List, Dict

from tqdm.auto import tqdm
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from .base import AnomalyDetector
from predpy.wrapper import VAE
from predpy.dataset import MultiTimeSeriesDataloader
from tsad.distributor import Distributor, Gaussian
from tsad.error_regressor import Regressor

NO_EMBEDDING_MODEL_MSG = "No embedding model found. Define error regressor "
"model or embedding distributor."
BOTH_MODELS_MSG = "Detected embedding distributor. "
"Only error regressor will be used."


class ReconstructionAndEmbeddingAnomalyDetector(AnomalyDetector):
    def __init__(
        self,
        time_series_model: VAE,
        distributor: Distributor = Gaussian(),
        target_cols_ids: List[int] = None,
        error_regressor: Regressor = None,
        regressor_coef: float = 1.0,
        embeddings_distributor: Distributor = None
    ):
        super().__init__(
            time_series_model, distributor)
        self.target_cols_ids = target_cols_ids
        self.error_regressor = error_regressor
        self.embeddings_distributor = embeddings_distributor
        self.regressor_coef = regressor_coef

    def get_seqences(self, batch):
        if next(self.time_series_model.parameters()).is_cuda:
            device = self.time_series_model.get_device()
        else:
            device = "cpu"
        if self.target_cols_ids is None:
            seqs = batch["sequence"]
        else:
            seqs = torch.index_select(
                batch["sequence"], dim=1,
                index=torch.tensor(self.target_cols_ids, device=device))
        return seqs

    def forward(
        self,
        records: torch.Tensor,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray]]:
        with torch.no_grad():
            x = self.get_seqences(records)
            z = self.time_series_model.encode(x)
            x_tilda = self.time_series_model.decode(z)
            errors = torch.abs(x - x_tilda).cpu().detach().numpy()

        if return_predictions:
            return errors, z, x_tilda
        return errors, z

    def dataset_forward(
        self,
        dataloader: MultiTimeSeriesDataloader,
        verbose: bool = True,
        return_predictions: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, pd.DataFrame]]:
        iterator = None
        if verbose:
            iterator = tqdm(dataloader, desc="Time series predictions")
        else:
            iterator = dataloader

        seqs = []
        zs = []
        preds = []
        for batch in iterator:
            x = self.get_seqences(batch)
            seqs += [x]
            with torch.no_grad():
                z = self.time_series_model.encode(x)
                zs += z
                x_tilda = self.time_series_model.decode(z)
                preds += x_tilda

        zs = torch.cat(zs, 0).cpu().detach().numpy()
        preds = torch.cat(preds, 0)
        if len(preds.shape) == 2:
            preds = preds.unsqueeze(dim=1)
        seqs = torch.cat(seqs, 0)
        errors = torch.abs(seqs - preds).cpu().detach().numpy()

        if return_predictions:
            preds = self.time_series_model.preds_to_dataframe(
                dataloader, preds.numpy())
            return errors, zs, preds
        return errors, zs

    def fit_distributor(
        self,
        data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        verbose: bool = False,
        **kwargs
    ):
        result, _ = self._any_forward(data, verbose=verbose)
        self.distributor.fit(result, verbose=verbose, **kwargs)

    def fit_threshold(
        self,
        normal_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        anomaly_data: Union[torch.Tensor, MultiTimeSeriesDataloader],
        class_weight: Dict = {0: 0.7, 1: 0.3},
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
        n_errs, n_z, n_preds = self._any_forward(
            normal_data, verbose, return_predictions=True)
        a_errs, a_z, a_preds = self._any_forward(
            anomaly_data, verbose, return_predictions=True)
        classes = [0]*len(n_errs) + [1]*len(a_errs)
        probs = self._fit_thresholders(
            n_errs, a_errs, class_weight, classes, n_z, a_z)

        err_th_pred_cls = self.thresholder.predict(probs)

        emb_th_pred_cls = np.zeros((len(classes),))
        emb_th_pred_cls[:len(n_errs)], n_pred_errs =\
            self._predict_with_embeddings(
                embs=n_z, errs=n_errs, return_pred_errs=True)
        emb_th_pred_cls[len(n_errs):], a_pred_errs =\
            self._predict_with_embeddings(
                embs=a_z, errs=a_errs, return_pred_errs=True)

        pred_cls = 1*np.logical_or(err_th_pred_cls, emb_th_pred_cls)

        # TODO porównywanie predykcji modelu błędów i modelu osadzeń
        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        self._plot_fit_results(
            plot_time_series=plot_time_series,
            plot_embeddings=plot_embeddings,
            plot_distribution=plot_distribution,
            n_res=n_errs, a_res=a_errs,
            n_pred_errs=n_pred_errs, a_pred_errs=a_pred_errs,
            classes=classes, pred_cls=pred_cls,
            n_data=normal_data, a_data=anomaly_data,
            model_n_ts_preds=n_preds,
            model_a_ts_preds=a_preds,
            n_embs=n_z, a_embs=a_z)
        # if plot_distribution:
        #     self.plot_gaussian_result(
        #         vals=np.concatenate([
        #             n_errs, a_errs]),
        #         classes=classes,
        #         title="Result of fitting")
        # if plot_time_series:
        #     self.plot_with_time_series(
        #         time_series=normal_data,
        #         pred_anomalies_ids=np.argwhere(
        #             pred_cls[:len(n_errs)] == 1).T[0],
        #         model_preds=n_preds,
        #         title="Detecting anomalies on normal data"
        #     )
        #     self.plot_with_time_series(
        #         time_series=anomaly_data,
        #         pred_anomalies_ids=np.argwhere(
        #             pred_cls[len(n_errs):] == 1).T[0],
        #         true_anomalies_ids=list(range(0, len(a_errs))),
        #         model_preds=a_preds,
        #         title="Detecting anomalies on anomaly data"
        #     )
        # if plot_embeddings:
        #     self.plot_embeddings(
        #         np.concatenate([n_z, a_z], axis=0), classes)

    def _fit_thresholders(
        self, n_errs, a_errs, class_weight, classes, n_z=None, a_z=None
    ) -> np.ndarray:
        probs = np.concatenate([
                self.distributor.predict(n_errs),
                self.distributor.predict(a_errs)
            ], axis=0)
        self.thresholder = LogisticRegression(
            class_weight=class_weight
        ).fit(probs, classes)

        if self.embeddings_distributor is not None:
            emb_probs = np.concatenate([
                    self.distributor.predict(n_z),
                    self.distributor.predict(a_z)
                ], axis=0)
            self.embeddings_thresholder = LogisticRegression(
                class_weight=class_weight
            ).fit(emb_probs, classes)

        return probs

    def _predict_with_embeddings(
        self,
        embs: np.ndarray,
        errs: np.ndarray = None,
        return_pred_errs: bool = False
    ):
        pred_errs = None
        if self.error_regressor is not None:
            classes = []
            if self.embeddings_distributor is not None:
                print(BOTH_MODELS_MSG)
            if issubclass(type(self.error_regressor), nn.Module):
                embs = torch.tensor(embs)
            pred_errs = self.error_regressor.predict(embs)
            for i, pred_err in enumerate(pred_errs):
                res = (self.regressor_coef*errs[i] < pred_err).all()
                classes += [int(res)]
            classes = np.array(classes)
        elif self.embeddings_distributor is not None:
            probs = self.embeddings_distributor.predict(embs)
            classes = self.embeddings_thresholder.predict(probs)
        else:
            raise ModuleNotFoundError(NO_EMBEDDING_MODEL_MSG)

        if return_pred_errs:
            return classes, pred_errs
        return classes

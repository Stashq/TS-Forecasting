from abc import abstractmethod
import csv
from datetime import timedelta
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from string import Template
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
import torch
from tqdm.auto import tqdm
from typing import Union, Tuple, Dict, List

from predpy.dataset import MultiTimeSeriesDataloader
from predpy.plotter.plotter import plot_anomalies
from predpy.wrapper import ModelWrapper
from predpy.wrapper import Reconstructor

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, MultiTimeSeriesDataloader.")


class AnomalyDetector(ModelWrapper):
    def __init__(
        self,
    ):
        self.thresholder = LogisticRegression()

    @abstractmethod
    def anomaly_score(self, model_output) -> float:
        pass

    def load_anom_scores(
        self, path: Path
    ) -> Tuple[List[float], List[int]]:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            scores = []
            classes = []
            for row in reader:
                scores += [float(row['score'])]
                classes += [int(row['class'])]
        return scores, classes

    def save_anom_scores(
        self,
        scores: List[float],
        classes: List[int],
        path: Path
    ):
        with open(path, 'w') as f:
            rows = [
                [scores[i], classes[i]]
                for i in range(len(scores))]
            writer = csv.writer(f)

            writer.writerow(['score', 'class'])
            writer.writerows(rows)

    def fit_thresholder(
        self,
        scores: List[float] = None,
        classes: List[int] = None,
        scaler: TransformerMixin = None,
        class_weight: Dict = {0: 0.75, 1: 0.25}
    ) -> np.ndarray:
        scores, classes = np.array(scores).reshape(-1, 1), np.array(classes)
        if scaler is not None:
            # scaling scores
            n_scores = scaler.fit_transform(
                scores[np.where(classes == 0)])
            a_scores = scaler.transform(
                scores[np.where(classes == 1)])
            # rewriting scaled scores
            scores[np.where(classes == 0)] = n_scores
            scores[np.where(classes == 1)] = a_scores

        # fitting thresholder
        self.thresholder = LogisticRegression(
            class_weight=class_weight).fit(scores, classes)
        pred_cls = self.thresholder.predict(scores)

        # printing classification results
        cm = confusion_matrix(classes, pred_cls)
        print(cm)

        return pred_cls

    def score_dataset(
        self, dataloader: MultiTimeSeriesDataloader,
        scale: bool = True, return_pred: bool = False
    ):
        scores, preds = [], []
        for batch in tqdm(
                dataloader, desc='Calculating dataset anomaly scores'):
            x = batch['sequence']
            score = self.anomaly_score(
                x, scale=scale, return_pred=return_pred)
            if return_pred:
                score, x_dash = score
                preds += [x_dash]
            scores += [score.tolist()]
        if return_pred:
            return scores, preds
        return scores

    def fit_detector(
        self,
        normal_data: MultiTimeSeriesDataloader,
        anomaly_data: MultiTimeSeriesDataloader,
        class_weight: Dict = {0: 0.75, 1: 0.25},
        load_path: Path = None,
        save_path: Path = None,
        plot: bool = False
    ):
        self.eval()
        if load_path is not None:
            scores, classes = self.load_anom_scores(load_path)
        else:
            n_scores = self.score_dataset(
                dataloader=normal_data, scale=False, return_pred=plot)
            a_scores = self.score_dataset(
                dataloader=anomaly_data, scale=False, return_pred=plot
            )
            if plot:
                n_scores, n_preds = n_scores
                a_scores, a_preds = a_scores

            scores = n_scores + a_scores
            classes = [0]*len(n_scores) + [1]*len(a_scores)
            if save_path is not None:
                self.save_anom_scores(scores, classes, save_path)

        pred_cls = self.fit_thresholder(
            scores=scores, classes=classes, scaler=self.scaler,
            class_weight=class_weight)

        if plot:
            n_preds = torch.cat(n_preds).numpy()
            a_preds = torch.cat(a_preds).numpy()

            n_preds = self.preds_to_df(
                normal_data, n_preds, return_quantiles=True)
            a_preds = self.preds_to_df(
                anomaly_data, a_preds, return_quantiles=True)

            n_pred_anom_ids = pred_cls[np.argwhere(classes == 0)]
            a_pred_anom_ids = pred_cls[np.argwhere(classes == 1)]

            self.plot_with_time_series(
                time_series=normal_data, pred_anomalies_ids=n_pred_anom_ids,
                model_preds=n_preds, anomalies_as_intervals=True,
                title='Normal data'
            )
            self.plot_with_time_series(
                time_series=anomaly_data, pred_anomalies_ids=a_pred_anom_ids,
                model_preds=a_preds, anomalies_as_intervals=True,
                title='Anomaly data'
            )

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

    def _anomalies_ids_to_points_df(
        self,
        anom_ids: List[int],
        time_series: MultiTimeSeriesDataloader,
    ):
        if issubclass(type(self), Reconstructor):
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
            is_ae=issubclass(type(self), Reconstructor),
            title=title, file_path=file_path
        )

    def _plot_fit_results(
        self,
        plot_time_series=False,
        n_res=None, a_res=None,
        n_boundries=None, a_boundries=None,
        pred_cls=None,
        n_data=None, a_data=None,
        model_n_ts_preds=None, model_a_ts_preds=None,
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

    def _plot_detection_results(
        self,
        plot_time_series=False,
        data=None,
        pred_cls=None,
        model_ts_preds=None,
    ):
        if plot_time_series:
            self.plot_with_time_series(
                time_series=data, pred_anomalies_ids=pred_cls,
                model_preds=model_ts_preds, title="Predicted anomalies")

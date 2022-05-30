from abc import abstractmethod
import csv
from datetime import timedelta
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from string import Template
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Union, Tuple, Dict, List, Literal
from sklearn.metrics import fbeta_score

from predpy.dataset import MultiTimeSeriesDataloader, MultiTimeSeriesDataset
from predpy.plotter.plotter import plot_anomalies
from predpy.wrapper import Reconstructor

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, MultiTimeSeriesDataloader.")


def _str_to_float_list(text: str) -> List[float]:
    floats = re.findall(r'\d+.\d+', text)
    res = []
    for f in floats:
        res += [float(f)]
    return res


def get_dataset(
    path: Path, window_size: int, ts_scaler: TransformerMixin = None
) -> MultiTimeSeriesDataset:
    df = pd.read_csv(
        path, header=None
    )
    try:
        df.columns = df.columns.astype(int)
    except TypeError:
        pass
    if ts_scaler is not None:
        df[:] = ts_scaler.transform(df)
    dataset = MultiTimeSeriesDataset(
        sequences=[df],
        window_size=window_size,
        target=df.columns.tolist()
    )
    return dataset


class AnomalyDetector:
    def __init__(
        self,
    ):
        self.thresholder = LogisticRegression()
        self.scores_scaler = MinMaxScaler()

    @abstractmethod
    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        pass

    def fit_run_detection(
        self, test_path: Path,
        window_size: int, min_points: int, plot: bool, scale_scores: bool,
        batch_size: int = 64, save_html_path=None,
        class_weight: Dict[Literal[0, 1], float] = {0: 0.5, 1: 0.5},
        test_cls_path: Path = None,
        rec_classes: List[int] = [],
        load_scores_path: Path = None,
        save_scores_path: Path = None,
        load_preds_path: Path = None,
        save_preds_path: Path = None,
        ts_scaler: TransformerMixin = None,
        start_plot_pos: int = None,
        end_plot_pos: int = None,
        f_score_beta: float = None,
        wdd_t_max: int = None,
        wdd_w_f: float = None
    ):
        """test_path file should contain columns of features
        without header in first line,
        test_cls file has to be csv with one column filled
        with values 0 (normal data) 1 (anomaly),
        min_points is minimal points required in record sequence to be anomaly,
        scale_scores should be True only if model requires
        scaling anomaly scores"""
        assert not (test_cls_path is None and len(rec_classes) == 0),\
            'rec_classes and test_cls_path cannot be both None.'
        dataset = get_dataset(
            path=test_path, window_size=window_size, ts_scaler=ts_scaler)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8
        )
        if load_scores_path is None and len(rec_classes) == 0:
            data_classes = pd.read_csv(
                test_cls_path, header=None)\
                .iloc[:, 0].to_list()
            rec_classes = dataset.get_recs_cls_by_data_cls(
                data_classes, min_points=min_points)

        self.fit_detector(
            dataloader=dataloader,
            classes=np.array(rec_classes),
            plot=plot, scale_scores=scale_scores,
            save_html_path=save_html_path,
            class_weight=class_weight,
            load_scores_path=load_scores_path,
            save_scores_path=save_scores_path,
            load_preds_path=load_preds_path,
            save_preds_path=save_preds_path,
            ts_scaler=ts_scaler,
            start_plot_pos=start_plot_pos,
            end_plot_pos=end_plot_pos,
            f_score_beta=f_score_beta,
            wdd_t_max=wdd_t_max, wdd_w_f=wdd_w_f
        )

    def fit_detector(
        self,
        dataloader: MultiTimeSeriesDataloader,
        classes: np.ndarray = None,
        load_scores_path: Path = None,
        save_scores_path: Path = None,
        load_preds_path: Path = None,
        save_preds_path: Path = None,
        class_weight: Dict = {0: 0.75, 1: 0.25},
        plot: bool = False,
        start_plot_pos: int = None,
        end_plot_pos: int = None,
        scale_scores: bool = False,
        ts_scaler: TransformerMixin = None,
        save_html_path: Path = None,
        f_score_beta: float = None,
        wdd_t_max: int = None,
        wdd_w_f: float = None
    ):
        assert classes is not None or load_scores_path is not None,\
            'Classes can not be None. Pass it or load from file.'
        if load_scores_path is not None:
            scores, classes = self.load_anom_scores(load_scores_path)
            if load_preds_path is not None:
                preds = pd.read_csv(load_preds_path)
        else:
            scores = self.score_dataset(
                dataloader=dataloader, scale=False, return_pred=plot)
            if plot:
                scores, preds = scores
                preds = self.preds_to_df(
                    dataloader, preds.numpy(), return_quantiles=True)
                if save_preds_path is not None:
                    os.makedirs(
                        os.path.dirname(save_preds_path), exist_ok=True)
                    preds.to_csv(save_preds_path)
            if save_scores_path is not None:
                os.makedirs(os.path.dirname(save_scores_path), exist_ok=True)
                self.save_anom_scores(
                    scores=scores, classes=classes, path=save_scores_path)
        pred_cls = self.fit_thresholder(
            scores=np.array(scores), classes=np.array(classes),
            scale_scores=scale_scores, class_weight=class_weight)

        if f_score_beta is not None:
            f_score = fbeta_score(
                classes, pred_cls, beta=f_score_beta, average='macro')
            print("F_%.2f_score: %.3f" % (f_score_beta, f_score))
        if wdd_t_max is not None and wdd_w_f is not None:
            wdd = dataloader.dataset.calculate_wdd(
                pred_rec_cls=pred_cls, true_rec_cls=classes,
                t_max=wdd_t_max, w_f=wdd_w_f
            )
            print("WDD score (t_max=%d, wf=%.2f): %.3f" %
                  (wdd_t_max, wdd_w_f, wdd))

        if plot:
            scores_df = self._get_scores_dataset(
                scores=np.array(scores), true_cls=np.array(classes),
                dataset=dataloader.dataset)
            self.plot_preds_and_anomalies(
                dataloader=dataloader, preds=preds,
                classes=np.array(classes), pred_cls=pred_cls,
                scaler=ts_scaler, save_html_path=save_html_path,
                start_pos=start_plot_pos, end_pos=end_plot_pos,
                scores_df=scores_df
            )

    def plot_preds_and_anomalies(
        self,
        dataloader: MultiTimeSeriesDataloader, preds: pd.DataFrame,
        classes: np.ndarray, pred_cls: np.ndarray,
        scaler: TransformerMixin = None,
        save_html_path: Path = None,
        start_pos=None, end_pos=None,
        scores_df: pd.DataFrame = None
    ):

        pred_anom_ids = np.argwhere(pred_cls == 1)
        pred_anom_ids = dataloader.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(pred_anom_ids).tolist())
        true_anom_ids = np.argwhere(classes == 1)
        true_anom_ids = dataloader.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(true_anom_ids).tolist())

        pred_anom_intervals = self._get_ids_ranges(pred_anom_ids)
        true_anom_intervals = self._get_ids_ranges(true_anom_ids)

        plot_anomalies(
            time_series=pd.concat(
                dataloader.dataset.sequences
            ).iloc[start_pos:end_pos],
            predictions=preds.iloc[start_pos:end_pos],
            pred_anomalies_intervals=pred_anom_intervals,
            true_anomalies_intervals=true_anom_intervals,
            scaler=scaler, is_ae=issubclass(type(self), Reconstructor),
            title='Anomaly detection results', file_path=save_html_path,
            scores_df=scores_df
        )

    def _get_scores_dataset(
        self, scores: np.ndarray, true_cls: np.ndarray,
        dataset: MultiTimeSeriesDataset
    ):
        assert scores.shape[0] == true_cls.shape[0]
        ids = dataset.get_labels().index
        if len(scores.shape) == 1:
            scores = np.expand_dims(scores, axis=1)
        true_cls = np.expand_dims(true_cls, axis=1)
        data = np.concatenate([scores, true_cls], axis=1)
        columns = [str(i) for i in range(scores.shape[1])] + ['class']
        df = pd.DataFrame(data, columns=columns, index=ids)
        return df

    def fit_detector_2_datasets(
        self,
        normal_data: MultiTimeSeriesDataloader,
        anomaly_data: MultiTimeSeriesDataloader,
        class_weight: Dict = {0: 0.75, 1: 0.25},
        load_path: Path = None,
        save_path: Path = None,
        plot: bool = False,
        scale_scores: bool = False,
        ts_scaler: TransformerMixin = None,
        save_html_path: Path = None,
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
            classes = np.array([0]*len(n_scores) + [1]*len(a_scores))
            if save_path is not None:
                self.save_anom_scores(scores, classes, save_path)

        pred_cls = self.fit_thresholder(
            scores=np.array(scores), classes=np.array(classes),
            scale_scores=scale_scores, class_weight=class_weight)

        if plot:
            self.plot_preds_and_anomalies(
                n_preds=n_preds, a_preds=a_preds,
                normal_data=normal_data, anomaly_data=anomaly_data,
                classes=classes, pred_cls=pred_cls,
                scaler=ts_scaler, save_html_path=save_html_path)

    def plot_preds_and_anomalies_2_datasets(
        self, n_preds: torch.Tensor, a_preds: torch.Tensor,
        normal_data: MultiTimeSeriesDataloader,
        anomaly_data: MultiTimeSeriesDataloader,
        classes: np.ndarray, pred_cls: np.ndarray,
        scaler: TransformerMixin = None,
        save_html_path: Path = None
    ):
        n_preds = torch.cat(n_preds).numpy()
        a_preds = torch.cat(a_preds).numpy()

        n_preds = self.preds_to_df(
            normal_data, n_preds, return_quantiles=True)
        a_preds = self.preds_to_df(
            anomaly_data, a_preds, return_quantiles=True)

        # # wersja z jednym wspólnym dataloaderem
        # # w którym są dane normalne i anomalie
        # n_pred_anom_ids = np.argwhere((classes == 0) & (pred_cls == 1))
        # a_pred_anom_ids = np.argwhere((classes == 1) & (pred_cls == 1))

        # wersja z osobnymi dataloaderami
        # dla danych normalnych i anomalii
        n_pred_anom_ids = np.argwhere(pred_cls[np.where(classes == 0)] == 1)
        a_pred_anom_ids = np.argwhere(pred_cls[np.where(classes == 1)] == 1)

        n_pred_anom_ids = normal_data.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(n_pred_anom_ids).tolist())
        a_pred_anom_ids = anomaly_data.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(a_pred_anom_ids).tolist())

        n_anom_intervals = self._get_ids_ranges(n_pred_anom_ids)
        a_anom_intervals = self._get_ids_ranges(a_pred_anom_ids)

        if save_html_path is not None:
            n_html_path = save_html_path + '_normal'
            a_html_path = save_html_path + '_anomaly'
        else:
            n_html_path, a_html_path = None, None
        plot_anomalies(
            time_series=pd.concat(normal_data.dataset.sequences),
            predictions=n_preds,
            pred_anomalies_intervals=n_anom_intervals,
            true_anomalies_intervals=None,
            scaler=scaler, is_ae=issubclass(type(self), Reconstructor),
            title='Normal data', file_path=n_html_path
        )
        plot_anomalies(
            time_series=pd.concat(anomaly_data.dataset.sequences),
            predictions=a_preds,
            pred_anomalies_intervals=a_anom_intervals,
            true_anomalies_intervals=None,
            scaler=scaler, is_ae=issubclass(type(self), Reconstructor),
            title='Anomaly data', file_path=a_html_path
        )

    def _get_ids_ranges(
        self,
        ids: List,
        max_break: Union[int, timedelta] = None
    ) -> Union[pd.Series, pd.DataFrame]:
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

    def load_anom_scores(
        self, path: Path
    ) -> Tuple[List[float], List[int]]:
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            scores = []
            classes = []
            for row in reader:
                scores += [_str_to_float_list(row['score'])]
                classes += [int(row['class'])]
        return scores, classes

    def save_anom_scores(
        self,
        scores: List,
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
        scores: np.ndarray,
        classes: np.ndarray,
        scale_scores: bool = False,
        class_weight: Dict = {0: 0.75, 1: 0.25}
    ) -> np.ndarray:
        if scale_scores is not None:
            # scaling scores
            n_scores = self.scores_scaler.fit_transform(
                scores[np.where(classes == 0)])
            a_scores = self.scores_scaler.transform(
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
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        scores_list, preds = [], []
        for batch in tqdm(
                dataloader, desc='Calculating dataset anomaly scores'):
            x = batch['sequence']
            scores = self.anomaly_score(
                x, scale=scale, return_pred=return_pred)
            if return_pred:
                scores, x_dash = scores
                preds += [x_dash]
            scores_list += scores
        if return_pred:
            preds = torch.cat(preds)
            return scores_list, preds
        return scores_list

    # def _anomalies_ids_to_points_df(
    #     self,
    #     anom_ids: List[int],
    #     time_series: MultiTimeSeriesDataloader,
    # ):
    #     if issubclass(type(self), Reconstructor):
    #         df = time_series.dataset.global_ids_to_data(
    #             anom_ids)
    #     else:
    #         df = time_series.dataset.get_labels().iloc[
    #             anom_ids]
    #     return df

    # def _anomalies_to_intervals(
    #     self,
    #     anomalies: pd.DataFrame,
    #     time_series: MultiTimeSeriesDataloader,
    # ):
    #     def get_previous(series: pd.Series, idx: Union[int, timedelta]):
    #         return series.iloc[series.index.get_loc(idx) - 1]
    #     step = time_series.dataset.sequences[0]\
    #         .index.to_series().diff().mode()[0]

    #     index = anomalies.index.to_series()
    #     diffs = index.diff()

    #     splits = diffs[diffs > step].index
    #     if len(anomalies) == 0:
    #         intervals = None
    #     elif splits.shape[0] == 0:
    #         intervals = [(index.iloc[0], index.iloc[-1])]
    #     else:
    #         intervals = [
    #             (index.iloc[0],
    #              get_previous(index, splits[0]))]
    #         intervals += [
    #             (splits[i],
    #              get_previous(index, splits[i+1]))
    #             for i in range(len(splits)-1)
    #         ]
    #         intervals += [(splits[-1], index.iloc[-1])]

    #     return intervals

    # def plot_with_time_series(
    #     self,
    #     time_series: MultiTimeSeriesDataloader,
    #     pred_anomalies_ids: List[int],
    #     true_anomalies_ids: List[int] = None,
    #     model_preds: pd.DataFrame = None,
    #     detector_boundries: pd.DataFrame = None,
    #     anomalies_as_intervals: bool = False,
    #     title: str = "Finding anomalies",
    #     file_path: str = None
    # ):
    #     pred_anom = self._anomalies_ids_to_points_df(
    #         anom_ids=pred_anomalies_ids, time_series=time_series)
    #     if anomalies_as_intervals:
    #         pred_anom_intervals = self._anomalies_to_intervals(
    #             anomalies=pred_anom, time_series=time_series)

    #     true_anom = None
    #     true_anom_intervals = None
    #     if true_anomalies_ids is not None:
    #         true_anom = time_series.dataset.global_ids_to_data(
    #             true_anomalies_ids)
    #         if anomalies_as_intervals:
    #             true_anom_intervals = self._anomalies_to_intervals(
    #                 anomalies=true_anom, time_series=time_series)

    #     ts = pd.concat(time_series.dataset.sequences)[
    #         time_series.dataset.target]
    #     plot_anomalies(
    #         time_series=ts,
    #         pred_anomalies=pred_anom,
    #         pred_anomalies_intervals=pred_anom_intervals,
    #         true_anomalies=true_anom,
    #         true_anomalies_intervals=true_anom_intervals,
    #         predictions=model_preds,
    #         detector_boundries=detector_boundries,
    #         is_ae=issubclass(type(self), Reconstructor),
    #         title=title, file_path=file_path
    #     )

    # def _plot_fit_results(
    #     self,
    #     plot_time_series=False,
    #     n_res=None, a_res=None,
    #     n_boundries=None, a_boundries=None,
    #     pred_cls=None,
    #     n_data=None, a_data=None,
    #     model_n_ts_preds=None, model_a_ts_preds=None,
    #     anomalies_as_intervals=True
    # ):
    #     if plot_time_series:
    #         self.plot_with_time_series(
    #             time_series=n_data,
    #             pred_anomalies_ids=np.argwhere(
    #                 pred_cls[:len(n_res)] == 1).T[0],
    #             model_preds=model_n_ts_preds,
    #             detector_boundries=n_boundries,
    #             anomalies_as_intervals=anomalies_as_intervals,
    #             title="Detecting anomalies on normal data"
    #         )
    #         self.plot_with_time_series(
    #             time_series=a_data,
    #             pred_anomalies_ids=np.argwhere(
    #                 pred_cls[len(n_res):] == 1).T[0],
    #             true_anomalies_ids=list(range(0, len(a_res))),
    #             model_preds=model_a_ts_preds,
    #             detector_boundries=a_boundries,
    #             anomalies_as_intervals=anomalies_as_intervals,
    #             title="Detecting anomalies on anomaly data"
    #         )

    # def _plot_detection_results(
    #     self,
    #     plot_time_series=False,
    #     data=None,
    #     pred_cls=None,
    #     model_ts_preds=None,
    # ):
    #     if plot_time_series:
    #         self.plot_with_time_series(
    #             time_series=data, pred_anomalies_ids=pred_cls,
    #             model_preds=model_ts_preds, title="Predicted anomalies")

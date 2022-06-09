from abc import abstractmethod
import csv
import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from string import Template
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Union, Tuple, Dict, List, Literal
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV

from predpy.dataset import MultiTimeSeriesDataloader, MultiTimeSeriesDataset
from predpy.plotter.plotter import plot_anomalies, get_ids_ranges
from predpy.wrapper import Reconstructor
from .data_loading import load_anom_scores, get_dataset

UNKNOWN_TYPE_MSG = Template("Unknown data type $data_type.\
Allowed types: torch.Tensor, MultiTimeSeriesDataloader.")


class AnomalyDetector:
    def __init__(
        self, score_names: List[str] = None
    ):
        self.score_names = score_names
        self.scores_in_use = score_names
        self.thresholder = LogisticRegression()
        self.scores_scaler = MinMaxScaler()

    @abstractmethod
    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        pass

    def set_score_in_use(self, score_names: List[str]):
        self.scores_in_use = score_names

    def fit_run_detection(
        self, test_path: Path,
        window_size: int, min_points: int, scale_scores: bool,
        model_train_date: str,
        plot_preds: bool = False,
        plot_scores: bool = False,
        batch_size: int = 64, save_html_path=None,
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
        wdd_w_f: float = None,
        wdd_ma_f: float = 0
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
        if load_scores_path is None and\
                (rec_classes is None or len(rec_classes) == 0):
            data_classes = pd.read_csv(
                test_cls_path, header=None)\
                .iloc[:, 0].to_list()
            rec_classes = dataset.get_recs_cls_by_data_cls(
                data_classes, min_points=min_points)

        ds_name = Path(test_path).stem
        self.fit_detector(
            dataloader=dataloader,
            ds_name=ds_name, model_train_date=model_train_date,
            classes=np.array(rec_classes),
            plot_preds=plot_preds,
            plot_scores=plot_scores,
            scale_scores=scale_scores,
            save_html_path=save_html_path,
            load_scores_path=load_scores_path,
            save_scores_path=save_scores_path,
            load_preds_path=load_preds_path,
            save_preds_path=save_preds_path,
            ts_scaler=ts_scaler,
            start_plot_pos=start_plot_pos,
            end_plot_pos=end_plot_pos,
            f_score_beta=f_score_beta,
            wdd_t_max=wdd_t_max, wdd_w_f=wdd_w_f,
            wdd_ma_f=wdd_ma_f
        )

    def fit_detector(
        self,
        dataloader: MultiTimeSeriesDataloader,
        ds_name: str, model_train_date: str,
        plot_preds: bool = False,
        plot_scores: bool = False,
        classes: np.ndarray = None,
        load_scores_path: Path = None,
        save_scores_path: Path = None,
        load_preds_path: Path = None,
        save_preds_path: Path = None,
        start_plot_pos: int = None,
        end_plot_pos: int = None,
        scale_scores: bool = False,
        ts_scaler: TransformerMixin = None,
        save_html_path: Path = None,
        f_score_beta: float = 0.5,
        wdd_t_max: int = None,
        wdd_w_f: float = None,
        wdd_ma_f: float = 0
    ):
        assert classes is not None or load_scores_path is not None,\
            'Classes can not be None. Pass it or load from file.'
        if load_scores_path is not None:
            scores, classes = load_anom_scores(load_scores_path)
            if load_preds_path is not None:
                preds = pd.read_csv(load_preds_path)
        else:
            return_pred = plot_preds or save_preds_path is not None
            scores = self.score_dataset(
                dataloader=dataloader, scale=False, return_pred=return_pred)
            if return_pred:
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
            scale_scores=scale_scores, dataset=dataloader.dataset
            # f_score_beta=f_score_beta
        )

        self.save_stats(
            true_cls=classes, pred_cls=pred_cls, wdd_t_max=wdd_t_max,
            wdd_w_f=wdd_w_f, wdd_ma_f=wdd_ma_f, dataset=dataloader.dataset,
            ds_name=ds_name, model_train_date=model_train_date,
            f_score_beta=f_score_beta
        )

        if plot_preds or plot_scores:
            if not plot_preds:
                preds = None
            if plot_scores:
                scores_df = self._get_scores_dataset(
                    scores=np.array(scores), true_cls=np.array(classes),
                    dataset=dataloader.dataset)
            else:
                scores_df = None
            self.plot_preds_and_anomalies(
                dataloader=dataloader, preds=preds,
                classes=np.array(classes), pred_cls=pred_cls,
                scaler=ts_scaler, save_html_path=save_html_path,
                start_pos=start_plot_pos, end_pos=end_plot_pos,
                scores_df=scores_df, ds_name=ds_name
            )

    def save_stats(
        self, true_cls: List[Literal[0, 1]], pred_cls: List[Literal[0, 1]],
        wdd_t_max: int, wdd_w_f: float, wdd_ma_f: float,
        dataset: MultiTimeSeriesDataset, ds_name: str,
        model_train_date: str, f_score_beta: float = 0.5,
        path: str = './fit_detector.json'
    ):
        m_name = self.model.__class__.__name__
        m_params = self.model.params
        wdd = ''
        f_score = fbeta_score(
            true_cls, pred_cls, beta=f_score_beta, average='macro')
        print("F_%.2f_score: %.3f" % (f_score_beta, f_score))
        if wdd_t_max is not None and wdd_w_f is not None:
            wdd = dataset.calculate_rec_wdd(
                pred_rec_cls=pred_cls, true_rec_cls=true_cls,
                t_max=wdd_t_max, w_f=wdd_w_f, ma_f=wdd_ma_f
            )
            print("WDD score (t_max=%d, wf=%.3f, ma=%.3f): %.3f" %
                  (wdd_t_max, wdd_w_f, wdd_ma_f, wdd))
        cm = confusion_matrix(true_cls, pred_cls)
        scores = {
            'cm': cm.tolist(),
            f'F_{f_score_beta}_score': f_score,
            f'WDD_tmax{wdd_t_max}_wf{wdd_w_f}_ma{wdd_ma_f}': wdd
        }
        row = dict(
            dataset=ds_name, model=m_name, train_date=model_train_date,
            params=m_params, scores=scores)

        if not os.path.exists(path) or os.stat(path).st_size < 2:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write('[]')
        with open(path, 'r') as f:
            file_content = json.load(f)
        file_content.append(row)
        with open(path, 'w') as f:
            json.dump(file_content, f)
        # with open('./fit_detector.csv', 'a') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(row)

    def plot_preds_and_anomalies(
        self,
        dataloader: MultiTimeSeriesDataloader,
        classes: np.ndarray, pred_cls: np.ndarray,
        scaler: TransformerMixin = None,
        save_html_path: Path = None,
        start_pos=None, end_pos=None,
        preds: pd.DataFrame = None,
        scores_df: pd.DataFrame = None,
        ds_name: str = ''
    ):

        pred_anom_ids = np.argwhere(pred_cls == 1)
        pred_anom_ids = dataloader.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(pred_anom_ids).tolist())
        true_anom_ids = np.argwhere(classes == 1)
        true_anom_ids = dataloader.dataset.\
            get_data_ids_by_rec_ids(np.squeeze(true_anom_ids).tolist())

        pred_anom_intervals = get_ids_ranges(pred_anom_ids)
        true_anom_intervals = get_ids_ranges(true_anom_ids)

        if preds is not None:
            preds = preds.iloc[start_pos:end_pos]

        plot_anomalies(
            time_series=pd.concat(
                dataloader.dataset.sequences
            ).iloc[start_pos:end_pos],
            predictions=preds,
            pred_anomalies_intervals=pred_anom_intervals,
            true_anomalies_intervals=true_anom_intervals,
            scaler=scaler, is_ae=issubclass(type(self), Reconstructor),
            title='Anomaly detection on "%s"' % ds_name,
            file_path=save_html_path, scores_df=scores_df,
            model_name=self.model.__class__.__name__
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
        columns = self.score_names + ['class']
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
            scores, classes = load_anom_scores(load_path)
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

        n_anom_intervals = get_ids_ranges(n_pred_anom_ids)
        a_anom_intervals = get_ids_ranges(a_pred_anom_ids)

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
            title='Normal data', file_path=n_html_path,
            model_name=self.model.__class__.__name__
        )
        plot_anomalies(
            time_series=pd.concat(anomaly_data.dataset.sequences),
            predictions=a_preds,
            pred_anomalies_intervals=a_anom_intervals,
            true_anomalies_intervals=None,
            scaler=scaler, is_ae=issubclass(type(self), Reconstructor),
            title='Anomaly data', file_path=a_html_path,
            model_name=self.model.__class__.__name__
        )

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
        dataset: MultiTimeSeriesDataset,
        scale_scores: bool = False,
        wdd_t_max: int = None,
        wdd_w_f: float = None,
        wdd_ma_f: float = 0,
        # class_weight: Dict = {0: 0.75, 1: 0.25},
        f_score_beta: float = 0.5
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
        scorer = make_scorer(fbeta_score, beta=f_score_beta, average='binary')
        # cols_ids = self._get_used_scores_cols_ids()
        # if wdd_t_max is not None and wdd_w_f is not None:
        #     scorer = make_scorer(
        #         dataset.calculate_rec_wdd, t_max=wdd_t_max,
        #         w_f=wdd_w_f, ma_f=wdd_ma_f)
        # else:
        #     scorer = make_scorer(
        #         fbeta_score, beta=f_score_beta, average='macro')
        gs = GridSearchCV(LogisticRegression(), param_grid={'class_weight': [
            {0.6, 0.4}, {0.7, 0.3}, {0.8, 0.2}, {0.9, 0.1}]},
            scoring=scorer)
        # train_scores = scores[:, cols_ids]
        train_scores = scores
        gs.fit(train_scores, classes)
        self.thresholder = gs.best_estimator_
        pred_cls = self.thresholder.predict(train_scores)

        # printing classification results
        cm = confusion_matrix(classes, pred_cls)
        print("Model %s" % self.model.__class__.__name__)
        print('Best class weights: %s' % str(self.thresholder.class_weight))
        print(cm)

        return pred_cls

    def _get_used_scores_cols_ids(self) -> List[int]:
        cols_ids = [self.score_names.index(col) for col in self.scores_in_use]
        return cols_ids

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

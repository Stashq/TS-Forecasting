import os
import csv
from sklearn.base import TransformerMixin
import pandas as pd
from typing import List

from predpy.experimentator import Experimentator
from .anomaly_detector_base import AnomalyDetector
from .data_loading import get_dataset, get_dataset_names


def exp_fit_run_detection(
    exp: Experimentator, min_points: int = 1, safe: bool = True,
    plot_preds: bool = False, plot_scores: bool = False,
    save_preds: bool = False, save_scores: bool = False,
    load_preds: bool = False, load_scores: bool = False,
    save_cls: bool = False, load_cls: bool = False,
    ds_ids: List[int] = None, m_ids: List[int] = None
):
    model_train_date = exp.exp_date
    if ds_ids is None:
        ds_ids = range(exp.datasets_params.shape[0])
    if m_ids is None:
        m_ids = range(exp.models_params.shape[0])

    for ds_id in ds_ids:
        window_size = exp.datasets_params.iloc[ds_id]['window_size']
        topic, collection_name, dataset_name =\
            get_dataset_names(
                exp.datasets_params.iloc[ds_id]['path'])
        test_cls_path = './data/%s/%s/test_label/%s.csv'\
            % (topic, collection_name, dataset_name)

        dataset = get_dataset(
            path='./data/%s/%s/test/%s.csv'
            % (topic, collection_name, dataset_name),
            window_size=window_size, ts_scaler=exp.get_targets_scaler(ds_id))
        data_classes = pd.read_csv(
            test_cls_path, header=None)\
            .iloc[:, 0].to_list()
        classes_path = './saved_scores_preds/%s/%s/record_classes/%d.csv'\
            % (collection_name, dataset_name, window_size)
        if load_cls:
            with open(classes_path, 'r') as f:
                rec_classes = [row[0] for row in csv.reader(f)]
        else:
            rec_classes = dataset.get_recs_cls_by_data_cls(
                data_classes, min_points=min_points)
            if save_cls:
                os.makedirs(os.path.dirname(classes_path), exist_ok=True)
                with open(classes_path, 'w') as f:
                    csv.writer(f).writerows(
                        [[cls_] for cls_ in rec_classes]
                    )

        n_models = exp.models_params.shape[0]

        for m_id in range(0, n_models):
            model_name = exp.models_params.iloc[m_id]['name_']
            try:
                model = exp.load_pl_model(
                    m_id, os.path.join(
                        'checkpoints', dataset_name, model_name))

                fit_run_detection(
                    model=model, window_size=window_size,
                    model_name=model_name,
                    model_train_date=model_train_date,
                    topic=topic, collection_name=collection_name,
                    dataset_name=dataset_name, rec_classes=rec_classes,
                    test_cls_path=test_cls_path, min_points=min_points,
                    ts_scaler=exp.get_targets_scaler(ds_id),
                    plot_preds=plot_preds, plot_scores=plot_scores,
                    save_preds=save_preds, save_scores=save_scores,
                    load_preds=load_preds, load_scores=load_scores
                )
            except Exception as e:
                if safe:
                    print(
                        'Problem with fit_run_detection on model "%s": %s.'
                        % (model_name, str(e.args)))
                else:
                    raise e


def fit_run_detection(
    model: AnomalyDetector, window_size: int,
    model_name: str, model_train_date: str,
    topic: str, collection_name: str, dataset_name: str,
    rec_classes: List[int] = None, test_cls_path: str = None,
    min_points: int = 1, ts_scaler: TransformerMixin = None,
    plot_preds: bool = False, plot_scores: bool = False,
    save_preds: bool = False, save_scores: bool = False,
    load_preds: bool = False, load_scores: bool = False,
):
    load_scores_path, save_scores_path = None, None
    load_preds_path, save_preds_path = None, None
    path_vars = (collection_name, dataset_name, model_name)
    if load_preds:
        load_preds_path = './saved_scores_preds/%s/%s/%s/preds.csv' % path_vars
        if not os.path.exists(load_preds_path):
            print(f'File {load_preds_path} not exists.')
            load_preds_path = None
    if load_scores:
        load_scores_path =\
            './saved_scores_preds/%s/%s/%s/anom_scores.csv' % path_vars
        if not os.path.exists(load_scores_path):
            print(f'File {load_scores_path} not exists.')
            load_scores_path = None
    if save_preds:
        save_preds_path =\
            './saved_scores_preds/%s/%s/%s/preds.csv' % path_vars
    if save_scores:
        save_scores_path =\
            './saved_scores_preds/%s/%s/%s/anom_scores.csv' % path_vars

    model.fit_run_detection(
        window_size=window_size,
        test_path='./data/%s/%s/test/%s.csv' % (
            topic, collection_name, dataset_name),
        model_train_date=model_train_date,
        rec_classes=rec_classes,
        test_cls_path=test_cls_path,
        min_points=min_points, scale_scores=True,
        ts_scaler=ts_scaler,
        load_scores_path=load_scores_path,
        save_scores_path=save_scores_path,
        load_preds_path=load_preds_path,
        save_preds_path=save_preds_path,
        plot_preds=plot_preds,
        plot_scores=plot_scores,
        save_html_path='./pages/%s/%s/%s.html' % (
            collection_name, dataset_name, model_name),
        f_score_beta=0.5,
        wdd_t_max=window_size/2,
        wdd_w_f=0.0005,
        wdd_ma_f=0.0005
    )

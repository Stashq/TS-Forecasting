import numpy as np
import os
import pandas as pd
import re


def str_to_cls_ndarray(text: str):
    classes = [int(cls_[0]) for cls_ in re.findall(r'\d\.0', text)]
    return np.array(classes)


def load_th_exp(path):
    df = pd.read_csv(path)
    df['preds_rec_cls'] = df['preds_rec_cls'].apply(
        lambda x: str_to_cls_ndarray(x))
    return df


def save_th_exp(df, path: str, mk_dirs: bool = False):
    if mk_dirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_col = df['preds_rec_cls']
    df['preds_rec_cls'] = df['preds_rec_cls'].apply(lambda x: x.tolist())
    df.to_csv(path, index=False)
    # bring back column
    df['preds_rec_cls'] = tmp_col

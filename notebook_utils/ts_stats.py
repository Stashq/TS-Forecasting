import numpy as np
import pandas as pd


def get_bollinger(series: np.ndarray, ws: int, m: float = 1.0):
    rol = pd.DataFrame(series).rolling(window=ws)
    ma = rol.mean().dropna()
    std = rol.std().dropna()
    bollinger = ma + m*std
    bollinger.name = f'bollinger_ws{ws}_m{m}'
    return bollinger


def get_std(series: np.ndarray, ws: int, na=None):
    rol = pd.DataFrame(series).rolling(window=ws)
    std = rol.std()
    if na == 'drop':
        std = std.dropna()
    elif type(na) in [int, float]:
        std = std.fillna(na)
    else:
        pass
    std.name = f'std_ws{ws}'
    return std


def get_diff(stat: pd.DataFrame, na='fill'):
    res = stat.diff()
    if na == 'fill':
        res = res.fillna(0)
    elif na == 'drop':
        res = res.dropna()
    res.name = stat.name + '_diff'
    return res

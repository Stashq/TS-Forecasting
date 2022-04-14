from typing import Literal, Callable, List
import pandas as pd
import numpy as np

NEGATIVITY_REMOVING_METHODS = Literal["abs", "resample", "zero"]


def apply_noise(
    row: pd.Series,
    make_noise: Callable,
    negativity: NEGATIVITY_REMOVING_METHODS = None,
    max_tries: int = 5,
    *func_args, **func_kwargs
) -> pd.Series:
    result = row.apply(make_noise, args=func_args, **func_kwargs)
    mask = result < 0
    if mask.any() and negativity is not None:
        if negativity == "abs":
            result[mask] = result[mask].apply(abs)
        elif negativity == "resample":
            i = 0
            while mask.any() and i < max_tries:
                result[mask] = result[mask].apply(
                    make_noise, args=func_args, **func_kwargs)
                mask = result < 0
                i += 1
            if mask.any():
                result[mask] = result[mask].apply(abs)
        elif negativity == "zero":
            result[mask] = 0
        else:
            ValueError("Unknown removing negativity method.")
    return result


def apply_noise_on_dataframes(
    dfs: List[pd.DataFrame],
    make_noise: Callable,
    negativity: NEGATIVITY_REMOVING_METHODS = None,
    max_tries: int = 5,
    *func_args, **func_kwargs
) -> List[pd.DataFrame]:
    for df in dfs:
        df.loc[:, :] = df.apply(
            apply_noise, make_noise=make_noise, negativity=negativity,
            max_tries=max_tries, args=func_args, **func_kwargs)
    return dfs


def white_noise(
    row: pd.Series,
    loc: float = 0.0,
    scale: float = 1.0
):
    result = row + loc + scale * np.random.randn()
    return result

from scipy.stats import norm
import numpy as np
from .base import Distributor
from typing import Union, List

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class Gaussian(Distributor):
    """Gaussian distributions where for every dimention
    different gaussian distribution is set without including correlation.
    """
    def __init__(self):
        self._n_dims = None
        self._means = []
        self._vars = []

    def fit(self, data: np.ndarray, verbose: bool = False):
        if len(data.shape) == 2:
            self._n_dims = 1
            mean, var = norm.fit(data)
            self._means += [mean]
            self._vars += [var]
        else:
            self._n_dims = data.shape[-1]
            for dim in range(self._n_dims):
                mean, var = norm.fit(data[..., dim])
                self._means += [mean]
                self._vars += [var]

    def cdf(self, data: np.ndarray, dim: int = None) -> np.ndarray:
        if len(self._means) == 0 or len(self._vars) == 0:
            raise AttributeError(BEFORE_FIT_ERROR)
        if dim is None:
            if len(data.shape) == 2:
                result = [
                    norm.cdf(
                        data, loc=self._means[0],
                        scale=self._vars[0])]
            else:
                result = [
                    norm.cdf(
                        data[..., dim], loc=self._means[dim],
                        scale=self._vars[dim])
                    for dim in range(self._n_dims)]
            return np.concatenate(result, axis=-1)
        else:
            return norm.cdf(
                data[..., dim], loc=self._means[dim], scale=self._vars[dim])

    def ppf(self, prob: Union[float, List[float]], dim: int = None) -> float:
        return norm.ppf(
            prob, loc=self._means[dim], scale=self._vars[dim])

    def pdf(self, data: np.ndarray, dim: int = None) -> np.ndarray:
        if self._means is None or self._vars is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        if dim is None:
            if len(data.shape) == 2:
                result = [norm.pdf(
                    data, loc=self._means[0],
                    scale=self._vars[0])]
            else:
                result = [
                    norm.pdf(
                        data[..., dim], loc=self._means[dim],
                        scale=self._vars[dim])
                    for dim in range(self._n_dims)]
            return np.concatenate(result, axis=-1)
        else:
            return norm.pdf(
                data[..., dim], loc=self._means[dim], scale=self._vars[dim])

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.cdf(data)

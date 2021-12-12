from scipy.stats import norm
import numpy as np
from .base import Distributor
from typing import Union, List

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class GaussianDistributor(Distributor):
    """Gaussian distributions where dimensions are agnostic to each other -
    for every dimention different gaussian distribution is set without
    including correlation.
    """
    def __init__(self):
        self._n_dims = None
        self._means = []
        self._vars = []
        # self._mean = None
        # self._cov = None
        # self._mvnormal = None

    def fit(self, data: np.ndarray):
        self._n_dims = data.shape[1]
        for dim in range(self._n_dims):
            mean, var = norm.fit(data[:, dim])
            self._means += [mean]
            self._vars += [var]
        # self._mean = np.mean(data, axis=0)
        # self._cov = np.cov(data, rowvar=False)
        # self._mvnormal = multivariate_normal(
        #     mean=self._mean, cov=self._cov, allow_singular=True)

    def cdf(self, data: np.ndarray, dim: int = None) -> np.ndarray:
        if len(self._means) == 0 or len(self._vars) == 0:
            raise AttributeError(BEFORE_FIT_ERROR)
        if dim is None:
            return np.vstack([
                norm.cdf(
                    data[:, dim], loc=self._means[dim], scale=self._vars[dim])
                for dim in range(self._n_dims)
            ]).T
        else:
            return norm.cdf(
                data[:, dim], loc=self._means[dim], scale=self._vars[dim])
        # probs = self._mvnormal.cdf(data)
        # if len(probs.shape) == 1:
        #     probs = np.expand_dims(probs, axis=1)
        # return probs

    def ppf(self, prob: Union[float, List[float]], dim: int = None) -> float:
        return norm.ppf(
            prob, loc=self._means[dim], scale=self._vars[dim])

    def pdf(self, data: np.ndarray, dim: int = None) -> np.ndarray:
        if self._means is None or self._vars is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        if dim is None:
            return np.vstack([
                norm.pdf(
                    data[:, dim], loc=self._means[dim], scale=self._vars[dim])
                for dim in range(self._n_dims)
            ]).T
        else:
            return norm.pdf(
                data[:, dim], loc=self._means[dim], scale=self._vars[dim])
        # probs = self._mvnormal.pdf(data)
        # if len(probs.shape) == 1:
        #     probs = np.expand_dims(probs, axis=1)
        # return probs

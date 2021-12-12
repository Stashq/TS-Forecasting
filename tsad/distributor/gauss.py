from scipy.stats import multivariate_normal, norm
import numpy as np
from .base import Distributor

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class GaussianDistributor(Distributor):
    def __init__(self):
        self._mean = None
        self._cov = None
        self._mvnormal = None

    def fit(self, data: np.ndarray):
        self._mean = np.mean(data, axis=0)
        self._cov = np.cov(data, rowvar=False)
        self._mvnormal = multivariate_normal(
            mean=self._mean, cov=self._cov, allow_singular=True)

    def cdf(self, data: np.ndarray) -> np.ndarray:
        if self._mvnormal is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        probs = self._mvnormal.cdf(data)
        if len(probs.shape) == 1:
            probs = np.expand_dims(probs, axis=1)
        return probs

    def ppf(
        self,
        prob: float,
        dim: int
    ) -> float:
        loc = self._mean[dim]
        scale = self._cov[dim][dim]
        return norm.ppf(prob, loc=loc, scale=scale)

    def pdf(self, data: np.ndarray) -> np.ndarray:
        if self._mvnormal is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        probs = self._mvnormal.pdf(data)
        if len(probs.shape) == 1:
            probs = np.expand_dims(probs, axis=1)
        return probs

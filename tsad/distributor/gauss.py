from scipy.stats import multivariate_normal
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

    def probs(self, data: np.ndarray) -> np.ndarray:
        if self._mvnormal is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        probs = self._mvnormal.pdf(data)
        probs = np.expand_dims(probs, axis=1)
        return probs

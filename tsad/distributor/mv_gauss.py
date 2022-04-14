from scipy.stats import multivariate_normal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .base import Distributor

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class MVGaussian(Distributor):
    """Gaussian distributions where for every dimention
    different gaussian distribution is set without including correlation.
    """
    def __init__(self):
        self.mv_norm = None
        self.scaler = MinMaxScaler()

    def fit(self, data: np.ndarray, verbose: bool = False):
        data = self._adjust_dims(data)
        data = self.scaler.fit_transform(data)
        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        self.mv_norm = multivariate_normal(mean, cov, allow_singular=True)

    def cdf(self, data: np.ndarray) -> np.ndarray:
        if self.mv_norm is None:
            raise AttributeError(BEFORE_FIT_ERROR)

        data = self._adjust_dims(data)
        data = self.scaler.transform(data)
        res = self.mv_norm.cdf(data)
        return np.expand_dims(res, axis=-1)

    def ppf(
        self,
        data: np.ndarray
    ):
        raise NotImplementedError("You can not use ppf for this distributor.")

    def pdf(self, data: np.ndarray) -> np.ndarray:
        if self.mv_norm is None:
            raise AttributeError(BEFORE_FIT_ERROR)

        data = self._adjust_dims(data)
        data = self.scaler.transform(data)
        res = self.mv_norm.pdf(data)
        return np.expand_dims(res, axis=-1)

    def _adjust_dims(self, data: np.ndarray) -> np.ndarray:
        if len(data.shape) == 3:
            data = data.reshape(len(data), -1)
        return data

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.cdf(data)

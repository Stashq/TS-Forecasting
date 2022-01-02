from scipy.stats import multivariate_normal
import numpy as np
from .base import Distributor

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class MVGaussian(Distributor):
    """Gaussian distributions where for every dimention
    different gaussian distribution is set without including correlation.
    """
    def __init__(self):
        self.mv_norm = None

    def fit(self, data: np.ndarray):
        mean = np.mean(data, axis=-1)
        cov = np.cov(data, rowvar=False)
        self.mv_norm = multivariate_normal(mean, cov)

    def cdf(self, data: np.ndarray) -> np.ndarray:
        if self.mv_norm is None:
            raise AttributeError(BEFORE_FIT_ERROR)
        return self.mv_norm.cdf(data)

    def ppf(
        self,
        data: np.ndarray
    ):
        raise NotImplementedError("You can not use ppf for this distributor.")

    def pdf(self, data: np.ndarray) -> np.ndarray:
        if self.mv_norm is None:
            raise AttributeError(BEFORE_FIT_ERROR)

        return self.mv_norm.pdf(data)

import numpy as np
from abc import ABC, abstractmethod


class Distributor(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray):
        """Fits distribution coefficients to data.

        Parameters
        ----------
        data : np.ndarray
            Data to fit.
        """
        pass

    @abstractmethod
    def cdf(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def ppf(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def pdf(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        pass
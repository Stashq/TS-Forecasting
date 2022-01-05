import numpy as np
from abc import ABC, abstractmethod


class Distributor(ABC):
    @abstractmethod
    def fit(self, data: np.ndarray, verbose: bool = False):
        """Fits distribution coefficients to data.

        Parameters
        ----------
        data : np.ndarray
            Data to fit.
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        pass

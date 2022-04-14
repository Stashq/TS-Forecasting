from sklearn.mixture import GaussianMixture
import numpy as np
from .base import Distributor
from typing import Dict, Tuple

BEFORE_FIT_ERROR =\
    "Attempt to obtain probabilities before fitting the distribution."


class GMM(Distributor):
    """Gaussian distributions where for every dimention
    different gaussian distribution is set without including correlation.
    """
    def __init__(self):
        self.gmm = None

    def fit(
        self,
        data: np.ndarray,
        n_componens: int = None,
        components_range: range = range(1, 11),
        bic: bool = True,
        verbose: bool = False
    ):
        if n_componens is not None:
            self.gmm, score = self._train_model(data, n_componens, bic)
            if verbose:
                print("Model score: %.4f" % score)
        else:
            models = self._train_models(data, components_range, bic)
            self._pick_best_model(models, verbose)

    def _train_model(
        self,
        data: np.ndarray,
        n_componens: int,
        bic: bool = True
    ) -> Tuple[GaussianMixture, float]:
        model = GaussianMixture(
            n_components=n_componens, random_state=0)
        model.fit(data)
        score = model.bic(data) if bic is True else model.aic(data)
        return model, score

    def _train_models(
        self,
        data: np.ndarray,
        components_range: range = range(1, 20),
        bic: bool = True
    ) -> Dict:
        models = {}
        for n_comp in components_range:
            model, score = self._train_model(data, n_comp, bic)
            models[n_comp] = (model, score)
        return models

    def _pick_best_model(self, models: Dict, verbose: bool = False):
        best_score = float("inf")
        best_n = -1
        for n_comp, (model, score) in models.items():
            if best_score > score:
                self.gmm = model
                best_score = score
                best_n = n_comp
        if verbose:
            print("GMM by n components scores:")
            print([(n_comp, format(score, '.2f'))
                   for n_comp, (_, score) in models.items()])
            print("Choosing model with %d components" % best_n)

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.gmm.score_samples(data).reshape(len(data), 1)

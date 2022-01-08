from abc import ABC, abstractmethod
# from typing import Union
import numpy as np
from string import Template

# import torch
# from torch import nn
# from sklearn.base import BaseEstimator

FORBIDDEN_TYPE_MSG = Template(
    "Expected model type to be: nn.Module or " +
    "sklearn.base.BaseEstimator. Got $model_type")


class Regressor(ABC):
    # def __init__(self, model: Union[nn.Module, BaseEstimator]):
    #     super().__init__()
    #     if not issubclass(type(self.model), nn.Module)\
    #             and not issubclass(type(self.model), BaseEstimator):
    #         raise ValueError(FORBIDDEN_TYPE_MSG.substitute(
    #             model_type=type(model)
    #         ))
    #     self.model = model

    # def predict(self, emb: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    #     if issubclass(type(self.model), nn.Module):
    #         if isinstance(emb, np.ndarray):
    #             emb = torch.tensor(emb)
    #         res = self.model(emb)
    #         res = res.cpu().detach().numpy()
    #     elif issubclass(type(self.model), BaseEstimator):
    #         if isinstance(emb, torch.Tensor):
    #             emb = emb.cpu().detach().numpy()
    #         res = self.model.predict(emb)
    #     return res

    @abstractmethod
    def predict(self, emb) -> np.ndarray:
        pass

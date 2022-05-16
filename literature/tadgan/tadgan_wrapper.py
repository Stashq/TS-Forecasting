from torch import nn, optim
from typing import Dict, Generator, List
import torch
from torch.nn.parameter import Parameter

from predpy.wrapper import Reconstructor
from .tadgan import TADGAN


class TADGANWrapper(Reconstructor):
    def __init__(
        self,
        model: TADGAN = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter] = None
    ):
        super(TADGANWrapper, self).__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )

    def get_loss(
        self, x
    ):
        pass

    def step(self, batch):
        # x, _ = self.get_Xy(batch)
        # res = self.model(x)
        # loss = self.get_loss(x, *res)
        # return loss
        pass

    def predict(self, x):
        with torch.no_grad():
            x_dash, _, _, _, _, _, _ = self(x)
            return x_dash

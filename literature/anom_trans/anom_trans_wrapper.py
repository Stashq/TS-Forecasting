from torch import nn, optim
from typing import Dict, Generator, List
import torch
from torch.nn.parameter import Parameter

from .anom_trans import AnomalyTransformer
from predpy.wrapper import Reconstructor


class ATWrapper(Reconstructor):
    def __init__(
        self,
        model: AnomalyTransformer = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None
    ):
        super(ATWrapper, self).__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.mse = nn.MSELoss()

    @property
    def automatic_optimization(self) -> bool:
        return False

    def get_loss(self, x_hat, P_list, S_list, lambda_, x):
        frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        return frob_norm - (
            lambda_
            * torch.linalg.norm(
                self.model.association_discrepancy(P_list, S_list, x),
                ord=1)
        )

    def min_loss(self, x, x_hat, P_layers, S_layers):
        P_list = P_layers
        S_list = [S.detach() for S in S_layers]
        lambda_ = -self.model.lambda_
        return self.get_loss(x_hat, P_list, S_list, lambda_, x).mean()

    def max_loss(self, x, x_hat, P_layers, S_layers):
        P_list = [P.detach() for P in P_layers]
        S_list = S_layers
        lambda_ = self.model.lambda_
        return self.get_loss(x_hat, P_list, S_list, lambda_, x).mean()

    def step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat, P, S = self.model(x)
        min_loss = self.min_loss(x, x_hat, P, S)
        max_loss = self.max_loss(x, x_hat, P, S)
        return min_loss, max_loss

    def val_step(self, batch):
        x, _ = self.get_Xy(batch)
        x_hat, _, _ = self.model(x)
        loss = self.mse(x, x_hat)
        return loss

    def training_step(self, batch, batch_idx):
        opt = self.configure_optimizers()
        opt.zero_grad()
        min_loss, max_loss = self.step(batch)
        self.manual_backward(
            min_loss,
            retain_graph=True
        )
        self.manual_backward(
            max_loss
        )
        self.log("train_min_loss", min_loss, prog_bar=True, logger=True)
        self.log("train_max_loss", max_loss, prog_bar=True, logger=True)
        opt.step()

    def validation_step(self, batch, batch_idx):
        loss = self.val_step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.val_step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def predict(self, x):
        with torch.no_grad():
            x_hat, _, _ = self(x)
            return x_hat

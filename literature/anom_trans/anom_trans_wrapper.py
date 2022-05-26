from torch import nn, optim
from typing import Dict, Generator, List, Union, Tuple
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .anom_trans import AnomalyTransformer
from predpy.wrapper import Reconstructor
from literature.anomaly_detector_base import AnomalyDetector


class ATWrapper(Reconstructor, AnomalyDetector):
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
        AnomalyDetector.__init__(self)
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.mse = nn.MSELoss()

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, x):
        return self.model(x)

    def get_loss(self, x_hat, P_list, S_list, lambda_, x):
        frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        return frob_norm - (
            lambda_
            * torch.linalg.norm(
                self.model.association_discrepancy(P_list, S_list),
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

    def predict(self, x):
        with torch.no_grad():
            x_hat, _, _ = self.model(x)
            return x_hat

    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        with torch.no_grad():
            x_hat, P_layers, S_layers = self.model(x)
            ad = F.softmax(
                -self.model.association_discrepancy(
                    P_layers, S_layers),
                dim=1
            )

        assert ad.shape[1] == self.model.N

        norm = torch.linalg.norm((x - x_hat), ord=2, dim=2)
        # norm = torch.tensor(
        #     [
        #         torch.linalg.norm(x[:, i, :] - x_hat[:, i, :], ord=2)
        #         for i in range(self.model.N)
        #     ]
        # )

        assert norm.shape[1] == self.model.N

        score = torch.mul(ad, norm)

        max_score = torch.max(score, dim=1).values.tolist()
        mean_score = torch.mean(score, dim=1).tolist()

        res_score = [max_mean for max_mean in zip(max_score, mean_score)]
        if scale:
            res_score = self.scores_scaler.transform(res_score).tolist()
        if return_pred:
            return res_score, x_hat
        return res_score

    def validation_step(self, batch, batch_idx):
        loss = self.val_step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        with torch.no_grad():
            min_loss, max_loss = self.step(batch)
            self.log("val_min_loss", min_loss, prog_bar=True, logger=True)
            self.log("val_max_loss", max_loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.val_step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        with torch.no_grad():
            min_loss, max_loss = self.step(batch)
            self.log("test_min_loss", min_loss, prog_bar=True, logger=True)
            self.log("test_max_loss", max_loss, prog_bar=True, logger=True)
        return loss

from torch import nn, optim
from typing import Dict, Generator, List, Union, Tuple
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .anom_trans import AnomalyTransformer
from predpy.wrapper import Reconstructor
from anomaly_detection.anomaly_detector_base import AnomalyDetector


class ATWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: AnomalyTransformer = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None,
        score_names: List[str] = ['xd_max', 'xd_l2', 's_max', 's_mean']
    ):
        AnomalyDetector.__init__(self, score_names=score_names)
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.mse = nn.MSELoss()
        self.score_names = score_names

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

    def get_selected_scores(
        self, s_names: List[str], x_diff, a_score
    ) -> List:
        res_score = []
        for s_name in s_names:
            res_score += [
                self.get_score(s_name, x_diff, a_score)]
        res_score = torch.concat(
            res_score, dim=1
        )
        res_score = res_score.tolist()
        return res_score

    def get_score(self, name: str, x_diff, a_score) -> torch.Tensor:
        # select data
        if name[:2] == 'xd':
            data = x_diff
        elif name[0] == 's':
            data = a_score.unsqueeze(-1)
        else:
            raise ValueError(f'Unknown score name {name}.')

        # # calculate described score
        # if name[-2:] == 'l2':
        #     score = torch.sum(torch.linalg.norm(data, dim=1, ord=2), dim=-1)
        # elif name[-3:] == 'max':
        #     score = torch.max(torch.max(data, dim=1).values, dim=-1).values
        # elif name[-4:] == 'mean':
        #     score = torch.mean(torch.mean(data, dim=1), dim=-1)
        # else:
        #     raise ValueError(f'Unknown score name {name}.')

        # score = score.unsqueeze(1)

        if name[-2:] == 'l2':
            score = torch.linalg.norm(data, dim=1, ord=2)
        elif name[-3:] == 'max':
            score = torch.max(data, dim=1).values
        elif name[-4:] == 'mean':
            score = torch.mean(data, dim=1)
        else:
            raise ValueError(f'Unknown score name {name}.')

        return score

    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False,
        return_only_a_score: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        with torch.no_grad():
            x_hat, P_layers, S_layers = self.model(x)
            ad = F.softmax(
                -self.model.association_discrepancy(
                    P_layers, S_layers),
                dim=1
            )
            x_diff = x - x_hat

        assert ad.shape[1] == self.model.N

        norm = torch.linalg.norm(x_diff, ord=2, dim=2)
        # norm = torch.tensor(
        #     [
        #         torch.linalg.norm(x[:, i, :] - x_hat[:, i, :], ord=2)
        #         for i in range(self.model.N)
        #     ]
        # )

        assert norm.shape[1] == self.model.N

        a_score = torch.mul(ad, norm)
        if return_only_a_score:
            return a_score

        res_scores = self.get_selected_scores(
            self.score_names, x_diff, a_score)
        # res_scores = a_score.tolist()
        if scale:
            res_scores = self.scores_scaler.transform(res_scores).tolist()
        if return_pred:
            return res_scores, x_hat
        return res_scores

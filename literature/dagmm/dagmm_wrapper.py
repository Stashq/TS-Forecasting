import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from typing import Dict, Generator, List

from .dagmm import DAGMM
from predpy.wrapper import Reconstructor


class DAGMMWrapper(Reconstructor):
    def __init__(
        self,
        model: DAGMM = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        params_to_train: Generator[Parameter, None, None] = None,
        lambda_energy: float = 0.1,
        lambda_cov_diag: float = 0.005
    ):
        super(DAGMMWrapper, self).__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=params_to_train
        )
        self.lambda_energy = lambda_energy
        self.lambda_cov_diag = lambda_cov_diag

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, _ = self.get_Xy(batch)

        x_hat, _, z, gamma = self(x)
        loss, _, _, _ = self.get_loss(
            x, x_hat, z, gamma)
        return loss

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, emb):
        return self.model.decode(emb)

    def get_loss(
        self, x, x_hat, z, gamma
    ):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.model.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.model.compute_energy(z, phi, mu, cov)
        loss = recon_error + self.lambda_energy * sample_energy\
            + self.lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag

    def predict(self, x):
        with torch.no_grad():
            x_hat, _, _, _ = self.model(x)
            return x_hat

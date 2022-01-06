"""Predictor - LightningModule wrapper for a model.

To work properly, dataset module __getitem__ method should return
dictionary with model input sequence named "sequence" and following after
it target value named "label". Compatibile with :py:mod:`dataset`.
"""
import torch
from torch import nn, optim
from typing import Dict, List
# from torch.autograd import Variable

from .vae import VAE


class PVAE(VAE):
    """Lightning module with functionalities for time series prediction
    models.\n

    To work properly, dataset module __getitem__ method should return
    dictionary with model input sequence named "sequence" and following after
    it target value named "label". It also has to have method called
    *get_labels* which return labels from selected range. Class is compatibile
    with dataset classes from :py:mod:`dataset`.

    Parameters
    ----------
    LightningModule : [type]
        Lightning module class.
    """
    def __init__(
        self,
        model: nn.Module = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        kld_weight: float = 0.1,
        sig_weight: float = 1.0,
        sigma_tuning_coef: float = 1.0
    ):
        super().__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass, optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            kld_weight=kld_weight)
        self.sigma_tuning = False
        self.sigma_tuning_coef = sigma_tuning_coef
        self.sig_weight = sig_weight

    def enable_sigma_tuning(self):
        self.sigma_tuning = True
        self.params_to_train = [
            self.model.x_log_sig_dense.bias,
            self.model.x_log_sig_dense.weight
        ]

    def disable_sigma_tuning(self):
        self.sigma_tuning = False
        for param in self.model.parameters():
            param.requires_grad = True
        self.params_to_train = None

    def get_loss(
        self,
        recons: torch.Tensor,
        input: torch.Tensor,
        z_mu: torch.Tensor,
        z_log_sig: torch.Tensor,
        x_mu: torch.Tensor,
        x_log_sig: torch.Tensor,
        tuning: bool = False
    ) -> dict:
        kld_loss = self.get_kld_loss(z_mu, z_log_sig)
        sig_loss = self.get_sigma_loss(input, x_mu, x_log_sig)
        loss = self.kld_weight * kld_loss\
            + kld_loss.detach() * self.sig_weight * sig_loss
        if tuning is False:
            recons_loss = self.criterion(recons, input)
            loss = loss + recons_loss
        return loss

    def get_sigma_loss(self, x, x_mu, x_log_sig):
        # x_sig = Variable(torch.exp(x_log_sig), requires_grad=True)
        x_sig = torch.exp(x_log_sig)
        # err = Variable(torch.abs(x - x_mu), requires_grad=True)
        err = torch.abs(x - x_mu)
        return self.criterion(err, self.sigma_tuning_coef * x_sig)

    def step(self, batch):
        sequences, _ = self.get_Xy(batch)

        x_tilda, z_mu, z_log_sig, x_mu, x_log_sig = self(sequences)
        loss = self.get_loss(
            recons=x_tilda, input=sequences, z_mu=z_mu, z_log_sig=z_log_sig,
            x_mu=x_mu, x_log_sig=x_log_sig, tuning=self.sigma_tuning)
        return loss

    def predict(
        self,
        sequence,
        get_x_log_sig: bool = False,
        get_z_log_sig: bool = False
    ):
        with torch.no_grad():
            _, _, z_log_sig, x_mu, x_log_sig = self(sequence)

        if get_x_log_sig & get_z_log_sig:
            res = (x_mu, x_log_sig, z_log_sig)
        elif get_x_log_sig:
            res = (x_mu, x_log_sig)
        elif get_z_log_sig:
            res = (x_mu, x_log_sig)
        else:
            res = x_mu
        return res

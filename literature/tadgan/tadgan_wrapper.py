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
        params_to_train: Generator[Parameter, None, None] = None
    ):
        super(TADGANWrapper, self).__init__(
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

    def wasserstein_loss(self, real, fake):
        """real and fake are predictions"""
        loss_real = torch.mean(real)
        loss_fake = torch.mean(fake)
        loss = loss_real - loss_fake
        return loss

    def get_mse_loss(self, x, z_enc=None):
        if z_enc is not None:
            seq_len = x.size(1)
            x_hat = self.model.decoder(z_enc, seq_len=seq_len)
        else:
            x_hat = self.model(x)
        loss_mse = self.mse(x, x_hat)
        return loss_mse

    def get_gp_loss(self, gradient):
        pass

    def get_ws_loss(
        self, x_real_score, x_fake_score,
        z_real_score, z_fake_score
    ):
        loss_x = self.wasserstein_loss(x_real_score, x_fake_score)
        loss_z = self.wasserstein_loss(z_real_score, z_fake_score)
        return loss_x, loss_z

    def step(self, batch, training: bool = True, mse: bool = False):
        x, _ = self.get_Xy(batch)
        batch_size, seq_len = x.shape[:2]
        res = ()

        if training:
            z = torch.empty(batch_size, self.model.z_size).uniform_(0, 1)
            z = z.to(self.device)
            x_hat = self.model.decoder(z, seq_len=seq_len)
            z_enc = self.model.encoder(x)

            x_real_score = self.model.critic_x(x)
            x_fake_score = self.model.critic_x(x_hat)

            z_real_score = self.model.critic_z(z_enc)
            z_fake_score = self.model.critic_z(z)

            loss_x = self.wasserstein_loss(
                real=x_real_score, fake=x_fake_score)
            loss_z = self.wasserstein_loss(
                real=z_real_score, fake=z_fake_score)
            res = (loss_x, loss_z)
        if mse:
            if training:
                x_hat2 = self.model.decoder(z_enc, seq_len=seq_len)
            else:
                x_hat2 = self.model(x)
            loss_mse = self.mse(x, x_hat2)
            res += (loss_mse,)

        return res

    def predict(self, x):
        with torch.no_grad():
            x_hat = self.model(x)
            return x_hat

    def configure_optimizers(self):
        opt_g = self.OptimizerClass([
            {'params': self.model.encoder.parameters()},
            {'params': self.model.decoder.parameters()}],
            lr=self.lr, **self.optimizer_kwargs)
        opt_d = self.OptimizerClass([
            {'params': self.model.critic_x.parameters()},
            {'params': self.model.critic_z.parameters()}],
            lr=self.lr, **self.optimizer_kwargs)
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # train generators
        if optimizer_idx == 0:
            loss_x, loss_z, loss_mse = self.step(batch, mse=True)
            loss = loss_x + loss_z + loss_mse
            m_name = "gen"
        # train discriminators
        elif optimizer_idx == 1:
            loss_x, loss_z = self.step(batch)
            loss_x = loss_x * (-1)
            loss_z = loss_z * (-1)
            loss = loss_x + loss_z
            m_name = "dis"
        self.log(m_name + "_x_loss", loss_x, prog_bar=True, logger=True)
        self.log(m_name + "_z_loss", loss_z, prog_bar=True, logger=True)
        self.log(m_name + "_train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, training=False, mse=True)[0]
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, training=False, mse=True)[0]
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

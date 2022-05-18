from torch import nn, optim
from typing import Dict, List, Tuple
import torch
from torch.autograd import Variable

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
        gen_dis_train_loops: Tuple[int] = (1, 3)
    ):
        super(TADGANWrapper, self).__init__(
            model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=None
        )
        self.mse = nn.MSELoss()
        self.gen_loops, self.dis_loops = gen_dis_train_loops

    @property
    def automatic_optimization(self) -> bool:
        return False

    def get_loss(self):
        pass

    def step(self):
        pass

    def get_mse_loss(self, x, z_enc=None):
        if z_enc is not None:
            seq_len = x.size(1)
            x_hat = self.model.decoder(z_enc, seq_len=seq_len)
        else:
            x_hat = self.model(x)
        loss_mse = self.mse(x, x_hat)
        return loss_mse

    def get_gp_loss(self, x, x_gen):
        alpha = torch.rand(x.shape)
        # Random Weighted Average
        ix = Variable(alpha * x + (1 - alpha) * x_gen)
        ix.requires_grad_(True)
        v_ix = self.model.critic_x(ix)
        # self.manual_backward(
        #     min_loss,
        #     retain_graph=True
        # )
        v_ix.mean().backward(retain_graph=True)
        gradients = ix.grad
        gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))
        return gp_loss

    def wasserstein_loss(self, real, fake):
        """real and fake are predictions"""
        loss_real = torch.mean(real)
        loss_fake = torch.mean(fake)
        loss = loss_real - loss_fake
        return loss

    def get_ws_loss(
        self, x_real_score, x_fake_score,
        z_real_score, z_fake_score
    ):
        loss_x = self.wasserstein_loss(x_real_score, x_fake_score)
        loss_z = self.wasserstein_loss(z_real_score, z_fake_score)
        return loss_x, loss_z

    def critic_step(
        self, x, include_mse: bool = False,
        include_gp: bool = False
    ):
        batch_size, seq_len = x.shape[:2]
        z = torch.empty(batch_size, self.model.z_size)\
            .uniform_(0, 1).to(self.device)
        x_gen = self.model.decoder(z, seq_len=seq_len)
        z_enc = self.model.encoder(x)

        x_real_score = self.model.critic_x(x)
        x_fake_score = self.model.critic_x(x_gen)

        z_real_score = self.model.critic_z(z_enc)
        z_fake_score = self.model.critic_z(z)

        loss_x = self.wasserstein_loss(
            real=x_real_score, fake=x_fake_score)
        loss_z = self.wasserstein_loss(
            real=z_real_score, fake=z_fake_score)

        if include_mse:
            x_hat = self.model.decoder(z_enc, seq_len=seq_len)
            loss_mse = self.mse(x, x_hat)
            loss_x += loss_mse
            loss_z += loss_mse
        if include_gp:
            gp_loss = self.get_gp_loss(x, x_gen)
            loss_x += gp_loss
            loss_z += gp_loss
        return loss_x, loss_z

    def mse_step(self, x):
        x_hat = self.model(x)
        loss_mse = self.mse(x, x_hat)
        return loss_mse

    def predict(self, x):
        with torch.no_grad():
            x_hat = self.model(x)
            return x_hat

    def configure_optimizers(self):
        opt_enc = self.OptimizerClass(
            self.model.encoder.parameters(),
            lr=self.lr, **self.optimizer_kwargs)
        opt_dec = self.OptimizerClass(
            self.model.decoder.parameters(),
            lr=self.lr, **self.optimizer_kwargs)
        opt_cr_x = self.OptimizerClass(
            self.model.critic_x.parameters(),
            lr=self.lr, **self.optimizer_kwargs)
        opt_cr_z = self.OptimizerClass(
            self.model.critic_z.parameters(),
            lr=self.lr, **self.optimizer_kwargs)
        return [opt_enc, opt_dec, opt_cr_x, opt_cr_z]

    def training_log(
        self, losses: Dict[str, float]
    ):
        for name, value in losses.items():
            self.log(name, value, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        x, _ = self.get_Xy(batch)

        opt_enc, opt_dec, opt_cr_x, opt_cr_z =\
            self.configure_optimizers()
        for _ in range(self.gen_loops):
            opt_enc.zero_grad()
            opt_dec.zero_grad()

            loss_x, loss_z = self.critic_step(
                x, include_mse=True)

            self.training_log({
                'train_gen_loss_x': loss_x,
                'train_gen_loss_z': loss_z
            })

            loss_x.backward()
            loss_z.backward()
            opt_enc.step()
            opt_dec.step()
        for _ in range(self.dis_loops):
            opt_cr_x.zero_grad()
            opt_cr_z.zero_grad()

            loss_x, loss_z = self.critic_step(
                x, include_gp=True)

            self.training_log({
                'train_dis_loss_x': loss_x,
                'train_dis_loss_z': loss_z
            })

            loss_x.backward()
            loss_z.backward()
            opt_cr_x.step()
            opt_cr_z.step()

    def validation_step(self, batch, batch_idx):
        x, _ = self.get_Xy(batch)
        loss = self.get_mse_loss(x)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = self.get_Xy(batch)
        loss = self.get_mse_loss(x)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

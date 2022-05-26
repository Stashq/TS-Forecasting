from torch import nn, optim
from typing import Dict, List, Tuple, Literal, Union
import torch
from torch.autograd import Variable

from predpy.wrapper import Reconstructor
from literature.anomaly_detector_base import AnomalyDetector
from .tadgan import TADGAN


class TADGANWrapper(Reconstructor, AnomalyDetector):
    def __init__(
        self,
        model: TADGAN = nn.Module(),
        lr: float = 1e-4,
        criterion: nn.Module = nn.MSELoss(),
        OptimizerClass: optim.Optimizer = optim.Adam,
        optimizer_kwargs: Dict = {},
        target_cols_ids: List[int] = None,
        gen_dis_train_loops: Tuple[int] = (1, 3),
        warmup_epochs: int = 0,
        alpha: float = 0.5
    ):
        AnomalyDetector.__init__(self)
        Reconstructor.__init__(
            self, model=model, lr=lr, criterion=criterion,
            OptimizerClass=OptimizerClass,
            optimizer_kwargs=optimizer_kwargs,
            target_cols_ids=target_cols_ids,
            params_to_train=None
        )
        self.mse = nn.MSELoss()
        self.gen_loops, self.dis_loops = gen_dis_train_loops
        self.warmup_epochs = warmup_epochs

    @property
    def automatic_optimization(self) -> bool:
        return False

    def forward(self, x):
        return self.model(x)

    def get_loss(self):
        pass

    def step(self):
        pass

    # def get_mse_loss(self, x, z_enc=None):
    #     if z_enc is not None:
    #         seq_len = x.size(1)
    #         x_hat = self.model.decoder(z_enc, seq_len=seq_len)
    #     else:
    #         x_hat = self.model(x)
    #     loss_mse = self.mse(x, x_hat)
    #     return loss_mse

    def get_gp_loss(self, a, a_gen, name: Literal['x', 'z']):
        if name == 'x':
            alpha = torch.rand(a.shape).to(self.device)
            # Random Weighted Average
            ix = Variable(alpha * a + (1 - alpha) * a_gen)
            ix.requires_grad_(True)
            v_ix = self.model.critic_x(ix)
            v_ix.mean().backward(retain_graph=True)
            gradients = ix.grad
            gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))
        elif name == 'z':
            alpha = torch.rand(a.shape).to(self.device)
            # Random Weighted Average
            iz = Variable(alpha * a + (1 - alpha) * a_gen)
            iz.requires_grad_(True)
            v_iz = self.model.critic_z(iz)
            v_iz.mean().backward(retain_graph=True)
            gradients = iz.grad
            gp_loss = torch.sqrt(torch.sum(torch.square(gradients).view(-1)))
        else:
            raise ValueError(
                'Unknown variable name "%s". Allowed: "x" or "z"' % name)
        return gp_loss

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

    def calculate_loss_x(self, x, model_part: Literal['gen', 'dis']):
        batch_size, seq_len = x.shape[:2]
        real_x = self.model.critic_x(x)
        # Wasserstein Loss
        real_x_score = torch.mean(real_x)

        # z = torch.empty(batch_size, self.model.z_size)\
        #     .uniform_(0, 1).to(self.device)
        z = torch.empty(batch_size, seq_len, self.model.z_size)\
            .uniform_(0, 1).to(self.device)
        x_gen = self.model.decoder(z, seq_len=seq_len)
        fake_x = self.model.critic_x(x_gen)
        # Wasserstein Loss
        fake_x_score = torch.mean(fake_x)

        if model_part == 'dis':
            gp_loss = self.get_gp_loss(x, x_gen, 'x')
            loss = gp_loss + fake_x_score - real_x_score
        elif model_part == 'gen':
            mse_loss = self.mse_step(x)
            loss = mse_loss + real_x_score - fake_x_score
        else:
            raise ValueError(
                'Unknown model_part "%s". Allowed: "gen" or "dis".'
                % model_part)
        return loss

    def calculate_loss_z(self, x, model_part: Literal['gen', 'dis']):
        batch_size, seq_len = x.shape[:2]
        z_enc = self.model.encoder(x)
        real_z = self.model.critic_z(z_enc)
        # Wasserstein Loss
        real_z_score = torch.mean(real_z)

        # z = torch.empty(batch_size, self.model.z_size)\
        #     .uniform_(0, 1).to(self.device)
        z = torch.empty(batch_size, seq_len, self.model.z_size)\
            .uniform_(0, 1).to(self.device)
        fake_z = self.model.critic_z(z)
        # Wasserstein Loss
        fake_z_score = torch.mean(fake_z)

        if model_part == 'dis':
            gp_loss = self.get_gp_loss(z_enc, z, 'z')
            loss = gp_loss + fake_z_score - real_z_score
        elif model_part == 'gen':
            mse_loss = self.mse_step(x)
            loss = mse_loss + real_z_score - fake_z_score
        else:
            raise ValueError(
                'Unknown model_part "%s". Allowed: "gen" or "dis".'
                % model_part)
        return loss

    def substep(
        self, x, opt, model_part: Literal['gen', 'dis'],
        var_name: Literal['x', 'z']
    ):
        opt.zero_grad()
        if self.current_epoch < self.warmup_epochs:
            loss = self.mse_step(x)
            self.training_log({
                'warmup_mse': loss.float()})
        else:
            if var_name == 'x':
                loss = self.calculate_loss_x(x, model_part=model_part)
            elif var_name == 'z':
                loss = self.calculate_loss_z(x, model_part=model_part)
            else:
                raise ValueError(
                    'Unknown variable name "%s". Allowed: "x" or "z"'
                    % var_name)
            self.training_log({
                var_name + '_' + model_part + '_train': loss.float()})
        loss.backward(retain_graph=True)
        opt.step()

        return loss

    def training_step(self, batch, batch_idx):
        x, _ = self.get_Xy(batch)

        opt_enc, opt_dec, opt_cr_x, opt_cr_z =\
            self.configure_optimizers()
        for _ in range(self.gen_loops):
            self.substep(x, opt_enc, 'gen', 'x')
            self.substep(x, opt_dec, 'gen', 'z')
        for _ in range(self.dis_loops):
            self.substep(x, opt_cr_z, 'dis', 'z')
            self.substep(x, opt_cr_x, 'dis', 'x')

    def anomaly_score(
        self, x, scale: bool = True, return_pred: bool = False
    ) -> Union[List[float], Tuple[List[float], List[torch.Tensor]]]:
        batch_size = x.size(0)
        with torch.no_grad():
            x_hat = self.model(x)
            loss_mse = torch.linalg.norm(
                (x - x_hat).reshape(batch_size, -1), ord=1, dim=1)
            loss_x = self.model.critic_x(x_hat)
            score = self.alpha * loss_mse + (1 - self.alpha) * loss_x

        score = score.tolist()
        if scale:
            score = self.scores_scaler.transform(score).flatten().tolist()
        if return_pred:
            return score, x_hat
        return score

    # def validation_step(self, batch, batch_idx):
    #     x, _ = self.get_Xy(batch)
    #     loss = self.get_mse_loss(x)
    #     self.log("val_loss", loss, prog_bar=True, logger=True)
    #     return loss

    # def test_step(self, batch, batch_idx):
    #     x, _ = self.get_Xy(batch)
    #     loss = self.get_mse_loss(x)
    #     self.log("test_loss", loss, prog_bar=True, logger=True)
    #     return loss

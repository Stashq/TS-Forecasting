from typing import List
import numpy as np
import torch
# from torch.nn.functional import mse_loss
from torch import nn
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
from torch.nn import functional as F
import pytorch_lightning as pl
from .base import Regressor


class NNRegressor(pl.LightningModule, Regressor):
    def __init__(self, c_in: int, h_sizes: List[int] = [128, 256]):
        super().__init__()

        layers = [
            nn.Linear(c_in, h_sizes[0]), nn.ReLU()
        ]
        for i in range(1, len(h_sizes)):
            layers += [
                nn.Linear(h_sizes[i-1], h_sizes[i]), nn.ReLU()
            ]
        layers += [
            nn.Linear(h_sizes[-1], 1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def predict(self, x) -> np.ndarray:
        res = self(x).cpu().detach().numpy()
        return res

    def step(self, batch):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

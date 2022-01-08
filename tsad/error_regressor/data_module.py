import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pytorch_lightning as pl


class RegressionDataModule(pl.LightningDataModule):
    def __init__(self, x, y, splits=[0.8, 0.1, 0.1], batch_size=64):
        super().__init__()
        self.x = x
        self.y = y
        self.splits = splits
        self.batch_size = batch_size

    def setup(self, stage):
        dataset = TensorDataset(self.x, self.y)
        len_ = len(dataset)
        splits = [int(frac * len_) for frac in self.splits]
        if len(dataset) != sum(splits):
            splits[0] += len(dataset) - sum(splits)
        self.train_ds, self.val_ds, self.test_ds =\
            torch.utils.data.random_split(
                dataset, splits)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

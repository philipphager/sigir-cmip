from typing import Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ClickModelDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: Dict,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["val"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.datasets["test_clicks"],
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            ),
            DataLoader(
                self.datasets["test_rels"], batch_size=self.batch_size, num_workers=0
            ),
        ]

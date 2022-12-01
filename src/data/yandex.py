from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import ParquetRelevanceDataset, ParquetClickDataset


class Yandex(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        n_documents: int,
        n_results: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.n_documents = n_documents
        self.n_results = n_results

        self.dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ParquetRelevanceDataset(
            "/Users/philipphager/.ltr_datasets/cache/relevance.parquet",
            self.batch_size,
        )

        self.click_train = ParquetClickDataset(
            "/Users/philipphager/.ltr_datasets/cache/clicks.parquet",
            self.batch_size,
        )

        if stage == TrainerFn.FITTING:
            self.train_clicks = self.click_train
            self.val_clicks = self.click_train

        elif stage == TrainerFn.TESTING:
            self.test_clicks = self.click_train

    def train_dataloader(self):
        return DataLoader(
            self.train_clicks,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_clicks,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.test_clicks,
                batch_size=None,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            ),
            DataLoader(
                self.dataset,
                batch_size=None,
                num_workers=0,
            ),
        ]

    def get_n_documents(self) -> int:
        return self.n_documents

    def get_n_results(self) -> int:
        return self.n_results

    def get_train_policy_scores(self) -> Optional[Tensor]:
        return None

    def get_test_policy_scores(self) -> Optional[Tensor]:
        return None

    def assert_setup(self):
        assert self.dataset is not None, "Call setup(stage='fit') first"

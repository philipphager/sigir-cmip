from typing import Optional, Tuple

import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import ParquetClickDataset, RatingDataset
from src.data.loader.base import Loader
from src.util.hydra import ConfigWrapper


class Yandex(pl.LightningDataModule):
    def __init__(
        self,
        rating_loader: Loader[RatingDataset],
        click_loader: Loader[ParquetClickDataset],
        config_wrapper: ConfigWrapper,
        train_val_test_split: Tuple[int, int, int],
        shuffle_clicks: bool,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        n_documents: int,
        n_results: int,
    ):
        super().__init__()
        self.rating_loader = rating_loader
        self.click_loader = click_loader
        self.config = config_wrapper.config
        self.train_val_test_split = train_val_test_split
        self.shuffle_clicks = shuffle_clicks
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
        self.dataset = self.rating_loader.load(split="train")
        self.clicks = self.click_loader.load(split="train", batch_size=256)

        assert (
            sum(self.train_val_test_split) == 1.0
        ), "Splits for train, val, test clicks must sum to 1.0"
        train_split, val_split, test_split = self.train_val_test_split

        self.train_clicks, self.test_clicks = self.clicks.split(
            train_size=train_split,
            shuffle=self.shuffle_clicks,
        )
        self.val_clicks, self.test_clicks = self.test_clicks.split(
            train_size=val_split / (val_split + test_split),
            shuffle=self.shuffle_clicks,
        )

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

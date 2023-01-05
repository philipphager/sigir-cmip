import logging
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.dataset import ClickDataset, ClickDatasetStats, RatingDataset
from src.data.loader.base import Loader
from src.util.cache import cache
from src.util.hydra import ConfigWrapper
from src.util.tensor import scatter_rank_add

logger = logging.getLogger(__name__)


class Yandex(pl.LightningDataModule):
    def __init__(
        self,
        rating_loader: Loader[RatingDataset],
        click_loader: Loader[ClickDataset],
        config_wrapper: ConfigWrapper,
        train_val_test_split: Tuple[int, int, int],
        shuffle_clicks: bool,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        n_documents: int,
        n_results: int,
        random_state: int,
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
        self.random_state = random_state

        self.dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        @cache(
            self.config.base_dir,
            "cache/train_click_stats",
            [
                self.config.data,
                self.config.random_state,
            ],
        )
        def get_train_click_stats():
            logger.info("Compute click statistics on Yandex")
            rank_clicks = torch.zeros((self.n_documents, self.n_results))
            rank_impressions = torch.zeros((self.n_documents, self.n_results))

            for batch in self.train_clicks:
                query_ids, x, y_click, n = batch
                impressions = (x > 0).float()
                rank_clicks += scatter_rank_add(y_click, x, self.n_documents)
                rank_impressions += scatter_rank_add(impressions, x, self.n_documents)

            return ClickDatasetStats(rank_clicks, rank_impressions)

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

        self.train_stats = get_train_click_stats()

    def train_dataloader(self):
        return DataLoader(
            self.train_clicks,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_clicks,
                batch_size=None,  # ParquetClickDataset takes care of batching.
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            ),
            DataLoader(
                self.dataset,
                batch_size=len(self.dataset),
                num_workers=0,
            ),
        ]

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
                batch_size=len(self.dataset),
                num_workers=0,
            ),
        ]

    def get_n_documents(self) -> int:
        return self.n_documents

    def get_n_results(self) -> int:
        return self.n_results

    def get_train_stats(self) -> ClickDatasetStats:
        return self.train_stats

    def get_train_policy_scores(self) -> Optional[Tensor]:
        return None

    def get_test_policy_scores(self) -> Optional[Tensor]:
        return None

    def assert_setup(self):
        assert self.dataset is not None, "Call setup(stage='fit') first"

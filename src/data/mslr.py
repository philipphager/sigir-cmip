import logging
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.dataset import ClickDatasetStats, RatingDataset
from src.data.loader.base import Loader
from src.data.simulation import Simulator
from src.data.simulation.logging_policy import LoggingPolicy
from src.util.cache import cache
from src.util.hydra import ConfigWrapper
from src.util.tensor import scatter_rank_add

logger = logging.getLogger(__name__)


class MSLR(pl.LightningDataModule):
    def __init__(
        self,
        rating_loader: Loader[RatingDataset],
        train_policy: LoggingPolicy,
        test_policy: LoggingPolicy,
        train_simulator: Simulator,
        val_simulator: Simulator,
        test_simulator: Simulator,
        config_wrapper: ConfigWrapper,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
        n_results: int,
        random_state: int,
    ):
        super().__init__()
        self.rating_loader = rating_loader
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.train_simulator = train_simulator
        self.val_simulator = val_simulator
        self.test_simulator = test_simulator
        self.config = config_wrapper.config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.n_results = n_results
        self.random_state = random_state

        self.dataset = None
        self.train_policy_scores = None
        self.test_policy_scores = None
        self.train_clicks = None
        self.val_clicks = None
        self.test_clicks = None
        self.train_stats = None

    def setup(self, stage: Optional[str] = None):
        @cache(
            self.config.base_dir,
            "cache/train_policy",
            [
                self.config.data.rating_loader,
                self.config.data.train_policy,
            ],
        )
        def train_policy_scores():
            dataset = self.rating_loader.load(
                split="train",
                load_features=self.train_policy.requires_features(),
            )
            self.train_policy.fit(dataset)
            return self.train_policy.predict(dataset)

        @cache(
            self.config.base_dir,
            "cache/test_policy",
            [
                self.config.data.rating_loader,
                self.config.data.test_policy,
            ],
        )
        def test_policy_scores():
            dataset = self.rating_loader.load(
                split="train",
                load_features=self.test_policy.requires_features(),
            )
            self.test_policy.fit(dataset)
            return self.test_policy.predict(dataset)

        @cache(
            self.config.base_dir,
            "cache/train_clicks",
            [
                self.config.data.rating_loader,
                self.config.data.train_policy,
                self.config.data.train_simulator,
            ],
        )
        def simulate_train():
            return self.train_simulator(self.dataset, self.train_policy_scores)

        @cache(
            self.config.base_dir,
            "cache/val_clicks",
            [
                self.config.data.rating_loader,
                self.config.data.train_policy,
                self.config.data.val_simulator,
            ],
        )
        def simulate_val():
            return self.val_simulator(self.dataset, self.train_policy_scores)

        @cache(
            self.config.base_dir,
            "cache/test_clicks",
            [
                self.config.data.rating_loader,
                self.config.data.test_policy,
                self.config.data.test_simulator,
            ],
        )
        def simulate_test():
            return self.test_simulator(self.dataset, self.test_policy_scores)

        @cache(
            self.config.base_dir,
            "cache/train_click_stats",
            [
                self.config.data.rating_loader,
                self.config.data.train_policy,
                self.config.data.train_simulator,
            ],
        )
        def get_train_click_stats():
            logger.info("Compute click statistics on MSLR")
            n_documents = self.get_n_documents()
            rank_clicks = scatter_rank_add(
                self.train_clicks.y_click, self.train_clicks.x, n_documents
            )
            rank_impressions = scatter_rank_add(
                (self.train_clicks.x > 0).float(), self.train_clicks.x, n_documents
            )

            return ClickDatasetStats(rank_clicks, rank_impressions)

        self.dataset = self.rating_loader.load(split="train")
        self.train_policy_scores = train_policy_scores()

        if stage == TrainerFn.FITTING:
            self.train_clicks = simulate_train()
            self.val_clicks = simulate_val()
            self.train_stats = get_train_click_stats()
        elif stage == TrainerFn.TESTING:
            self.test_policy_scores = test_policy_scores()
            self.test_clicks = simulate_test()

    def train_dataloader(self):
        return DataLoader(
            self.train_clicks,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                self.val_clicks,
                batch_size=self.batch_size,
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
                batch_size=self.batch_size,
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
        self.assert_setup()
        return self.dataset.n.sum()

    def get_n_queries(self) -> int:
        self.assert_setup()
        return self.dataset.query_id.max()

    def get_n_results(self) -> int:
        return self.n_results

    def get_train_stats(self) -> ClickDatasetStats:
        return self.train_stats

    def has_train_policy_scores(self):
        self.assert_setup()
        return self.train_policy_scores is not None

    def has_test_policy_scores(self):
        self.assert_setup()
        return self.test_policy_scores is not None

    def get_train_policy_scores(self):
        assert self.has_train_policy_scores()
        return self.train_policy_scores

    def get_test_policy_scores(self):
        assert self.has_test_policy_scores()
        return self.test_policy_scores

    def assert_setup(self):
        assert self.dataset is not None, "Call setup(stage='fit') first"

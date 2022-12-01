from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from src.data.loader import Loader
from src.data.simulation import Simulator
from src.data.simulation.logging_policy import LoggingPolicy


class MSLR(pl.LightningDataModule):
    def __init__(
        self,
        loader: Loader,
        train_policy: LoggingPolicy,
        test_policy: LoggingPolicy,
        train_simulator: Simulator,
        val_simulator: Simulator,
        test_simulator: Simulator,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        persistent_workers: bool,
    ):
        super().__init__()
        self.loader = loader
        self.train_policy = train_policy
        self.test_policy = test_policy
        self.train_simulator = train_simulator
        self.val_simulator = val_simulator
        self.test_simulator = test_simulator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.dataset = None
        self.train_policy_scores = None
        self.test_policy_scores = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = self.loader.load("train")

        if stage == TrainerFn.FITTING:
            self.train_policy.fit(self.dataset)
            self.train_policy_scores = self.train_policy.predict(self.dataset)

            self.train_clicks = self.train_simulator(
                self.dataset,
                self.train_policy_scores,
            )
            self.val_clicks = self.val_simulator(
                self.dataset,
                self.train_policy_scores,
            )

        elif stage == TrainerFn.TESTING:
            self.test_policy.fit(self.dataset)
            self.test_policy_scores = self.test_policy.predict(self.dataset)

            self.test_clicks = self.test_simulator(
                self.dataset,
                self.test_policy_scores,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_clicks,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_clicks,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

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
                batch_size=self.batch_size,
                num_workers=0,
            ),
        ]

    def get_n_documents(self) -> int:
        self.assert_setup()
        return self.dataset.n.sum()

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

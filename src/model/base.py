from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.data.dataset import ClickDatasetStats
from src.evaluation.base import ClickMetric, Metric, PolicyMetric, RelevanceMetric
from src.evaluation.util import join_metrics, sample_policy_data

CLICK_DATASET_IDX = 0


class ClickModel(LightningModule, ABC):
    def __init__(
        self,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        lp_scores: Optional[torch.FloatTensor],
    ):
        super().__init__()
        self.metrics = metrics
        self.n_results = n_results
        self.lp_scores = lp_scores
        self.generator = torch.Generator().manual_seed(random_state)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def validation_step(self, batch, idx, dl_idx: int):
        metrics = []

        if dl_idx == CLICK_DATASET_IDX:
            q, x, y_click, n = batch
            y_predict_click, _ = self.forward(q, x, true_clicks=y_click)

            loss = self.loss(y_predict_click, y_click, n)
            metrics += [{"loss": loss}]
            metrics += self._get_click_metrics(y_predict_click, y_click, n)

            click_probs = y_predict_click.mean(dim=0)
        else:
            click_probs = []
            q, x, y, n = batch
            y_predict = self.forward(q, x, click_pred=False)
            metrics += self._get_relevance_metrics(y_predict, y, n)

        metrics = join_metrics(metrics, stage="val")
        self.log_dict(metrics, logger=False)

        return metrics, click_probs

    def validation_epoch_end(self, outputs):
        """
        Log click probabilities.
        """
        click_outputs = outputs[0]
        click_metrics = [co[0] for co in click_outputs]
        click_metrics_name = click_metrics[0].keys()
        click_metrics = torch.stack(
            [torch.stack(list(op.values()), dim=0) for op in click_metrics], dim=0
        )
        click_metrics_dict = {
            name: val
            for name, val in zip(click_metrics_name, click_metrics.mean(dim=0))
        }
        metrics_dict = click_metrics_dict | outputs[1][0][0]
        self.logger.log_metrics(metrics_dict, step=self.current_epoch)

        click_probs = torch.stack([co[1] for co in click_outputs]).mean(dim=0)
        self.logger.log_table(
            key="Appendix/click_probs",
            columns=[str(i) for i in range(1, self.n_results + 1)],
            data=[click_probs.tolist()],
            step=self.current_epoch,
        )

    def test_step(self, batch, idx: int, dl_idx: int):
        metrics = []

        if dl_idx == CLICK_DATASET_IDX:
            q, x, y_click, n = batch
            y_predict_click, _ = self.forward(q, x, true_clicks=y_click)
            loss = self.loss(y_predict_click, y_click, n)

            metrics += [{"loss": loss}]
            metrics += self._get_click_metrics(y_predict_click, y_click, n)
        else:
            q, x, y, n = batch
            y_predict = self.forward(q, x, click_pred=False)

            y_lp = self.lp_scores.to(self.device)
            metrics += self._get_relevance_metrics(y_predict, y, n)

            # FIXME: Remove null check when Yandex has a logging policy.
            if self.lp_scores is not None:
                metrics += self._get_policy_metrics(y_predict, y_lp, y, n)
                policy_df = sample_policy_data(y_predict, y_lp, y, n, n_samples=50_000)

                self.logger.log_table(
                    key="Appendix/policy",
                    dataframe=policy_df,
                    step=self.current_epoch,
                )

        metrics = join_metrics(metrics, stage="test")
        self.log_dict(metrics, logger=False)

        return metrics

    def test_epoch_end(self, outputs):
        """
        Log click probabilities.
        """
        click_metrics_name = outputs[0][0].keys()
        click_metrics = torch.stack(
            [torch.stack(list(op.values()), dim=0) for op in outputs[0]], dim=0
        )
        click_metrics_dict = {
            name: val
            for name, val in zip(click_metrics_name, click_metrics.mean(dim=0))
        }
        metrics_dict = click_metrics_dict | outputs[1][0]
        self.logger.log_metrics(metrics_dict, step=self.current_epoch)

    def _get_click_metrics(self, y_predict_click, y_click, n) -> List[Dict[str, float]]:
        return [
            metric(y_predict_click, y_click, n)
            for metric in self.metrics
            if isinstance(metric, ClickMetric)
        ]

    def _get_relevance_metrics(self, y_predict, y_true, n) -> List[Dict[str, float]]:
        return [
            metric(y_predict, y_true, n)
            for metric in self.metrics
            if isinstance(metric, RelevanceMetric)
        ]

    def _get_policy_metrics(
        self, y_predict, y_logging_policy, y, n
    ) -> List[Dict[str, float]]:
        return [
            metric(y_predict, y_logging_policy, y, n)
            for metric in self.metrics
            if isinstance(metric, PolicyMetric)
        ]


class NeuralClickModel(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        lp_scores: Optional[torch.FloatTensor] = None,
    ):
        super().__init__(metrics, n_results, random_state, lp_scores)
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lp_scores = lp_scores

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def training_step(self, batch, idx):
        q, x, y_click, n = batch

        y_predict_click, _ = self.forward(q, x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = [{"loss": loss}]
        metrics = join_metrics(metrics, "train")
        self.log_dict(metrics, logger=False)
        if idx % self.trainer.log_every_n_steps == 0:
            self.logger.log_metrics(metrics, step=self.global_step)

        return loss


class StatsClickModel(ClickModel, ABC):
    """
    Base class for non-trainable click models. All parameters have to be set during
    initialization and fit() cannot be called.
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        train_stats: ClickDatasetStats,
        lp_scores: Optional[torch.FloatTensor] = None,
    ):
        super().__init__(metrics, n_results, random_state, lp_scores)
        self.loss = loss
        self.lp_scores = lp_scores

        self.setup_parameters(train_stats)
        self.freeze()

    @abstractmethod
    def setup_parameters(self, train_stats: ClickDatasetStats):
        pass

    def configure_optimizers(self):
        raise NotImplementedError("Optimization is not supported")

    def training_step(self, batch, idx):
        raise NotImplementedError("Training is not supported, use __init__() instead")

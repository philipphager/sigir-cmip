from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import nn

from src.data.dataset import ClickDatasetStats
from src.evaluation.metrics import get_click_metrics, get_relevance_metrics

CLICK_DATASET_IDX = 0


class ClickModel(LightningModule, ABC):
    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def validation_step(self, batch, idx, dl_idx: int):
        if dl_idx == CLICK_DATASET_IDX:
            q, x, y_click, n = batch
            y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
            metrics = get_click_metrics(y_predict_click, y_click, n, "val_")
            metrics["val_loss"] = self.loss(y_predict_click, y_click, n)
        else:
            query_ids, x, y, n = batch
            y_predict = self.forward(x, click_pred=False)
            metrics = get_relevance_metrics(y_predict, y, None, n, "val_")

        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, idx: int, dl_idx: int):
        if dl_idx == CLICK_DATASET_IDX:
            q, x, y_click, n = batch
            y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
            loss = self.loss(y_predict_click, y_click, n)

            metrics = get_click_metrics(y_predict_click, y_click, n, "test_clicks_")
            metrics["test_loss"] = loss
            self.log_dict(metrics)
        else:
            query_ids, x, y, n = batch
            y_predict = self.forward(x, click_pred=False)
            y_lp = self.lp_scores[idx * len(query_ids) : (idx + 1) * len(query_ids)].to(
                self.device
            )
            metrics = get_relevance_metrics(y_predict, y, y_lp, n, "test_rels_")
            self.log_dict(
                {
                    key: val
                    for key, val in metrics.items()
                    if key not in ["test_rels_agreement_ratio", "n_pairs"]
                }
            )

        return metrics

    def test_epoch_end(self, outputs):
        # Weighted sum to account for different number of pairs in different batches
        agreement_ratios = torch.stack(
            [metrics["test_rels_agreement_ratio"] for metrics in outputs[1]]
        )
        n_pairs = torch.stack(
            [
                torch.tensor(metrics["n_pairs"], device=self.device)
                for metrics in outputs[1]
            ]
        )
        test_agreement_ratio = torch.sum(agreement_ratios * n_pairs) / torch.sum(
            n_pairs
        )
        self.log("test_rels_agreement_ratio", test_agreement_ratio)


class NeuralClickModel(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        lp_scores: Optional[torch.FloatTensor] = None,
    ):
        super().__init__()
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

        y_predict_click, _ = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = get_click_metrics(y_predict_click, y_click, n, "train_")
        metrics["train_loss"] = loss
        self.log_dict(metrics)

        return loss


class StatsClickModel(ClickModel, ABC):
    """
    Base class for non-trainable click models. All parameters have to be set during
    initialization and fit() cannot be called.
    """

    def __init__(
        self,
        loss: nn.Module,
        train_stats: ClickDatasetStats,
        lp_scores: Optional[torch.FloatTensor] = None,
    ):
        super().__init__()
        self.loss = loss
        self.lp_scores = lp_scores
        self.train_stats = train_stats

        self.setup_parameters(train_stats)
        self.freeze()

    @abstractmethod
    def setup_parameters(self, train_stats: ClickDatasetStats):
        pass

    def configure_optimizers(self):
        raise NotImplementedError("Optimization is not supported")

    def training_step(self, batch, idx):
        raise NotImplementedError("Training is not supported, use __init__() instead")

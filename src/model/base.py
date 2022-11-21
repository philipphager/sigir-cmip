from abc import abstractmethod
from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from src.evaluation.metrics import get_click_metrics, get_relevance_metrics


class ClickModel(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        lp_scores: torch.FloatTensor = None,
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

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch

        y_predict_click, _ = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = get_click_metrics(y_predict_click, y_click, n, "train_")
        metrics["train_loss"] = loss
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, idx):
        q, x, y, y_click, n = batch

        y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        click_metrics = get_click_metrics(y_predict_click, y_click, n, "val_")
        relevance_metrics = get_relevance_metrics(y_predict, y, None, n, "val_")
        metrics = click_metrics | relevance_metrics
        metrics["val_loss"] = loss
        self.log_dict(metrics)

        return loss

    def test_step(self, batch, idx, dl_idx):
        if dl_idx == 0:
            # Click dataset
            q, x, y, y_click, n = batch
            y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
            loss = self.loss(y_predict_click, y_click, n)

            metrics = get_click_metrics(y_predict_click, y_click, n, "test_clicks_")
            metrics["test_loss"] = loss
        else:
            # Rating dataset
            query_ids, x, y, n = batch
            print(query_ids)
            y_predict = self.forward(x, click_pred=False)
            y_lp = self.lp_scores[query_ids].to(self.device)
            metrics = get_relevance_metrics(y_predict, y, y_lp, n, "test_rels_")

        # self.log_dict(metrics)
        return metrics

    def test_epoch_end(self, outputs):
        print(outputs)

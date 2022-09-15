from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import nn

from src.evaluation.metrics import get_metrics


class PBM(pl.LightningModule):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
    ):
        super().__init__()
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_results = n_results

        self.relevance = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.examination = nn.Sequential(nn.Embedding(n_results, 1), nn.Sigmoid())

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "adagrad":
            return torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def forward(
        self, x: torch.Tensor, click_pred: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_items = x.shape

        relevance = self.relevance(x)
        if click_pred:
            ranks = torch.arange(n_items, device=self.device).repeat(n_batch, 1)
            examination = self.examination(ranks)
            y_predict = examination * relevance

            return y_predict.squeeze(), relevance.squeeze()
        else:
            return relevance.squeeze()

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch

        y_predict, _ = self.forward(x)
        loss = self.loss(y_predict, y_click, n)
        metrics = get_metrics(loss, prefix="train_")

        self.log_dict(metrics)
        return loss.sum(dim=1).mean()

    def validation_step(self, batch, idx):
        q, x, y, y_click, n = batch

        y_predict_click, y_predict = self.forward(x)
        loss = self.loss(y_predict_click, y_click, n)
        metrics = get_metrics(loss, y_predict, y, n, "val_")

        self.log_dict(metrics)
        return loss.sum(dim=1).mean()

    def test_step(self, batch, idx, dl_idx):
        if dl_idx == 0:
            q, x, y, y_click, n = batch
            y_predict_click, y_predict = self.forward(x)
            loss = self.loss(y_predict_click, y_click, n)
            metrics = get_metrics(loss, y_predict, y, n, "test_clicks_")
        else:
            query_ids, x, y, n = batch
            y_predict = self.forward(x, click_pred=False)
            metrics = get_metrics(
                y_predict=y_predict, y_true=y, n=n, prefix="test_rels_"
            )

        self.log_dict(metrics)
        return metrics

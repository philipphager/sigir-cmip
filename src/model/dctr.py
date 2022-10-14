from typing import Tuple

import torch
from torch import nn

from .base import ClickModel
from ..evaluation.metrics import get_click_metrics


class DCTR(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
    ):
        super().__init__(loss, optimizer, learning_rate)
        self.automatic_optimization = False
        self.clicks = nn.Parameter(torch.zeros(n_documents), requires_grad=False)
        self.impressions = nn.Parameter(torch.zeros(n_documents), requires_grad=False)

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch

        self.clicks[x] += y_click
        self.impressions[x] += 1

        y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = get_click_metrics(y_predict_click, y_click, n, "train_")
        metrics["train_loss"] = loss
        self.log_dict(metrics)

        return loss

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_predict = self.clicks[x] / self.impressions[x].clip(min=1)

        if click_pred:
            return y_predict, y_predict
        else:
            return y_predict

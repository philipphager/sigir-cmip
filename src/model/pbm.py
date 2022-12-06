from typing import Tuple

import torch
from torch import nn

from .base import ClickModel


class PBM(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, optimizer, learning_rate, lp_scores)

        self.relevance = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.examination = nn.Sequential(nn.Embedding(n_results, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
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

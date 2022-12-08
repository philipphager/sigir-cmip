from typing import Tuple

import torch
from torch import nn

from .base import NeuralClickModel


class UBM(NeuralClickModel):
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
        self.examination = nn.Sequential(nn.Embedding(n_results**2, 1), nn.Sigmoid())

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
            latest_clicked_ranks = torch.zeros(
                n_batch, n_items, dtype=torch.long, device=self.device
            )
            for r in range(n_items - 1):
                latest_clicked_ranks[:, r + 1] = torch.where(
                    true_clicks[:, r] == 1, ranks[:, r] + 1, latest_clicked_ranks[:, r]
                )

            examination = self.examination(latest_clicked_ranks + ranks * n_items)

            y_predict = examination * relevance
            return y_predict.squeeze(), relevance.squeeze()
        else:
            return relevance.squeeze()

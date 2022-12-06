from typing import Tuple

import torch
from torch import nn

from .base import ClickModel


class CACM_minus(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
        pos_embedd_dim: int,
        click_embedd_dim: int,
        inner_state_dim: int,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, optimizer, learning_rate, lp_scores)

        self.relevance = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.pos_embedd = nn.Embedding(n_results, pos_embedd_dim)
        self.click_embedd = nn.Embedding(2, click_embedd_dim)

        self.examination = nn.GRU(
            pos_embedd_dim + click_embedd_dim,
            hidden_size=inner_state_dim,
            num_layers=1,
            batch_first=True,
        )

        self.reduction = nn.Sequential(nn.Linear(inner_state_dim, 1), nn.Sigmoid())

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
            prev_clicks = torch.cat(
                [torch.zeros(n_batch, 1, device=self.device), true_clicks[:, :-1]],
                dim=1,
            )
            embedds = torch.cat(
                [self.pos_embedd(ranks), self.click_embedd(prev_clicks.long())], dim=2
            )
            examination = self.reduction(self.examination(embedds)[0])

            y_predict = examination * relevance
            return y_predict.squeeze(), relevance.squeeze()
        else:
            return relevance.squeeze()

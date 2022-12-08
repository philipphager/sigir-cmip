from typing import Tuple

import torch
from torch import nn

from .base import NeuralClickModel


class DBN(NeuralClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        estimate_gamma: bool,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, optimizer, learning_rate, lp_scores)

        self.attractiveness = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.satisfaction = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())

        if estimate_gamma:
            self.gamma = nn.Sequential(nn.Embedding(1, 1), nn.Sigmoid())
        else:
            self.gamma = nn.Embedding.from_pretrained(torch.ones((1, 1)))

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attractiveness = self.attractiveness(x).squeeze()
        satisfaction = self.satisfaction(x).squeeze()
        relevance = attractiveness * satisfaction

        if click_pred:
            gamma = self.gamma(torch.zeros_like(x)).squeeze()
            exam_rank = gamma * (1 - satisfaction * true_clicks)

            # Shift examination probabilities to subsequent rank and set examination
            # at first position to 1.
            exam_rank = torch.roll(exam_rank, 1)
            exam_rank[:, 0] = 1
            examination = torch.cumprod(exam_rank, dim=1)

            y_predict = examination * attractiveness
            return y_predict, relevance
        else:
            return relevance

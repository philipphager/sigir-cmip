from typing import Tuple

import torch
from torch import nn

from .base import ClickModel


class DBN(ClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
    ):
        super().__init__(loss, optimizer, learning_rate)

        self.attractiveness = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.satisfaction = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.gamma = nn.Sequential(nn.Embedding(1, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_items = x.shape

        attractiveness = self.attractiveness(x)
        satisfaction = self.satisfaction(x)
        if click_pred:
            exam_rank = self.gamma(torch.zeros_like(x)) * (
                1 - satisfaction * true_clicks.unsqueeze(2)
            )
            examination = torch.cumprod(exam_rank, dim=1)

            y_predict = examination * attractiveness
            return y_predict.squeeze(), (attractiveness * satisfaction).squeeze()
        else:
            return (attractiveness * satisfaction).squeeze()

from typing import List, Tuple

import torch
from torch import nn

from ..evaluation.base import Metric
from .base import NeuralClickModel


class PBM(NeuralClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
        random_state: int,
        metrics: List[Metric],
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(
            loss, optimizer, learning_rate, metrics, n_results, random_state, lp_scores
        )

        self.relevance = nn.Sequential(nn.Embedding(n_documents, 1), nn.Sigmoid())
        self.examination = nn.Sequential(nn.Embedding(n_results, 1), nn.Sigmoid())
        # self.apply(self._init_weights)

    def forward(
        self,
        q: torch.Tensor,
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

    def on_train_end(self):
        self.logger.log_table(
            key="Appendix/propensities",
            columns=[str(i) for i in range(1, self.n_results + 1)],
            data=self.examination(torch.arange(self.n_results))
            .transpose(0, 1)
            .tolist(),
        )

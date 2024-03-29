from typing import List, Tuple

import torch
from torch import nn

from ..evaluation.base import Metric
from .base import NeuralClickModel


class NCM(NeuralClickModel):
    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
        random_state: int,
        n_queries: int,
        query_embedd_dim: int,
        doc_embedd_dim: int,
        click_embedd_dim: int,
        inner_state_dim: int,
        metrics: List[Metric],
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(
            loss, optimizer, learning_rate, metrics, n_results, random_state, lp_scores
        )

        self.query_embedd = nn.Embedding(n_queries + 1, query_embedd_dim)
        self.doc_embedd = nn.Embedding(n_documents, doc_embedd_dim)
        self.click_embedd = nn.Embedding(2, click_embedd_dim)

        self.query_embedd_dim = query_embedd_dim
        self.doc_embedd_dim = doc_embedd_dim
        self.click_embedd_dim = click_embedd_dim

        self.click_prob = nn.GRU(
            query_embedd_dim + doc_embedd_dim + click_embedd_dim,
            hidden_size=inner_state_dim,
            num_layers=1,
            batch_first=True,
        )

        self.reduction = nn.Sequential(nn.Linear(inner_state_dim, 1), nn.Sigmoid())

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_items = x.shape

        if click_pred:
            prev_clicks = torch.cat(
                [torch.zeros(n_batch, 1, device=self.device), true_clicks[:, :-1]],
                dim=1,
            )
            embedd_seq = torch.cat(
                [
                    torch.cat(
                        [
                            self.query_embedd(q).unsqueeze(1),
                            torch.zeros(
                                n_batch,
                                1,
                                self.doc_embedd_dim + self.click_embedd_dim,
                                device=self.device,
                            ),
                        ],
                        dim=2,
                    ),
                    torch.cat(
                        [
                            torch.zeros(
                                n_batch,
                                n_items,
                                self.query_embedd_dim,
                                device=self.device,
                            ),
                            self.doc_embedd(x),
                            self.click_embedd(prev_clicks.long()),
                        ],
                        dim=2,
                    ),
                ],
                dim=1,
            )
            y_predict = self.reduction(self.click_prob(embedd_seq)[0][:, 1:])

            return y_predict.squeeze(), None
        else:
            x_flatten = x.reshape(n_batch * n_items, 1)
            rel_embedds = torch.cat(
                [
                    torch.cat(
                        [
                            self.query_embedd(q)
                            .unsqueeze(1)
                            .repeat_interleave(n_items, dim=0),
                            torch.zeros(
                                n_batch * n_items,
                                1,
                                self.doc_embedd_dim + self.click_embedd_dim,
                                device=self.device,
                            ),
                        ],
                        dim=2,
                    ),
                    torch.cat(
                        [
                            torch.zeros(
                                n_batch * n_items,
                                1,
                                self.query_embedd_dim,
                                device=self.device,
                            ),
                            self.doc_embedd(x_flatten),
                            torch.zeros(
                                n_batch * n_items,
                                1,
                                self.click_embedd_dim,
                                device=self.device,
                            ),
                        ],
                        dim=2,
                    ),
                ],
                dim=1,
            )
            relevance = self.reduction(self.click_prob(rel_embedds)[0][:, 1]).reshape(
                n_batch, n_items
            )
            return relevance

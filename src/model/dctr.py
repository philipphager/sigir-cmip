from typing import Tuple

import torch
from torch import nn

from .base import ClickModel
from ..evaluation.metrics import get_click_metrics


class DCTR(ClickModel):
    """
    dCTR model as in [Deffayet 2022] with Beta(alpha=1, beta=1) prior.
    Document relevance is CTR per document.
    """

    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        prior_clicks: int,
        prior_impressions: int,
    ):
        super().__init__(loss, optimizer, learning_rate)
        # Turn off optimization for count-based click model
        self.automatic_optimization = False

        self.clicks = nn.Parameter(
            torch.full((n_documents,), prior_clicks, dtype=torch.float),
            requires_grad=False,
        )
        self.impressions = nn.Parameter(
            torch.full((n_documents,), prior_impressions, dtype=torch.float),
            requires_grad=False,
        )

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch

        # Ignore padding
        impressions = (x > 0).float()
        # CTR per document
        self.clicks.index_add_(0, x.ravel(), y_click.ravel())
        self.impressions.index_add_(0, x.ravel(), impressions.ravel())

        y_predict_click, y_predict = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = get_click_metrics(y_predict_click, y_click, n, "train_")
        metrics["train_loss"] = loss
        self.log_dict(metrics)

        # Update global_step counter for checkpointing
        self.optimizers().step()

        return loss

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_predict = self.clicks[x] / self.impressions[x]

        if click_pred:
            return y_predict, y_predict
        else:
            return y_predict


class RankedDCTR(ClickModel):
    """
    drCTR model as in [Deffayet 2022] with Beta(alpha=1, beta=1) prior.
    Document relevance is CTR per document and rank times the inverse rank CTR.
    """

    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
        prior_clicks: int,
        prior_impressions: int,
    ):
        super().__init__(loss, optimizer, learning_rate)
        # Turn off optimization for count-based click model
        self.automatic_optimization = False
        self.n_documents = n_documents
        self.n_result = n_results

        self.document_clicks = nn.Parameter(
            torch.full((n_documents * n_results,), prior_clicks, dtype=torch.float),
            requires_grad=False,
        )
        self.document_impressions = nn.Parameter(
            torch.full(
                (n_documents * n_results,), prior_impressions, dtype=torch.float
            ),
            requires_grad=False,
        )
        self.rank_clicks = nn.Parameter(torch.zeros(n_results), requires_grad=False)
        self.rank_impressions = nn.Parameter(
            torch.zeros(n_results), requires_grad=False
        )

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch
        n_batch, n_items = x.shape

        # Ignore padding
        impressions = (x > 0).float()
        # CTR per rank
        self.rank_clicks += y_click.sum(dim=0)
        self.rank_impressions += impressions.sum(dim=0)
        # CTR per document and rank
        ranks = torch.arange(n_items, device=self.device).repeat(n_batch, 1)
        idx = (x * n_items + ranks).ravel()
        self.document_clicks.index_add_(0, idx, y_click.ravel())
        self.document_impressions.index_add_(0, idx, impressions.ravel())

        y_predict_click, _ = self.forward(x, true_clicks=y_click)
        loss = self.loss(y_predict_click, y_click, n)

        metrics = get_click_metrics(y_predict_click, y_click, n, "train_")
        metrics["train_loss"] = loss
        self.log_dict(metrics)

        # Update global_step counter for checkpointing
        self.optimizers().step()

        return loss

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_items = x.shape

        rank_ctr = self.rank_clicks / self.rank_impressions
        relevance = torch.zeros_like(x, device=self.device, dtype=torch.float)

        for i in range(n_items):
            idx = x[:, i]
            batch_idx = torch.arange(n_batch, device=self.device)
            
            clicks = self.document_clicks.view(self.n_documents, self.n_result)[idx]
            impressions = self.document_impressions.view(self.n_documents, self.n_result)[idx]
            relevance[batch_idx, i] = (rank_ctr * (clicks / impressions)).sum(1)

        if click_pred:
            ranks = torch.arange(n_items, device=self.device)
            idx = x * n_items + ranks
            y_predict = self.document_clicks[idx] / self.document_impressions[idx]

            return y_predict, relevance
        else:
            return relevance

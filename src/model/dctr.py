from typing import Tuple

import torch
from scipy.stats import beta
from torch import nn

from .base import ClickModel
from ..evaluation.metrics import get_click_metrics


def fit_beta(y: torch.tensor, eps: float = 1e-4):
    ctr = y.clip(min=eps, max=1 - eps)
    a, b, _, _ = beta.fit(ctr, method="MLE", floc=0, fscale=1)
    return a, b


class DCTR(ClickModel):
    """
    dCTR model as in [Deffayet 2022], computing relevance as CTR per document.
    We smooth CTRs using empirical bayes, estimating priors for a beta distribution
    on the train dataset as in [Chapelle and Zhang 2009].
    """

    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
    ):
        super().__init__(loss, optimizer, learning_rate)
        # Turn off optimization for count-based click model
        self.automatic_optimization = False
        self.n_documents = n_documents
        # Init priors to avoid division by zero in validation before training
        self.prior_clicks = 1
        self.prior_impressions = 1
        # Count clicks and impressions per document
        self.clicks = nn.Parameter(
            torch.zeros(n_documents, dtype=torch.float),
            requires_grad=False,
        )
        self.impressions = nn.Parameter(
            torch.zeros(n_documents, dtype=torch.float),
            requires_grad=False,
        )

    def on_train_start(self):
        # Access full train dataset
        train = self.trainer.train_dataloader.dataset.datasets

        # Sum clicks and impressions per document over all ranks
        clicks = train.get_document_rank_clicks(self.n_documents)
        self.clicks += clicks.sum(dim=1)

        impressions = train.get_document_rank_impressions(self.n_documents)
        self.impressions += impressions.sum(dim=1)

        # Compute CTRs for documents that got at least one impression
        dctr = self.clicks / self.impressions.clip(min=1)
        dctr = dctr[self.impressions > 0]

        # Fit beta prior on CTRs, A being clicks and B non-clicks
        a, b = fit_beta(dctr)
        self.prior_clicks = a
        self.prior_impressions = a + b

    def training_step(self, batch, idx):
        q, x, y, y_click, n = batch

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
        clicks = self.clicks[x] + self.prior_clicks
        impressions = self.impressions[x] + self.prior_impressions
        y_predict = clicks / impressions

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
            impressions = self.document_impressions.view(
                self.n_documents, self.n_result
            )[idx]
            relevance[batch_idx, i] = (rank_ctr * (clicks / impressions)).sum(1)

        if click_pred:
            ranks = torch.arange(n_items, device=self.device)
            idx = x * n_items + ranks
            y_predict = self.document_clicks[idx] / self.document_impressions[idx]

            return y_predict, relevance
        else:
            return relevance

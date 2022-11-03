import logging
from typing import Tuple

import torch
from scipy.stats import beta
from torch import nn

from .base import ClickModel

logger = logging.getLogger(__name__)


def fit_beta(y: torch.Tensor, eps: float = 1e-4):
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
        # Init priors to 1.0, avoid division by zero in validation before training
        self.prior_clicks = nn.Parameter(
            torch.tensor([1.0]),
            requires_grad=False,
        )
        self.prior_impressions = nn.Parameter(
            torch.tensor([1.0]),
            requires_grad=False,
        )
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
        ctr = self.clicks / self.impressions.clip(min=1)
        ctr = ctr[self.impressions > 0]

        # Fit beta prior on CTRs, A being clicks and B non-clicks
        a, b = fit_beta(ctr)
        self.prior_clicks = nn.Parameter(
            torch.tensor([a]),
            requires_grad=False,
        )
        self.prior_impressions = nn.Parameter(
            torch.tensor([a + b]),
            requires_grad=False,
        )
        logger.info(f"dCTR with Beta({a}, {b}), prior CTR per document: {a / (a + b)}")

    def training_step(self, batch, idx):
        # Update global_step counter for checkpointing
        self.optimizers().step()
        return super().training_step(batch, idx)

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
    drCTR model as in [Deffayet 2022], computing relevance as CTR per document and rank.
    We smooth CTRs using empirical bayes, estimating priors for a beta distribution
    on the train dataset as in [Chapelle and Zhang 2009].
    Document relevance is the CTR per document and rank times the inverse rank CTR.
    """

    def __init__(
        self,
        loss: nn.Module,
        optimizer: str,
        learning_rate: float,
        n_documents: int,
        n_results: int,
    ):
        super().__init__(loss, optimizer, learning_rate)
        # Turn off optimization for count-based click model
        self.automatic_optimization = False
        self.n_documents = n_documents
        self.n_result = n_results

        self.prior_clicks = nn.Parameter(
            torch.ones(n_results),
            requires_grad=False,
        )
        self.prior_impressions = nn.Parameter(
            torch.ones(n_results),
            requires_grad=False,
        )
        self.clicks = nn.Parameter(
            torch.zeros((n_documents, n_results), dtype=torch.float),
            requires_grad=False,
        )
        self.impressions = nn.Parameter(
            torch.zeros((n_documents, n_results), dtype=torch.float),
            requires_grad=False,
        )
        self.rank_clicks = nn.Parameter(
            torch.zeros(n_results),
            requires_grad=False,
        )
        self.rank_impressions = nn.Parameter(
            torch.zeros(n_results),
            requires_grad=False,
        )

    def on_train_start(self):
        # Access full train dataset
        train = self.trainer.train_dataloader.dataset.datasets

        clicks = train.get_document_rank_clicks(self.n_documents)
        impressions = train.get_document_rank_impressions(self.n_documents)

        self.clicks += clicks
        self.impressions += impressions
        self.rank_clicks += clicks.sum(dim=0)
        self.rank_impressions += impressions.sum(dim=0)

        prior_clicks = torch.zeros(self.n_result)
        prior_impressions = torch.zeros(self.n_result)

        # Fit beta prior on document CTRs per rank i
        for i in range(self.n_result):
            ctr = self.clicks[:, i] / self.impressions[:, i].clip(min=1)
            ctr = ctr[self.impressions[:, i] > 0]

            a, b = fit_beta(ctr)
            prior_clicks[i] = a
            prior_impressions[i] = a + b
            logger.info(
                f"ranked-dCTR with Beta({a}, {b}), prior CTR at rank {i}: {a / (a + b)}"
            )

        self.prior_clicks = nn.Parameter(
            prior_clicks,
            requires_grad=False,
        )
        self.prior_impressions = nn.Parameter(
            prior_impressions,
            requires_grad=False,
        )

    def training_step(self, batch, idx):
        # Update global_step counter for checkpointing
        self.optimizers().step()
        return super().training_step(batch, idx)

    def forward(
        self,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_results = x.shape

        ranks = torch.arange(n_results, device=self.device).repeat(n_batch, 1)

        # Compute ctr per document, per rank
        x = x.reshape(-1)
        clicks = self.clicks[x] + self.prior_clicks
        impressions = self.impressions[x] + self.prior_impressions
        document_ctr = clicks / impressions

        # Compute ctr per rank
        rank_ctr = self.rank_clicks / self.rank_impressions

        # Compute relevance as sum of rank-weighted document CTRs
        relevance = ((1 / rank_ctr) * document_ctr).sum(dim=1)
        relevance = relevance.reshape(n_batch, n_results)

        if click_pred:
            # Fetch CTR of document at the given rank
            ranks = ranks.reshape(-1, 1)
            y_predict = torch.gather(document_ctr, dim=1, index=ranks)
            y_predict = y_predict.reshape(n_batch, n_results)

            return y_predict, relevance
        else:
            return relevance

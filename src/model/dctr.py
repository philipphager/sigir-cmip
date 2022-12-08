import logging
from typing import List, Tuple

import torch
from scipy.stats import beta
from torch import nn

from ..data.dataset import ClickDatasetStats
from ..evaluation.base import Metric
from .base import StatsClickModel

logger = logging.getLogger(__name__)


def fit_beta(y: torch.Tensor, eps: float = 1e-4):
    ctr = y.clip(min=eps, max=1 - eps).detach().cpu().numpy()
    a, b, _, _ = beta.fit(ctr, method="MLE", floc=0, fscale=1)
    return a, b


class DCTR(StatsClickModel):
    """
    dCTR model as in [Deffayet 2022], computing relevance as CTR per document.
    We smooth CTRs using empirical bayes, estimating priors for a beta distribution
    on the train dataset as in [Chapelle and Zhang 2009].
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        train_stats: ClickDatasetStats,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, metrics, train_stats, lp_scores)

    def setup_parameters(self, train_stats: ClickDatasetStats):
        # Sum clicks and impressions per document over all ranks
        clicks = train_stats.document_rank_clicks.to(self.device)
        self.clicks = nn.Parameter(clicks.sum(dim=1))

        impressions = train_stats.document_rank_impressions.to(self.device)
        self.impressions = nn.Parameter(impressions.sum(dim=1))

        # Compute CTRs for documents that got at least one impression
        ctr = self.clicks / self.impressions.clip(min=1)
        ctr = ctr[self.impressions > 0]

        # Fit beta prior on CTRs, A being clicks and B non-clicks
        a, b = fit_beta(ctr)
        self.prior_clicks = a
        self.prior_impressions = a + b
        logger.info(f"dCTR with Beta({a}, {b}), prior CTR per document: {a / (a + b)}")

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


class RankedDCTR(StatsClickModel):
    """
    drCTR model as in [Deffayet 2022], computing relevance as CTR per document and rank.
    We smooth CTRs using empirical bayes, estimating priors for a beta distribution
    on the train dataset as in [Chapelle and Zhang 2009].
    Document relevance is the CTR per document and rank times the inverse rank CTR.
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        train_stats: ClickDatasetStats,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, metrics, train_stats, lp_scores)

    def setup_parameters(self, train_stats: ClickDatasetStats):
        clicks = train_stats.document_rank_clicks.to(self.device)
        impressions = train_stats.document_rank_impressions.to(self.device)

        self.clicks = nn.Parameter(clicks)
        self.impressions = nn.Parameter(impressions)
        self.rank_clicks = nn.Parameter(clicks.sum(dim=0))
        self.rank_impressions = nn.Parameter(impressions.sum(dim=0))

        n_results = len(self.rank_clicks)
        prior_clicks = torch.zeros(n_results).to(self.device)
        prior_impressions = torch.zeros(n_results).to(self.device)

        # Fit beta prior on document CTRs per rank i
        for i in range(n_results):
            ctr = self.clicks[:, i] / self.impressions[:, i].clip(min=1)
            ctr = ctr[self.impressions[:, i] > 0]

            a, b = fit_beta(ctr)
            prior_clicks[i] = a
            prior_impressions[i] = a + b
            logger.info(
                f"ranked-dCTR with Beta({a}, {b}), prior CTR at rank {i}: {a / (a + b)}"
            )

        self.prior_clicks = nn.Parameter(prior_clicks)
        self.prior_impressions = nn.Parameter(prior_impressions)

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

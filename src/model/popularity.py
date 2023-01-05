from typing import List, Tuple

import torch
from torch import nn

from ..data.dataset import ClickDatasetStats
from ..evaluation.base import Metric
from .base import StatsClickModel


class TopPop(StatsClickModel):
    """
    TopPop model as in [Deffayet 2022]
    Document relevance is the number of clicks per document, no CTR prediction.
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        train_stats: ClickDatasetStats,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, metrics, n_results, random_state, train_stats, lp_scores)

    def setup_parameters(self, train_stats: ClickDatasetStats):
        clicks = train_stats.document_rank_clicks
        self.clicks = nn.Parameter(clicks.sum(dim=1))

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relevance = self.clicks[x]

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full(x.shape, 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance


class TopPopObs(StatsClickModel):
    """
    TopPopObs model as in [Deffayet 2022]
    Document relevance is the number of clicks times the number of impressions per
    document, no CTR prediction.
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        train_stats: ClickDatasetStats,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, metrics, n_results, random_state, train_stats, lp_scores)

    def setup_parameters(self, train_stats: ClickDatasetStats):
        clicks = train_stats.document_rank_clicks.to(self.device)
        self.clicks = nn.Parameter(clicks.sum(dim=1))

        impressions = train_stats.document_rank_impressions.to(self.device)
        self.impressions = nn.Parameter(impressions.sum(dim=1))

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relevance = self.clicks[x] * self.impressions[x]

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full(x.shape, 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance


class RankedTopObs(StatsClickModel):
    """
    WeightedTopObs model as in [Deffayet 2022]
    Document relevance is the rank CTR times the number of impressions per document and
    rank, no CTR prediction.
    """

    def __init__(
        self,
        loss: nn.Module,
        metrics: List[Metric],
        n_results: int,
        random_state: int,
        train_stats: ClickDatasetStats,
        lp_scores: torch.FloatTensor = None,
        **kwargs,
    ):
        super().__init__(loss, metrics, n_results, random_state, train_stats, lp_scores)

    def setup_parameters(self, train_stats: ClickDatasetStats):
        clicks = train_stats.document_rank_clicks.to(self.device)
        impressions = train_stats.document_rank_impressions.to(self.device)

        rank_clicks = clicks.sum(dim=0)
        rank_impressions = impressions.sum(dim=0)
        rank_ctr = rank_clicks / rank_impressions.clip(min=1)

        self.impressions = nn.Parameter(impressions)
        self.rank_ctr = nn.Parameter(rank_ctr)

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        click_pred: bool = True,
        true_clicks: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_batch, n_results = x.shape

        # Weight impressions per document by rank ctr
        x = x.reshape(-1)
        relevance = (self.rank_ctr * self.impressions[x]).sum(dim=1)
        relevance = relevance.reshape(n_batch, n_results)

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full((n_batch, n_results), 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance

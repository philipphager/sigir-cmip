from typing import Tuple

import torch
from torch import nn

from .base import ClickModel


class TopPop(ClickModel):
    """
    TopPop model as in [Deffayet 2022]
    Document relevance is the number of clicks per document, no CTR prediction.
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
        self.clicks = nn.Parameter(
            torch.zeros(n_documents, dtype=torch.float),
            requires_grad=False,
        )

    def on_train_start(self):
        # Access full train dataset
        train = self.trainer.train_dataloader.dataset.datasets

        # Sum clicks and impressions per document over all ranks
        clicks = train.get_document_rank_clicks(self.n_documents)
        self.clicks += clicks.sum(dim=1).to(self.device)

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
        relevance = self.clicks[x]

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full(x.shape, 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance


class TopPopObs(ClickModel):
    """
    TopPopObs model as in [Deffayet 2022]
    Document relevance is the number of clicks times the number of impressions per
    document, no CTR prediction.
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
        self.clicks += clicks.sum(dim=1).to(self.device)

        impressions = train.get_document_rank_impressions(self.n_documents)
        self.impressions += impressions.sum(dim=1).to(self.device)

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
        relevance = self.clicks[x] * self.impressions[x]

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full(x.shape, 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance


class RankedTopObs(ClickModel):
    """
    WeightedTopObs model as in [Deffayet 2022]
    Document relevance is the rank CTR times the number of impressions per document and
    rank, no CTR prediction.
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
        clicks = clicks.to(self.device)
        impressions = train.get_document_rank_impressions(self.n_documents)
        impressions = impressions.to(self.device)

        self.clicks += clicks
        self.impressions += impressions
        self.rank_clicks += clicks.sum(dim=0)
        self.rank_impressions += impressions.sum(dim=0)

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

        # Compute ctr per rank
        rank_ctr = self.rank_clicks / self.rank_impressions

        # Weight impressions per document by rank ctr
        x = x.reshape(-1)
        relevance = (rank_ctr * self.impressions[x]).sum(dim=1)
        relevance = relevance.reshape(n_batch, n_results)

        if click_pred:
            # Dummy click prediction
            y_predict = torch.full(relevance.shape, 0.5, device=x.device)
            return y_predict, relevance
        else:
            return relevance

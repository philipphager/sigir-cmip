from typing import Tuple

import torch
from torch import nn

from .base import ClickModel
from ..evaluation.metrics import get_click_metrics


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
        self.clicks += clicks.sum(dim=1)

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
        y_predict = torch.full(x.shape, 0.5, device=x.device)
        relevance = self.clicks[x]

        if click_pred:
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
        self.clicks += clicks.sum(dim=1)

        impressions = train.get_document_rank_impressions(self.n_documents)
        self.impressions += impressions.sum(dim=1)

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
        y_predict = torch.full(x.shape, 0.5, device=x.device)
        relevance = self.clicks[x] * self.impressions[x]

        if click_pred:
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

        self.document_impressions = nn.Parameter(
            torch.zeros(n_documents * n_results, dtype=torch.float),
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
        # Impressions per document and rank
        ranks = torch.arange(n_items, device=self.device).repeat(n_batch, 1)
        idx = (x * n_items + ranks).ravel()
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

        ranks = torch.arange(n_items, device=self.device).repeat(n_batch, 1)
        idx = x * n_items + ranks

        rank_ctr = self.rank_clicks / self.rank_impressions
        relevance = rank_ctr * self.document_impressions[idx]
        y_predict = torch.full(x.shape, 0.5, device=x.device)

        if click_pred:
            return y_predict, relevance
        else:
            return relevance

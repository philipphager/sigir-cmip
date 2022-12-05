import logging

import torch
from torch.utils.data import Dataset

from src.data.dataset import RatingDataset
from src.data.simulation.query_dist.base import QueryDist
from src.data.simulation.user_model.base import UserModel
from src.util.tensor import scatter_rank_add

logger = logging.getLogger(__name__)


class ClickDataset(Dataset):
    def __init__(
        self,
        query_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        y_click: torch.Tensor,
        n: torch.Tensor,
    ):
        self.query_ids = query_ids
        self.x = x
        self.y = y
        self.y_click = y_click
        self.n = n

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i: int):
        return self.query_ids[i], self.x[i], self.y_click[i], self.n[i]

    def get_document_rank_clicks(self, n_documents) -> torch.Tensor:
        return scatter_rank_add(self.y_click, self.x, n_documents)

    def get_document_rank_impressions(self, n_documents) -> torch.Tensor:
        impressions = (self.x > 0).float()
        return scatter_rank_add(impressions, self.x, n_documents)


class Simulator:
    def __init__(
        self,
        user_model: UserModel,
        query_dist: QueryDist,
        n_sessions: int,
        rank_size: int,
    ):
        self.user_model = user_model
        self.n_sessions = n_sessions
        self.query_dist = query_dist
        self.rank_size = rank_size

    def __call__(
        self, dataset: RatingDataset, lp_scores: torch.FloatTensor, eps: float = 1e-9
    ):
        query_ids, x, y, n = dataset[:]

        # Sample queries
        logger.info(f"Sampling queries using {self.query_dist} distribution")
        sample_ids = self.query_dist(len(query_ids), self.n_sessions)
        query_ids = query_ids[sample_ids]
        x = x[sample_ids]
        y = y[sample_ids]
        n = n[sample_ids]

        # Sample top-k rankings using Gumbel noise trick
        logger.info("Sample top-k rankings")
        y_predict = lp_scores[sample_ids]
        noise = torch.rand_like(y_predict.float())
        y_predict = y_predict - torch.log(-torch.log(noise))
        idx = torch.argsort(-y_predict)[:, : self.rank_size]
        x_impressed = torch.gather(x, 1, idx)
        y_impressed = torch.gather(y, 1, idx)
        n = n.clamp(max=self.rank_size)

        # Sample clicks
        logger.info("Sample clicks")
        y_clicks = self.user_model(y_impressed)

        return ClickDataset(query_ids, x_impressed, y_impressed, y_clicks, n)

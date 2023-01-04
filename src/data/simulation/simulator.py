import logging

import torch

from src.data.dataset import ClickDataset, RatingDataset
from src.data.simulation.query_dist.base import QueryDist
from src.data.simulation.user_model.base import UserModel

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(
        self,
        user_model: UserModel,
        query_dist: QueryDist,
        n_sessions: int,
        rank_size: int,
        temperature: int,
        random_state: int,
    ):
        self.user_model = user_model
        self.n_sessions = n_sessions
        self.query_dist = query_dist
        self.rank_size = rank_size
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(random_state)

    def __call__(
        self,
        dataset: RatingDataset,
        lp_scores: torch.FloatTensor,
        eps: float = 1e-9,
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
        noise = torch.rand(y_predict.size(), generator=self.generator)
        y_predict = y_predict - self.temperature * torch.log(-torch.log(noise))
        idx = torch.argsort(-y_predict)[:, : self.rank_size]
        x_impressed = torch.gather(x, 1, idx)
        y_impressed = torch.gather(y, 1, idx)
        n = n.clamp(max=self.rank_size)

        # Sample clicks
        logger.info("Sample clicks")
        y_clicks = self.user_model(y_impressed)

        return ClickDataset(query_ids, x_impressed, y_impressed, y_clicks, n)

import torch
from torch.distributions.categorical import Categorical

from src.simulation.query_dist.base import QueryDist


class UniformQueryDist(QueryDist):
    def __init__(self):
        return

    def __call__(self, n_queries: int, n_sessions: int) -> torch.LongTensor:
        return Categorical(probs=torch.ones(n_queries)).sample(
            sample_shape=(n_sessions,)
        )

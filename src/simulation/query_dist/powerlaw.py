import torch
from torch.distributions.categorical import Categorical

from src.simulation.query_dist.base import QueryDist


class PowerLawQueryDist(QueryDist):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.gen = (
            torch.Generator()
        )  # Important in order to get the same distribution at training and test time

    def __call__(self, n_queries: int, n_sessions: int) -> torch.LongTensor:
        shuffle_q = torch.randperm(n_queries, generator=self.gen)
        probs = torch.arange(1, n_queries + 1).pow(-self.alpha)
        probs = probs[shuffle_q]
        return Categorical(probs=probs).sample(sample_shape=(n_sessions,))

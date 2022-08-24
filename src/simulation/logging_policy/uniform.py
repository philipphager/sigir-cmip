import torch

from src.data.preprocessing import RatingDataset
from src.simulation.logging_policy.base import LoggingPolicy


class UniformPolicy(LoggingPolicy):
    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]
        return torch.rand_like(y.float())

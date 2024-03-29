import torch

from src.data.dataset import RatingDataset
from src.data.simulation.logging_policy.base import LoggingPolicy
from src.model.loss import mask_padding


class NoisyOraclePolicy(LoggingPolicy):
    def __init__(self, noise: float, random_state: int):
        self.noise = noise
        self.generator = torch.Generator().manual_seed(random_state)

    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]

        y = y + self.noise * torch.randn(y.size(), generator=self.generator)
        y = mask_padding(y, n, -torch.inf)

        return y

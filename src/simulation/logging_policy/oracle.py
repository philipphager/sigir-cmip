import torch

from src.data.preprocessing import RatingDataset
from src.model.loss import mask_padding
from src.simulation.logging_policy.base import LoggingPolicy


class NoisyOraclePolicy(LoggingPolicy):
    def __init__(self, noise: float):
        self.noise = noise

    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]
        y = y + self.noise * torch.randn_like(y.float())
        y = mask_padding(y, n, -torch.inf)
        return y

import torch

from src.data.dataset import RatingDataset
from src.data.simulation.logging_policy.base import LoggingPolicy
from src.model.loss import mask_padding


class UniformPolicy(LoggingPolicy):
    def __init__(self, random_state: int):
        self.random_state = random_state

    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]

        generator = torch.Generator().manual_seed(self.random_state)
        y = torch.rand(y.size(), generator=generator)
        y = mask_padding(y, n, -torch.inf)

        return y

import torch

from src.data.dataset import RatingDataset
from src.data.simulation.logging_policy.base import LoggingPolicy
from src.model.loss import mask_padding


class UniformPolicy(LoggingPolicy):
    def __init__(self, random_state: int):
        self.generator = torch.Generator().manual_seed(random_state)

    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]

        y = torch.ones_like(y).float() / n.unsqueeze(1)
        y = mask_padding(y, n, -torch.inf)

        return y

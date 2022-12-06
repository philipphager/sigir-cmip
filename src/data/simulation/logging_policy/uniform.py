import torch

from src.data.dataset import RatingDataset
from src.data.simulation.logging_policy.base import LoggingPolicy
from src.model.loss import mask_padding


class UniformPolicy(LoggingPolicy):
    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]
        y = torch.rand_like(y.float())
        y = mask_padding(y, n, -torch.inf)
        return y

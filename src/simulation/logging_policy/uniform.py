import torch

from src.data.preprocessing import RatingDataset
from src.model.loss import mask_padding
from src.simulation.logging_policy.base import LoggingPolicy


class UniformPolicy(LoggingPolicy):
    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        query_ids, x, y, n = dataset[:]
        y = torch.rand_like(y.float())
        y = mask_padding(y, n, 0)
        return y

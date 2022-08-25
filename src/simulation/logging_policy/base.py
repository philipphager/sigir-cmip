from abc import ABC

import torch

from src.data.preprocessing import RatingDataset


class LoggingPolicy(ABC):
    def fit(self, dataset: RatingDataset):
        pass

    def predict(self, dataset: RatingDataset) -> torch.Tensor:
        pass

from abc import ABC, abstractmethod
from typing import Union

import torch

from src.data.dataset import FeatureRatingDataset, RatingDataset


class LoggingPolicy(ABC):
    @abstractmethod
    def fit(self, dataset: Union[RatingDataset, FeatureRatingDataset]):
        pass

    @abstractmethod
    def predict(
        self, dataset: Union[RatingDataset, FeatureRatingDataset]
    ) -> torch.FloatTensor:
        pass

    def requires_features(self) -> bool:
        """
        Indicates if policy requires document feature vectors.
        Set to true if fit() and predict() expect a FeatureRatingDataset
        """
        return False

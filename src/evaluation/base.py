from abc import ABC, abstractmethod
from typing import Dict

import torch


class Metric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class ClickMetric(Metric):
    @abstractmethod
    def __call__(
        self,
        y_predict_click: torch.FloatTensor,
        y_click: torch.LongTensor,
        n: torch.LongTensor,
    ) -> Dict[str, float]:
        pass


class RelevanceMetric(Metric):
    @abstractmethod
    def __call__(
        self,
        y_predict: torch.FloatTensor,
        y_true: torch.LongTensor,
        n: torch.LongTensor,
    ) -> Dict[str, float]:
        pass


class PolicyMetric(Metric):
    @abstractmethod
    def __call__(
        self,
        y_predict: torch.FloatTensor,
        y_logging_policy: torch.FloatTensor,
        y_true: torch.LongTensor,
        n: torch.LongTensor,
    ) -> Dict[str, float]:
        pass

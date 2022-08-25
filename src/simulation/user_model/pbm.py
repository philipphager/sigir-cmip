import torch

from src.simulation.user_model.base import UserModel
from src.simulation.user_model.util import (
    get_binary_relevance,
    get_graded_relevance,
    get_position_bias,
)


class BinaryPBM(UserModel):
    def __init__(self, click_noise: float = 0.1, position_bias: float = 1.0):
        self.click_noise = click_noise
        self.position_bias = position_bias

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_binary_relevance(y, self.click_noise)
        examination = get_position_bias(n_results, self.position_bias)
        return relevance * examination


class GradedPBM(UserModel):
    def __init__(self, click_noise: float = 0.1, position_bias: float = 1.0):
        self.click_noise = click_noise
        self.position_bias = position_bias

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, self.click_noise)
        examination = get_position_bias(n_results, self.position_bias)
        return relevance * examination

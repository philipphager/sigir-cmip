import torch

from src.data.simulation.user_model.base import UserModel
from src.data.simulation.user_model.util import (
    get_binary_relevance,
    get_graded_relevance,
    get_position_bias,
)


class BinaryPBM(UserModel):
    def __init__(
        self,
        click_noise: float,
        position_bias: float,
        random_state: int,
        random_state_increment: int,
    ):
        self.click_noise = click_noise
        self.position_bias = position_bias
        self.generator = torch.Generator().manual_seed(
            random_state + random_state_increment
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_binary_relevance(y, self.click_noise)
        examination = get_position_bias(n_results, self.position_bias)
        return torch.bernoulli(relevance * examination, generator=self.generator)


class GradedPBM(UserModel):
    def __init__(
        self,
        click_noise: float,
        position_bias: float,
        random_state: int,
        random_state_increment: int,
    ):
        self.click_noise = click_noise
        self.position_bias = position_bias
        self.generator = torch.Generator().manual_seed(
            random_state + random_state_increment
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, self.click_noise)
        examination = get_position_bias(n_results, self.position_bias)
        return torch.bernoulli(relevance * examination, generator=self.generator)

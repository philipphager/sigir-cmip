import torch

from src.data.simulation.user_model.base import UserModel
from src.data.simulation.user_model.util import get_graded_relevance, get_position_bias


class GradedCarousel(UserModel):
    def __init__(
        self,
        position_bias: float,
        attractiveness_noise: float,
        click_noise: float,
        carousel_length: float,
        gamma: float,
        random_state: int,
        random_state_increment: int,
    ):
        self.attractiveness_noise = attractiveness_noise
        self.click_noise = click_noise
        self.position_bias = position_bias
        self.carousel_length = carousel_length
        self.gamma = gamma
        self.generator = torch.Generator().manual_seed(
            random_state + random_state_increment
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, noise=0).reshape(
            n_queries, n_results // self.carousel_length, self.carousel_length
        )
        attractiveness = relevance + self.attractiveness_noise * torch.randn(
            relevance.size(),
            generator=self.generator,
        )
        attractiveness = attractiveness.clip(self.click_noise, 1)
        satisfaction = attractiveness
        carousel_examination = (
            get_position_bias(n_results // self.carousel_length, self.position_bias)
            .unsqueeze(1)
            .expand(n_queries, -1, self.carousel_length)
        )

        y_click = torch.zeros_like(y).reshape(
            n_queries, n_results // self.carousel_length, self.carousel_length
        )
        examination = torch.ones_like(y_click)

        for j in range(self.carousel_length):
            if j > 0:
                examination[:, :, j] = (
                    examination[:, :, j - 1]
                    * self.gamma
                    * (1 - y_click[:, :, j - 1] * satisfaction[:, :, j - 1])
                )
            y_click[:, :, j] = torch.bernoulli(
                examination[:, :, j] * attractiveness[:, :, j],
                generator=self.generator,
            )

        y_click = y_click * carousel_examination

        return y_click.reshape(n_queries, n_results)

    def get_optimal_order(self, n_results) -> torch.LongTensor:
        prior_exposure = (
            get_position_bias(n_results // self.carousel_length, self.position_bias)
            .unsqueeze(1)
            .repeat(1, self.carousel_length)
        )
        prior_exposure *= torch.tensor(self.gamma).pow(
            torch.arange(self.carousel_length)
        )
        return torch.argsort(prior_exposure.flatten(), descending=True)

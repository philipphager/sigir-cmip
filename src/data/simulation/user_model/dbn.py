import torch

from src.data.simulation.user_model.base import UserModel
from src.data.simulation.user_model.util import (
    get_graded_relevance,
)


class GradedDBN(UserModel):
    def __init__(
        self,
        attractiveness_noise: float = 0.1,
        click_noise: float = 0.1,
        gamma: float = 0.9,
    ):
        self.attractiveness_noise = attractiveness_noise
        self.click_noise = click_noise
        self.gamma = gamma

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, noise=0)

        # Use noisy relevance as attractiveness, add min attractiveness
        noise = self.attractiveness_noise * torch.randn_like(relevance)
        attractiveness = relevance + noise
        attractiveness = attractiveness.clip(self.click_noise, 1)

        satisfaction = relevance
        examination = torch.ones_like(y).float()
        y_click = torch.zeros_like(y)

        for i in range(n_results):
            if i > 0:
                examination[:, i] = (
                    examination[:, i - 1]
                    * self.gamma
                    * (1 - y_click[:, i - 1] * satisfaction[:, i - 1])
                )

            y_click[:, i] = torch.bernoulli(examination[:, i] * attractiveness[:, i])

        return y_click

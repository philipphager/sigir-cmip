import torch

from src.data.simulation.user_model.base import UserModel
from src.data.simulation.user_model.util import get_graded_relevance


class GradedDBN(UserModel):
    def __init__(
        self,
        attractiveness_noise: float,
        click_noise: float,
        gamma: float,
        random_state: int,
        random_state_increment: int,
    ):
        self.attractiveness_noise = attractiveness_noise
        self.click_noise = click_noise
        self.gamma = gamma
        self.generator = torch.Generator().manual_seed(
            random_state + random_state_increment
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, noise=0)

        # Use noisy relevance as attractiveness, add min attractiveness
        noise = self.attractiveness_noise * torch.randn(
            relevance.size(), generator=self.generator
        )
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

            y_click[:, i] = torch.bernoulli(
                examination[:, i] * attractiveness[:, i],
                generator=self.generator,
            )

        return y_click


class MixtureDBN(UserModel):
    def __init__(
        self,
        attractiveness_noise: float,
        click_noise: float,
        gamma: float,
        mixture_param: float,
        random_state: int,
        random_state_increment: int,
    ):
        self.attractiveness_noise = attractiveness_noise
        self.click_noise = click_noise
        self.gamma = gamma
        self.mixture_param = mixture_param
        self.generator = torch.Generator().manual_seed(
            random_state + random_state_increment
        )

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, noise=0)

        # Use noisy relevance as attractiveness, add min attractiveness
        noise = self.attractiveness_noise * torch.randn(
            relevance.size(),
            generator=self.generator,
        )
        attractiveness = relevance + noise
        attractiveness = attractiveness.clip(self.click_noise, 1)

        satisfaction = relevance
        examination = torch.ones_like(y)  # .float()
        y_click = torch.zeros_like(y)

        mixture = torch.bernoulli(
            self.mixture_param * torch.ones(n_queries), generator=self.generator
        )
        td_idx = torch.nonzero(mixture).squeeze()
        bu_idx = torch.nonzero(1 - mixture).squeeze()
        # Top-down
        for i in range(n_results):
            if i > 0:
                examination[td_idx, i] = (
                    examination[td_idx, i - 1]
                    * self.gamma
                    * (1 - y_click[td_idx, i - 1] * satisfaction[td_idx, i - 1])
                )

            y_click[td_idx, i] = torch.bernoulli(
                examination[td_idx, i] * attractiveness[td_idx, i],
                generator=self.generator,
            )

        # Bottom-up
        for i in range(n_results, 0, -1):
            if i < n_results:
                examination[bu_idx, i - 1] = (
                    examination[bu_idx, i]
                    * self.gamma
                    * (1 - y_click[bu_idx, i] * satisfaction[bu_idx, i])
                )

            y_click[bu_idx, i - 1] = torch.bernoulli(
                examination[bu_idx, i - 1] * attractiveness[bu_idx, i - 1],
                generator=self.generator,
            )

        return y_click

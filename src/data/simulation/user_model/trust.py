import torch

from src.data.simulation.user_model.base import UserModel
from src.data.simulation.user_model.util import (
    get_binary_relevance,
    get_false_positive_ctr,
    get_graded_relevance,
    get_position_bias,
    get_true_positive_ctr,
)


class BinaryTrustBias(UserModel):
    """
    User model with position bias, trust bias, and binary relevance as in:
    - [Affine Corrections - Vardasbi et al., 2020]
    - [Mixture-based Corrections - Vardasbi et al., 2021]

    For document d at position k, clicks are defined by:
    P(C = 1 | d, k) = P(E = 1 | k) * P(R = 1 | d) * P(C = 1 | E = 1, R = 1, k)
                    + P(E = 1 | k) * P(R = 0 | d) * P(C = 1 | E = 1, R = 0, k)
    """

    def __init__(self, position_bias: float, random_state: int):
        self.position_bias = position_bias
        self.generator = torch.Generator().manual_seed(random_state)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_binary_relevance(y, noise=0)
        examination = get_position_bias(n_results, self.position_bias)
        tp_clicks = get_true_positive_ctr(n_results)
        fp_clicks = get_false_positive_ctr(n_results)
        click_probabilities = examination * (
            relevance * tp_clicks + (1 - relevance) * fp_clicks
        )

        return torch.bernoulli(click_probabilities, generator=self.generator)

    def get_optimal_order(self, n_results) -> torch.LongTensor:
        return torch.argsort(
            get_position_bias(n_results, self.position_bias), descending=True
        )


class GradedTrustBias(UserModel):
    """
    User model with position bias, trust bias, and graded relevance as in:
    - [Affine Corrections - Vardasbi et al., 2020]
    - [Mixture-based Corrections - Vardasbi et al., 2021]

    For document d at position k, clicks are defined by:
    P(C = 1 | d, k) = P(E = 1 | k) * P(R = 1 | d) * P(C = 1 | E = 1, R = 1, k)
                    + P(E = 1 | k) * P(R = 0 | d) * P(C = 1 | E = 1, R = 0, k)
    """

    def __init__(self, position_bias: float, random_state: int):
        self.position_bias = position_bias
        self.generator = torch.Generator().manual_seed(random_state)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        n_queries, n_results = y.shape

        relevance = get_graded_relevance(y, noise=0)
        examination = get_position_bias(n_results, self.position_bias)
        tp_clicks = get_true_positive_ctr(n_results)
        fp_clicks = get_false_positive_ctr(n_results)

        click_probabilities = examination * (
            relevance * tp_clicks + (1 - relevance) * fp_clicks
        )

        return torch.bernoulli(click_probabilities, generator=self.generator)

    def get_optimal_order(self, n_results) -> torch.LongTensor:
        return torch.argsort(
            get_position_bias(n_results, self.position_bias), descending=True
        )

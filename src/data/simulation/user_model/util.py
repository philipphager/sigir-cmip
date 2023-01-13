import torch


def get_position_bias(n_results: int, strength: float) -> torch.Tensor:
    return (1 / (torch.arange(n_results) + 1)) ** strength


def get_binary_relevance(y: torch.Tensor, noise: float) -> torch.Tensor:
    return (y >= 3).float().clip(min=noise)


def get_graded_relevance(y: torch.Tensor, noise: float) -> torch.Tensor:
    return (y / 4).clip(min=noise)


def get_true_positive_ctr(n_results: int):
    """
    Probability to click on a relevant and observed item as in Equation 10 in:
    [Mixture-based Corrections - Vardasbi et al., 2021]
    """
    k = torch.arange(n_results) + 1
    return 1 - (torch.minimum(k, torch.tensor(20)) + 1) / 100


def get_false_positive_ctr(n_results: int):
    """
    Probability to click on a non-relevant and observed item as in Equation 11 in:
    [Mixture-based Corrections - Vardasbi et al., 2021]
    """
    k = torch.arange(n_results) + 1
    return 0.65 / torch.minimum(k, torch.tensor(10))

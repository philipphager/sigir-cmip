import torch


def get_position_bias(n_results: int, strength: float) -> torch.Tensor:
    return 1 / (1 + torch.arange(n_results)) ** strength


def get_binary_relevance(y: torch.Tensor, noise: float) -> torch.Tensor:
    return (y >= 3).float().clip(min=noise)


def get_graded_relevance(y: torch.Tensor, noise: float) -> torch.Tensor:
    return noise + (1 - noise) * (2**y - 1) / (2**4 - 1)

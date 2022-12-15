from typing import Dict, List

import torch


def join_metrics(metrics: List[Dict[str, float]], stage: str = "") -> Dict[str, float]:
    """
    Merges a list of dictionaries (containing metrics values) into a single dictionary
    and adds the model stage (val, test) as a prefix.
    """
    output = {}

    for metric in metrics:
        for k, v in metric.items():
            output[f"Metrics/{stage}/{k}"] = v

    return output


def add_label(x: torch.Tensor, label: int):
    """
    Adds a new column to a tensor containing the given label
    """
    labels = torch.full((len(x), 1), label)
    return torch.hstack([x, labels])


def random_split(x: torch.Tensor, splits: int):
    """
    Shuffle and split a tensor into equal parts
    """
    idx = torch.randperm(len(x))
    return torch.chunk(x[idx], splits)


def padding_mask(n: torch.Tensor, n_batch: int, n_results: int) -> torch.Tensor:
    mask = torch.arange(n_results).repeat(n_batch, 1).type_as(n)
    return mask < n.unsqueeze(-1)


def hstack(
    y_predict: torch.Tensor,
    y_logging_policy: torch.Tensor,
    y_true: torch.LongTensor,
) -> torch.Tensor:
    """
    Flatten all tensors to 1d and stack them into columns of a matrix of size:
    (n_results * n_queries) x 3
    Each row contains thus one observation of:
    (y_predict, y_logging_policy, y_true) or (x, y, z)
    """
    return torch.column_stack(
        [
            y_predict.ravel(),
            y_logging_policy.ravel(),
            y_true.ravel(),
        ]
    )

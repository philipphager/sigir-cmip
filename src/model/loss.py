import torch
from torch import nn


class BinaryCrossEntropy(nn.Module):
    def forward(
        self,
        y_predict: torch.Tensor,
        y_true: torch.Tensor,
        n: torch.Tensor,
        eps: float = 1e-10,
    ) -> torch.float:
        loss = -(
            (y_true) * torch.log(y_predict.clip(min=eps))
            + (1 - y_true) * torch.log((1 - y_predict).clip(min=eps))
        )

        loss = mask_padding(loss, n)
        return loss.sum(dim=1).mean()


def mask_padding(x: torch.Tensor, n: torch.Tensor, fill: float = 0.0):
    n_batch, n_results = x.shape
    n = n.unsqueeze(-1)
    mask = torch.arange(n_results).repeat(n_batch, 1).type_as(x)
    x = x.float()
    x[mask >= n] = fill

    return x

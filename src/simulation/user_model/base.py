from abc import ABC

import torch


class UserModel(ABC):
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        pass

from abc import ABC

import torch


class QueryDist(ABC):
    def __call__(self, n_queries: int, n_sessions: int) -> torch.LongTensor:
        pass

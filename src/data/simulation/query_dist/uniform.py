import torch


class UniformQueryDist:
    def __init__(self, random_state: int):
        self.generator = torch.Generator().manual_seed(random_state)

    def __call__(self, n_queries: int, n_sessions: int) -> torch.LongTensor:
        return torch.randint(
            n_queries,
            (n_sessions,),
            generator=self.generator,
        ).long()

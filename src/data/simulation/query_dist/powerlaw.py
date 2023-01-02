import torch


class PowerLawQueryDist:
    def __init__(self, alpha: float, random_state: int):
        self.alpha = alpha
        self.random_state = random_state
        self.generator = torch.Generator().manual_seed(random_state)

    def __call__(self, n_queries: int, n_sessions: int) -> torch.LongTensor:
        # Create new generator to fix query distribution across consecutive calls
        query_generator = torch.Generator().manual_seed(self.random_state)

        shuffle_q = torch.randperm(n_queries, generator=query_generator)
        probs = torch.arange(1, n_queries + 1).pow(-self.alpha)
        probs = probs[shuffle_q]

        return torch.multinomial(
            probs,
            n_sessions,
            replacement=True,
            generator=self.generator,
        ).long()

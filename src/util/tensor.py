import torch


def scatter_rank_sum(y: torch.Tensor, x: torch.Tensor, n_documents: int):
    """
    Sum values in y based on the value and rank of indices in x.
    Let x be document ids:
    [
        [0, 1, 2],
        [1, 0, 2]
    ]

    And y be:
    [
        [1, 0, 1],
        [0, 1, 1],
    ]

    The the result is of shape: n_documents x n_results:
    [
        [1, 1, 0],
        [0, 0, 0],
        [0, 0, 2],
    ]
    """
    n_batch, n_results = y.shape

    ranks = torch.arange(n_results).repeat((n_batch, 1))
    idx = (x * n_results + ranks).ravel()
    src = y.ravel()

    out = torch.zeros(n_documents * n_results).type_as(src)
    out.scatter_add_(-1, idx, src)

    # Return sum of y per document, per rank.
    return out.reshape((n_documents, n_results))

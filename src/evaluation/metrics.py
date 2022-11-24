from typing import List

import torch

from src.model.loss import mask_padding


def ndcg(
    y_predict: torch.FloatTensor,
    y_true: torch.LongTensor,
    n: torch.LongTensor,
    ranks: torch.LongTensor,
) -> torch.FloatTensor:
    """
    Exponential nDCG, as in [RankNet - Burges et al., 2005]

    - y_predict: torch.FloatTensor[batch_size, max_docs] -> relevance predictions
    - y_true: torch.LongTensor[batch_size, max_docs] -> relevance ground truth
    - n: torch.LongTensor[batch_size] -> number of documents in each query
    - ranks: torch.LongTensor -> cutoff ranks for nDCG (0 for no cutoff)
    return: torch.FloatTensor[max_docs] -> ndcg@k for each cutoff rank k
    """
    # Mask padding in queries with less than max docs
    y_predict = mask_padding(y_predict, n, fill=-float("inf"))
    y_true = mask_padding(y_true, n, fill=0)

    sorted_pred = y_true[
        torch.arange(len(y_true)).unsqueeze(1),
        torch.argsort(y_predict, dim=1, descending=True),
    ]
    sorted_rels = torch.sort(y_true, dim=1, descending=True)[0]

    propensities = torch.log2(
        torch.arange(
            2, y_predict.shape[1] + 2, dtype=torch.float, device=sorted_rels.device
        )
    )
    dcg = (2**sorted_pred - 1) / propensities
    dcg = mask_padding(dcg, n, fill=0)
    idcg = (2**sorted_rels - 1) / propensities

    dcg = torch.cumsum(dcg, dim=1)
    idcg = torch.maximum(torch.cumsum(idcg, dim=1), torch.ones_like(idcg))
    rank_ndcg = (dcg / idcg).mean(dim=0)

    return rank_ndcg[ranks - 1]


def perplexity(
    y_predict_click: torch.FloatTensor,
    y_click: torch.LongTensor,
    n: torch.LongTensor,
    ranks: torch.LongTensor,
    eps: float = 1e-10,
) -> torch.FloatTensor:
    """
    Perplexity, as in [Dupret and Piwowarski, 2008]

    - y_predict_click: torch.FloatTensor[batch_size, max_docs] -> click predictions
    - y_click: torch.LongTensor[batch_size, max_docs] -> click ground truth
    - n: torch.LongTensor[batch_size] -> number of documents in each query
    - ranks: torch.LongTensor[batch_size] -> cutoff ranks for ppl (0 for no cutoff)
    - eps: float -> Clipping value to avoid -inf in log computation
    return: torch.FloatTensor[max_docs] -> perplexity@k for each cutoff rank k
    """
    # Count documents per rank
    n_batch, n_results = y_click.shape
    mask = torch.arange(n_results, device=n.device).repeat((n_batch, 1))
    mask = mask < n.unsqueeze(1)
    docs_per_rank = mask.sum(dim=0)

    # Binary cross-entropy with log2
    loss = -(
        y_click * torch.log2(y_predict_click.clip(min=eps))
        + (1 - y_click) * torch.log2((1 - y_predict_click).clip(min=eps))
    )

    # Mask padding and average perplexity per rank
    loss = mask_padding(loss, n, fill=0)
    rank_ppl = 2 ** (loss.sum(dim=0) / docs_per_rank)
    mean_ppl = rank_ppl.mean()
    ppl = torch.cat([mean_ppl.unsqueeze(0), rank_ppl])

    return ppl[ranks]


def agreement_ratio(
    y_predict: torch.FloatTensor,
    y_logging_policy: torch.FloatTensor,
    y_true: torch.LongTensor,
    n: torch.LongTensor,
    disjoint_pairs: bool,
) -> float:
    pairs = _get_disjoint_pairs(n) if disjoint_pairs else _get_all_pairs(n)

    # Drop pairs of equal relevance
    pairs = [
        pairs[i][:, y_true[i, pairs_q[0]] != y_true[i, pairs_q[1]]]
        for i, pairs_q in enumerate(pairs)
    ]

    true_pref = _get_preference(y_true, pairs)
    lp_pref = _get_preference(y_logging_policy, pairs)
    cm_pref = _get_preference(y_predict, pairs)

    lp_wrong = torch.logical_xor(true_pref, lp_pref)
    cm_wrong = torch.logical_xor(true_pref, cm_pref)

    agreement_ratio = (
        torch.logical_and(lp_wrong, cm_wrong).sum()
        / torch.logical_or(lp_wrong, cm_wrong).sum()
    )

    return agreement_ratio, torch.cat(pairs, dim=1).shape[1]


def _get_disjoint_pairs(n: torch.LongTensor) -> List[torch.Tensor]:
    """
    Get indices for all disjoint pairs of documents per query. I.e. each document is
    only part of one pair per query.
    :param n: Tensor containing number of documents per query
    :return: List[LongTensor(2, n_docs // 2)](n_queries)
    """
    randperm = [torch.randperm(k) for k in n]

    return [
        torch.stack([randperm[i][: k // 2], randperm[i][k // 2 : 2 * k // 2]], dim=0)
        for i, k in enumerate(n.tolist())
    ]


def _get_all_pairs(n: torch.LongTensor) -> List[torch.LongTensor]:
    """
    Get indices for all pairs of documents per query.
    :param n: Tensor containing number of documents per query
    :return: List[LongTensor(2, n_docs * (n_docs-1) / 2)](n_queries)
    """
    return [torch.tril_indices(k, k, -1) for k in n]


def _get_preference(y: torch.Tensor, pairs: List[torch.LongTensor]) -> torch.BoolTensor:
    """
    Returns if the first document in a pair of documents is preferred over the second
    item.
    :param y: Tensor of scores per document per query as predicted by policy
    :param pairs: List of document pairs per query
    :return: Boolean tensor of single dimension containing all pair preferences
    """

    return torch.cat(
        [
            (y[i, query_pairs[0]] > y[i, query_pairs[1]])
            for i, query_pairs in enumerate(pairs)
        ],
        dim=0,
    )


def get_click_metrics(
    y_predict_click: torch.FloatTensor,
    y_click: torch.FloatTensor,
    n: torch.LongTensor,
    prefix: str,
):
    ranks = torch.tensor([1, 5, 10, 0], device=y_click.device)
    batch_ppl = perplexity(y_predict_click, y_click, n, ranks).detach()

    return {
        f"{prefix}ppl{k}": batch_ppl[i]
        for i, k in enumerate(["@1", "@5", "@10", "_avg"])
    }


def get_relevance_metrics(
    y_predict: torch.FloatTensor,
    y_true: torch.LongTensor,
    y_lp: torch.FloatTensor,
    n: torch.LongTensor,
    prefix: str,
):
    ranks = torch.tensor([1, 5, 10, 0], device=y_true.device)
    batch_ndcg = ndcg(y_predict, y_true, n, ranks).detach()
    relevance_metrics = {
        f"{prefix}ndcg{k}": batch_ndcg[i] for i, k in enumerate(["@1", "@5", "@10", ""])
    }
    if y_lp is not None:
        batch_agreement_ratio, batch_n_pairs = agreement_ratio(
            y_predict, y_lp, y_true, n, disjoint_pairs=False
        )
        relevance_metrics |= {
            f"{prefix}agreement_ratio": batch_agreement_ratio,
            "n_pairs": batch_n_pairs,
        }

    return relevance_metrics

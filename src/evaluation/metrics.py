import torch


def ndcg(
    y_predict: torch.FloatTensor,
    y_true: torch.LongTensor,
    n: torch.LongTensor,
    k: int = None,
) -> torch.FloatTensor:
    """
    nDCG with gain

    - y_predict : torch.FloatTensor[batch_size, max_docs] -> relevance predictions
    - y_true : torch.LongTensor[batch_size, max_docs] -> relevance ground truth
    - n : torch.LongTensor[batch_size] -> number of documents in each query
    - k : int -> cutoff rank for nDCG
    """
    # For when we have less than max_docs
    mask = torch.arange(y_predict.shape[1], device=y_predict.device).repeat(len(n), 1)
    y_predict[mask > n.unsqueeze(1)] = torch.tensor(-float("inf"))
    y_true[mask > n.unsqueeze(1)] = 0

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
    dcg[mask > n.unsqueeze(1)] = 0
    idcg = (2**sorted_rels - 1) / propensities

    if k is None:
        return torch.sum(dcg, dim=1) / torch.maximum(
            torch.sum(idcg, dim=1), torch.ones_like(n)
        )  # we put 0 nDCG when iDCG = 0
    else:
        return torch.sum(dcg[:, :k], dim=1) / torch.maximum(
            torch.sum(idcg[:, :k], dim=1), torch.ones_like(n)
        )  # we put 0 nDCG when iDCG = 0


def perplexity(
    y_predict_click: torch.FloatTensor, y_true_click: torch.LongTensor, k: int = None
) -> torch.FloatTensor:
    """
    Perplexity, as in [Dupret and Piwowarski, 2008]

    - y_predict : torch.FloatTensor[batch_size, rank_size] -> click predictions
    - y_true : torch.LongTensor[batch_size, rank_size] -> click ground truth
    - k : int -> rank for PPL@k, average perplexity if k == None.
    """
    pass


def get_metrics(
    y_predict: torch.Tensor,
    y_true: torch.Tensor,
    n: torch.Tensor,
    prefix: str = "",
    y_predict_clicks: torch.Tensor = None,
    y_true_clicks: torch.Tensor = None,
):
    return {
        f"{prefix}ndcg@1": ndcg(y_predict, y_true, n, k=1).mean().detach(),
        f"{prefix}ndcg@5": ndcg(y_predict, y_true, n, k=5).mean().detach(),
        f"{prefix}ndcg@10": ndcg(y_predict, y_true, n, k=10).mean().detach(),
        f"{prefix}ndcg": ndcg(y_predict, y_true, n).mean().detach(),
    }

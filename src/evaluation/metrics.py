import torch


def ndcg(
    y_predict: torch.FloatTensor,
    y_true: torch.LongTensor,
    n: torch.LongTensor,
    ranks: torch.LongTensor,
) -> torch.FloatTensor:
    """
    nDCG with gain

    - y_predict : torch.FloatTensor[batch_size, max_docs] -> relevance predictions
    - y_true : torch.LongTensor[batch_size, max_docs] -> relevance ground truth
    - n : torch.LongTensor[batch_size] -> number of documents in each query
    - ranks : torch.LongTensor -> cutoff ranks for nDCG (0 for no cutoff)
    """
    # For when we have less than max_docs
    mask = torch.arange(y_predict.shape[1], device=y_predict.device).repeat(len(n), 1)
    y_predict[mask >= n.unsqueeze(1)] = torch.tensor(-float("inf"))
    y_true[mask >= n.unsqueeze(1)] = 0

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
    dcg[mask >= n.unsqueeze(1)] = 0
    idcg = (2**sorted_rels - 1) / propensities

    return (
        torch.cumsum(dcg, dim=1)
        / torch.maximum(torch.cumsum(idcg, dim=1), torch.ones_like(idcg))
    )[:, ranks - 1]


def perplexity(loss: torch.FloatTensor, ranks: torch.LongTensor) -> torch.FloatTensor:
    """
    Perplexity, as in [Dupret and Piwowarski, 2008]

    - loss : Non-aggregated cross-entropy loss
    - ranks : torch.LongTensor -> ranks for PPL (0 for average PPL)
    """
    log2loss = loss / torch.log(torch.full_like(loss, 2))
    ppl_r = 2 ** log2loss.mean(dim=0)
    ppl = torch.cat([ppl_r.mean().unsqueeze(0), ppl_r], dim=0)
    return ppl[ranks]


def get_metrics(
    loss: torch.FloatTensor = None,
    y_predict: torch.Tensor = None,
    y_true: torch.Tensor = None,
    n: torch.Tensor = None,
    prefix: str = "",
):
    metrics = {}

    if loss is not None:
        metrics = {f"{prefix}loss": loss.sum(dim=1).mean()}

        ranks = torch.tensor([1, 5, 10, 0], device=loss.device)
        batch_ppl = perplexity(loss, ranks=ranks).detach()
        ppl_dict = {
            f"{prefix}ppl{k}": batch_ppl[i]
            for i, k in enumerate(["@1", "@5", "@10", "_avg"])
        }
        metrics = {**metrics, **ppl_dict}

    if y_true is not None:
        n_clamp = torch.clamp(n, max=y_true.shape[1])
        ranks = torch.tensor([1, 5, 10, 0], device=y_predict.device)
        batch_ndcg = ndcg(y_predict, y_true, n_clamp, ranks=ranks).mean(dim=0).detach()
        ndcg_dict = {
            f"{prefix}ndcg{k}": batch_ndcg[i]
            for i, k in enumerate(["@1", "@5", "@10", ""])
        }
        metrics = {**metrics, **ndcg_dict}

    return metrics

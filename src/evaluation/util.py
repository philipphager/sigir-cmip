from typing import Dict, List


def join_metrics(metrics: List[Dict[str, float]], stage: str = "") -> Dict[str, float]:
    """
    Merges a list of dictionaries (containing metrics values) into a single dictionary
    and adds the model stage (val, test) as a prefix.
    """
    output = {}

    for metric in metrics:
        for k, v in metric.items():
            output[f"{stage}_{k}"] = v

    return output

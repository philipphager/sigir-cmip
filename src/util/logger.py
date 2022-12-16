import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from src.data.dataset import ClickDatasetStats


def log_dataset_stats(
    click_stats: ClickDatasetStats, wandb_logger: WandbLogger, config: DictConfig
):
    """
    Log dataset stats to WandB.
    """
    average_clicks_per_rank = (
        torch.mean(click_stats.document_rank_clicks, dim=0)
        / torch.mean(click_stats.document_rank_impressions, dim=0)
    ).tolist()
    wandb_logger.log_table(
        key="Appendix/average_clicks_per_rank",
        columns=[str(i) for i in range(1, config.data.n_results + 1)],
        data=[average_clicks_per_rank],
    )
    wandb_logger.log_table(
        key="Appendix/example_docs",
        columns=[str(i) for i in range(1, config.data.n_results + 1)],
        data=click_stats.document_rank_impressions[:10].tolist(),
    )

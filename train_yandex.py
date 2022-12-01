import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.data.yandex import ParquetDataset, ParquetClickDataset
from src.util.cache import cache
from src.util.file import get_checkpoint_directory, hash_config

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings("ignore", ".*exists and is not empty*")

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(config.random_state)

    dataset = ParquetClickDataset(
        "/Users/philipphager/.ltr_datasets/cache/Yandex-Clicks.parquet", batch_size=256
    )

    datamodule = instantiate(
        config.datamodule,
        datasets={"train": dataset, "val": dataset},
    )

    trainer = instantiate(config.train_val_trainer)
    model = instantiate(config.model, n_documents=1_000_000)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()

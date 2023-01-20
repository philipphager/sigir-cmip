import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from src.util.hydra import ConfigWrapper

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info("Pre-populating cache")
    seed_everything(config.random_state)

    dataset = instantiate(config.data, config_wrapper=ConfigWrapper(config))
    dataset.setup("fit")
    dataset.setup("test")


if __name__ == "__main__":
    main()

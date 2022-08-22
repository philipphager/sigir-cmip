import os

import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.debug(OmegaConf.to_yaml(config))
    logger.debug("Working directory : {}".format(os.getcwd()))

    dataset = instantiate(config.data)
    print(dataset.load("train"))


if __name__ == "__main__":
    main()

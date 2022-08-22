import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    print(OmegaConf.to_yaml(config))
    print("Working directory : {}".format(os.getcwd()))

    dataset = instantiate(config.data)
    print(dataset.load("train"))


if __name__ == "__main__":
    main()

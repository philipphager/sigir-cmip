import logging
import os

import hydra
import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.debug(OmegaConf.to_yaml(config))
    logger.debug("Working directory : {}".format(os.getcwd()))

    dataset = instantiate(config.data)
    train = dataset.load("train")

    train_simulator = instantiate(config.train_simulator)
    train_clicks = train_simulator(train)
    val_clicks = train_simulator(train)  # Fixme: Relative generation

    test_simulator = instantiate(config.test_simulator)
    test_clicks = test_simulator(train)

    train_loader = instantiate(config.train_loader, dataset=train_clicks)
    val_loader = instantiate(config.test_loader, dataset=val_clicks)
    test_loader = instantiate(config.test_loader, dataset=test_clicks)

    trainer = instantiate(config.trainer)
    model = instantiate(config.model)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    logging.debug(
        f"Inferred examination probability: {model.examination(torch.arange(10))}"
    )


if __name__ == "__main__":
    main()

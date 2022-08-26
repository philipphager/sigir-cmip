import logging
import os

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))

    dataset = instantiate(config.data)
    train = dataset.load("train")
    n_documents = train.n.sum() + 1

    train_simulator = instantiate(config.train_simulator)
    train_clicks = train_simulator(train)

    val_simulator = instantiate(config.val_simulator)
    val_clicks = val_simulator(train)

    test_simulator = instantiate(config.test_simulator)
    test_clicks = test_simulator(train)

    train_loader = instantiate(config.train_loader, dataset=train_clicks)
    val_loader = instantiate(config.val_test_loader, dataset=val_clicks)
    test_loader = instantiate(config.val_test_loader, dataset=test_clicks)

    trainer = instantiate(config.trainer)
    model = instantiate(config.model, n_documents=n_documents)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")

    logging.info(
        f"Inferred examination probability: {model.examination(torch.arange(10))}"
    )


if __name__ == "__main__":
    main()

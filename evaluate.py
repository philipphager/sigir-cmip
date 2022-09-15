import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(config.random_state)

    dataset = instantiate(config.data)
    train = dataset.load("train")
    n_documents = train.n.sum() + 1

    test_simulator = instantiate(config.test_simulator)
    test_clicks = test_simulator(train)

    datamodule = instantiate(
        config.datamodule, datasets={"test_clicks": test_clicks, "test_rels": train}
    )

    with open(config.data.base_dir + "wandb_id.txt", "r") as f:
        wandb_id = f.readline()
    wandb_logger = instantiate(config.test_logger, id=wandb_id)

    trainer = instantiate(config.test_trainer, logger=wandb_logger)
    model = instantiate(config.model, n_documents=n_documents)

    trainer.test(
        model,
        datamodule,
        ckpt_path=config.data.base_dir + "checkpoints/" + config.filename + ".ckpt",
    )


if __name__ == "__main__":
    main()

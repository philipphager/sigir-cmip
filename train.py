import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.util.file import hash_config, get_checkpoint_directory

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

    dataset = instantiate(config.data)
    train = dataset.load("train")
    n_documents = train.n.sum() + 1

    train_simulator = instantiate(config.train_simulator)
    train_clicks = train_simulator(train)

    val_simulator = instantiate(config.val_simulator)
    val_clicks = val_simulator(train)

    datamodule = instantiate(
        config.datamodule, datasets={"train": train_clicks, "val": val_clicks}
    )

    checkpoint_path = get_checkpoint_directory(config)
    checkpoint_path.unlink(missing_ok=True)

    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))
    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb_logger.experiment.config.update(wandb_config)
    trainer = instantiate(config.train_val_trainer, logger=wandb_logger)
    model = instantiate(config.model, n_documents=n_documents)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()

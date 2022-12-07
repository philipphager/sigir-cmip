import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.util.file import get_checkpoint_directory, hash_config
from src.util.hydra import ConfigWrapper

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(config.random_state)

    dataset = instantiate(config.data, config_wrapper=ConfigWrapper(config))
    dataset.setup("fit")

    checkpoint_path = get_checkpoint_directory(config)
    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))

    trainer = instantiate(config.test_trainer, logger=wandb_logger)
    model = instantiate(
        config.model,
        n_documents=dataset.get_n_documents(),
        train_stats=dataset.get_train_stats(),
        lp_scores=dataset.get_train_policy_scores(),
    )

    trainer.test(model, dataset, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()

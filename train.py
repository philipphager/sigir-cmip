import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

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

    dataset = instantiate(config.data)
    dataset.setup("fit")

    checkpoint_path = get_checkpoint_directory(config)
    checkpoint_path.unlink(missing_ok=True)

    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))
    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb_logger.experiment.config.update(wandb_config)

    trainer = instantiate(config.train_val_trainer)
    model = instantiate(
        config.model,
        n_documents=dataset.get_n_documents(),
        lp_scores=dataset.get_train_policy_scores(),
    )
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()

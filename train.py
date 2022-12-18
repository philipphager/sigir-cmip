import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.model.base import NeuralClickModel, StatsClickModel
from src.util.file import get_checkpoint_directory, hash_config
from src.util.hydra import ConfigWrapper
from src.util.logger import log_dataset_stats

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings("ignore", ".*exists and is not empty*")

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    if os.path.exists(config.base_dir + "checkpoints/" + hash_config(config) + ".ckpt"):
        print("Checkpoint found, skipping training...")
        return

    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(config.random_state)

    dataset = instantiate(config.data, config_wrapper=ConfigWrapper(config))
    dataset.setup("fit")

    checkpoint_path = get_checkpoint_directory(config)
    checkpoint_path.unlink(missing_ok=True)

    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))
    wandb_config = OmegaConf.to_container(config, resolve=True)
    wandb_logger.experiment.config.update(wandb_config)
    log_dataset_stats(dataset.get_train_stats(), wandb_logger, config)

    early_stopping = instantiate(config.early_stopping)
    progress_bar = instantiate(config.progress_bar)
    model_checkpoint = instantiate(
        config.model_checkpoint, filename=hash_config(config)
    )

    trainer = instantiate(
        config.train_val_trainer,
        logger=wandb_logger,
        callbacks=[early_stopping, progress_bar, model_checkpoint],
    )
    model = instantiate(
        config.model,
        n_documents=dataset.get_n_documents(),
        n_queries=dataset.get_n_queries(),
        train_stats=dataset.get_train_stats(),
        lp_scores=dataset.get_train_policy_scores(),
    )

    if isinstance(model, NeuralClickModel):
        trainer.fit(model, dataset)
    elif isinstance(model, StatsClickModel):
        trainer.validate(model, dataset)
        trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()

import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.util.cache import cache
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

    @cache(config.data.base_dir, "dataset", [config.data, config.random_state])
    def load_dataset(config):
        dataset = instantiate(config.data)
        return dataset.load("train")

    # Check if we should train on a partial dataset not the full one.
    @cache(
        config.data.base_dir,
        "policies",
        [config.data, config.train_policy, config.random_state],
    )
    def load_policy(config, dataset):
        policy = instantiate(config.train_policy)
        policy.fit(dataset)
        return policy.predict(dataset)

    @cache(
        config.data.base_dir,
        "train_clicks",
        [config.data, config.train_policy, config.train_simulator, config.random_state],
    )
    def simulate_train(config, dataset, policy):
        simulator = instantiate(config.train_simulator)
        return simulator(dataset, policy)

    @cache(
        config.data.base_dir,
        "val_clicks",
        [config.data, config.train_policy, config.val_simulator, config.random_state],
    )
    def simulate_val(config, dataset, policy):
        simulator = instantiate(config.val_simulator)
        return simulator(dataset, policy)

    dataset = load_dataset(config)
    policy = load_policy(config, dataset)

    n_documents = dataset.n.sum() + 1
    train_clicks = simulate_train(config, dataset, policy)
    val_clicks = simulate_val(config, dataset, policy)
    datamodule = instantiate(
        config.datamodule,
        datasets={"train": train_clicks, "val": val_clicks},
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

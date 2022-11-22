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
        [config.data, config.test_policy, config.random_state],
    )
    def load_policy(config, dataset):
        policy = instantiate(config.test_policy)
        policy.fit(dataset)
        return policy.predict(dataset)

    @cache(
        config.data.base_dir,
        "test_clicks",
        [config.data, config.test_policy, config.test_simulator, config.random_state],
    )
    def simulate_test(config, dataset, policy):
        simulator = instantiate(config.test_simulator)
        return simulator(dataset, policy)

    dataset = load_dataset(config)
    policy = load_policy(config, dataset)

    n_documents = dataset.n.sum() + 1
    test_clicks = simulate_test(config, dataset, policy)
    datamodule = instantiate(
        config.datamodule,
        datasets={"test_clicks": test_clicks, "test_rels": dataset},
    )

    checkpoint_path = get_checkpoint_directory(config)
    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))
    trainer = instantiate(config.test_trainer, logger=wandb_logger)
    model = instantiate(config.model, n_documents=n_documents, lp_scores=policy)

    trainer.test(model, datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()

import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.util.cache import cache
from src.util.file import hash_config, get_checkpoint_directory

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(config: DictConfig):
    logger.info(OmegaConf.to_yaml(config))
    logger.info("Working directory : {}".format(os.getcwd()))
    seed_everything(config.random_state)

    @cache(config.data.base_dir, "dataset", [config.data])
    def load_dataset(config):
        dataset = instantiate(config.data)
        return dataset.load("train")

    @cache(config.data.base_dir, "test_clicks", [config.data, config.test_simulator])
    def simulate_test(config, dataset):
        simulator = instantiate(config.test_simulator)
        return simulator(dataset)

    dataset = load_dataset(config)
    n_documents = dataset.n.sum() + 1

    test_clicks = simulate_test(config, dataset)
    datamodule = instantiate(
        config.datamodule,
        datasets={"test_clicks": test_clicks, "test_rels": dataset},
    )

    checkpoint_path = get_checkpoint_directory(config)
    wandb_logger = instantiate(config.wandb_logger, id=hash_config(config))
    trainer = instantiate(config.test_trainer, logger=wandb_logger)
    model = instantiate(config.model, n_documents=n_documents)

    trainer.test(model, datamodule, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    main()

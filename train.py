import logging
import os
import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import rank_zero_only

warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*"
)
warnings.filterwarnings("ignore", ".*exists and is not empty*")

logger = logging.getLogger(__name__)


@rank_zero_only
def write_wandb_id(trainer: Trainer, dirpath: str):
    with open(dirpath + "wandb_id.txt", "w") as f:
        f.write(trainer.logger.experiment.id)


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

    if os.path.exists(
        config.data.base_dir + "checkpoints/" + config.filename + ".ckpt"
    ):
        os.remove(config.data.base_dir + "checkpoints/" + config.filename + ".ckpt")
    trainer = instantiate(config.train_val_trainer)
    model = instantiate(config.model, n_documents=n_documents)

    trainer.fit(model, datamodule)

    """logging.info(
        f"Inferred examination probability: {model.examination(torch.arange(10, device = model.device))}"
    )"""

    write_wandb_id(trainer, config.data.base_dir)


if __name__ == "__main__":
    main()

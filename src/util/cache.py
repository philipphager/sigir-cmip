import functools
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, List, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def cache(
    base_directory: Union[str, Path],
    directory: Union[str, Path],
    configs: List[DictConfig],
) -> Any:
    """
    Decorator for storing the output of a function to disk using a list of Hydra configs
    as hash keys. Thus, an annotated function is only executed when the config changes.
    Hydra configs are used for hashing instead of function arguments to avoid adding
    hashing to custom datasets, tensors, etc.

    :param base_directory: Main cache directory.
    :param directory: Subdirectory to use inside the main cache directory.
    :param configs: Hydra DictConfigs to use as hash key for the wrapped function.
    :return: Original function output as returned by pickle.
    """

    def hash_configs(configs):
        key = hashlib.sha256()

        for config in configs:
            # Ensure to resolves interpolated values, e.g.: ${train_policy}
            if isinstance(config, OmegaConf) or isinstance(config, DictConfig):
                OmegaConf.resolve(config)

            key.update(str.encode(str(config)))

        return key.hexdigest()

    def wrapper_factory(func: Callable):
        base_dir = (Path(base_directory) / Path(directory)).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)
        key = hash_configs(configs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            file = Path(f"{key}.pickle")
            path = (base_dir / file).resolve()

            if not path.exists():
                output = func(*args, **kwargs)
                pickle.dump(output, open(path, "wb"))
                logger.info(f"Storing output of '{func.__name__}' to: {path}")
                return output
            else:
                logger.info(f"Loading output of '{func.__name__}' from: {path}")
                return pickle.load(open(path, "rb"))

        return wrapper

    return wrapper_factory

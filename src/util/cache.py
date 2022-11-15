import functools
import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Union, Any

logger = logging.getLogger(__name__)


def hash_args(*args, **kwargs) -> str:
    kwargs = [f"{k}={v}" for k, v in sorted(kwargs.items())]
    args = [str(k) for k in args]
    key = "_".join(args + kwargs)
    return key


def method_args(*args, **kwargs) -> str:
    kwargs = [f"{k}={v}" for k, v in sorted(kwargs.items())]
    args = [str(k) for k in args[1:]]
    key = "_".join(args + kwargs)
    return key


def cache(directory: Union[str, Path], hash_method: Callable = hash_args) -> Any:
    """
    Decorator to store the output of a function to disk using function arguments as
    hash key. The base directory of the cache is specified by setting the ENV variable,
    by default:

    export CACHE_DIRECTORY = ~/.ltr_datasets/

    :param directory: Subdirectory to use inside the main cache directory.
    :param hash_method: Creates a string key based on method arguments.
        Use `hash_args` to hash all function arguments.
        Use `method_args` to ignore `self` or `cls` parameters.
    :return: Original function output as returned by pickle.
    """

    def wrapper_factory(func: Callable):
        base_dir = os.getenv("CACHE_DIRECTORY", "~/.ltr_datasets/")
        base_dir = (Path(base_dir) / Path(directory)).expanduser()
        base_dir.mkdir(parents=True, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = hash_method(*args, **kwargs)
            file = Path(f"{key}.pickle")
            path = (base_dir / file).resolve()

            if not path.exists():
                output = func(*args, **kwargs)
                pickle.dump(output, open(path, "wb"))
                print(f"Storing output of '{func.__name__}' to: {path}")
                return output
            else:
                print(f"Loading output of '{func.__name__}' from: {path}")
                return pickle.load(open(path, "rb"))

        return wrapper

    return wrapper_factory

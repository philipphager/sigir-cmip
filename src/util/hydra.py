from omegaconf import DictConfig


class ConfigWrapper:
    """
    Hydra instantiate tries to use any DictConfig (or plain dictionary) that looks
    like a Hydra config file. Passing `config.data` as an additional parameter
    (e.g. for internal caching inside the model) will throw an error.

    >>> instantiate(config.model, data_config=config.data)

    This wrapper is a workaround to enable passing config files.
    >>> instantiate(config.model, data_config=ConfigWrapper(config.data))
    """

    def __init__(self, config: DictConfig):
        self.config = config

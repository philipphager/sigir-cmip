from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd
from loguru import logger

from src.data.preprocessing import Pipeline, RatingDataset


class DatasetLoader(ABC):
    def __init__(
        self,
        name: str,
        fold: int,
        n_results: int,
        load_features: bool,
        pipeline: Pipeline,
        base_dir: str,
    ):
        self.name = name
        self.fold = fold
        self.n_results = n_results
        self.load_features = load_features
        self.pipeline = pipeline
        self.base_dir = Path(base_dir).expanduser()

        assert fold in self.folds

    @property
    def cache_directory(self) -> Path:
        """
        Directory to cache pre-processed datasets
        """
        path = self.base_dir / "cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_directory(self) -> Path:
        """
        Directory for extracted datasets
        """
        path = self.base_dir / "dataset"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def download_directory(self) -> Path:
        """
        Download directory for raw dataset .zip files
        """
        path = self.base_dir / "download"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, split: str) -> RatingDataset:
        logger.debug(f"Loading {self.name}, fold: {self.fold}, split: {split}")
        assert split in self.splits, f"Split must one of {self.splits}"
        path = self.cache_directory / f"{self.name}-{self.fold}-{split}.parquet"

        if not path.exists():
            df = self._parse(split, self.load_features)
            df.to_parquet(path)

        df = pd.read_parquet(path)
        return self.pipeline(df)

    @property
    @abstractmethod
    def folds(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def splits(self) -> List[str]:
        pass

    @abstractmethod
    def _parse(self, split: str, load_features: bool) -> pd.DataFrame:
        pass

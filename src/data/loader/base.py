import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import pandas as pd

from src.data.dataset import RatingDataset
from src.data.loader.preprocessing import Pipeline

logger = logging.getLogger(__name__)


class RatingDatasetLoader(ABC):
    """
    Base class for downloading and preprocessing supervised LTR datasets
    with relevance annotations.
    """

    def __init__(
        self,
        name: str,
        fold: int,
        load_features: bool,
        pipeline: Pipeline,
        base_dir: str,
    ):
        self.name = name
        self.fold = fold
        self.load_features = load_features
        self.pipeline = pipeline
        self.base_dir = Path(base_dir).expanduser()

        assert fold in self.folds

    @property
    def rating_directory(self) -> Path:
        """
        Directory to pre-processed rating datasets
        """
        path = self.base_dir / "rating-dataset"
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
        logger.info(f"Loading {self.name}, fold: {self.fold}, split: {split}")
        assert split in self.splits, f"Split must one of {self.splits}"
        path = self.rating_directory / f"{self.name}-{self.fold}-{split}.parquet"

        if not path.exists():
            df = self._parse(split, self.load_features)
            df = self.pipeline(df)
            df.to_parquet(path)

        return RatingDataset(path)

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

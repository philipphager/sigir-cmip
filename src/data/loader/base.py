import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, TypeVar, Union

import pandas as pd

from src.data.dataset import FeatureRatingDataset, RatingDataset
from src.data.loader.preprocessing import Pipeline

logger = logging.getLogger(__name__)


T = TypeVar("T")


class Loader(Generic[T], ABC):
    def __init__(self, base_dir: Union[Path, str]):
        self.base_dir = Path(base_dir).expanduser()

    @property
    def output_directory(self) -> Path:
        """
        Directory for pre-processed datasets
        """
        path = self.base_dir / "processed"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_directory(self) -> Path:
        """
        Download directory for extracting zipped datasets
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

    @abstractmethod
    def load(self, **kwargs) -> T:
        pass


class RatingLoader(Loader[RatingDataset]):
    """
    Base class for downloading and preprocessing supervised LTR datasets
    with relevance annotations.
    """

    def __init__(
        self,
        name: str,
        fold: int,
        base_dir: str,
        pipeline: Optional[Pipeline] = None,
    ):
        super(RatingLoader, self).__init__(base_dir)
        self.name = name
        self.fold = fold
        self.pipeline = pipeline
        self.base_dir = Path(base_dir).expanduser()

    def load(self, split: str, load_features: bool = False) -> RatingDataset:
        assert self.fold in self.folds, f"Fold must one of {self.folds}"
        assert split in self.splits, f"Split must one of {self.splits}"
        logger.info(
            f"Loading {self.name}, fold: {self.fold}, "
            f"split: {split}, features: {load_features}"
        )

        df = self._parse(split)

        if self.pipeline is not None:
            df = self.pipeline(df)

        return FeatureRatingDataset(df) if load_features else RatingDataset(df)

    @property
    @abstractmethod
    def folds(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def splits(self) -> List[str]:
        pass

    @abstractmethod
    def _parse(self, split: str) -> pd.DataFrame:
        pass

import logging
from typing import List

import pandas as pd

from src.data.dataset import ParquetClickDataset
from src.data.loader.base import DatasetLoader, RatingDatasetLoader
from src.util.file import download, verify_file, extract, copy_file

logger = logging.getLogger(__name__)


class YandexRatingLoader(RatingDatasetLoader):
    url = "https://www.dropbox.com/s/xo69hdfcf1k46oi/Yandex.zip?dl=1"
    zip_file = "Yandex.zip"
    file = "Yandex"
    checksum = "eaf66ce121a25a35ded7c673e8460ae5b11069b960112c7ec382bd179e7ff3c0"

    def __init__(
        self,
        name: str,
        fold: int,
        base_dir: str,
    ):
        super().__init__(name, fold, base_dir)

    def _parse(self, split: str, load_features: bool) -> pd.DataFrame:
        zip_path = download(self.url, self.download_directory / self.zip_file)
        verify_file(zip_path, self.checksum)
        dataset_path = extract(zip_path, self.dataset_directory / self.file)
        path = dataset_path / self.file / "relevance.parquet"

        return pd.read_parquet(path)

    @property
    def folds(self) -> List[int]:
        return [1]

    @property
    def splits(self) -> List[str]:
        return ["train"]


class YandexClickLoader(DatasetLoader[ParquetClickDataset]):
    url = "https://www.dropbox.com/s/xo69hdfcf1k46oi/Yandex.zip?dl=1"
    zip_file = "Yandex.zip"
    file = "Yandex"
    checksum = "eaf66ce121a25a35ded7c673e8460ae5b11069b960112c7ec382bd179e7ff3c0"

    def __init__(self, name: str, fold: int, base_dir: str):
        super().__init__(base_dir)
        self.name = name
        self.fold = fold

    def load(self, split: str, batch_size: int) -> ParquetClickDataset:
        assert self.fold in self.folds, f"Fold must one of {self.folds}"
        assert split in self.splits, f"Split must one of {self.splits}"
        logger.info(f"Loading {self.name}, fold: {self.fold}, split: {split}")

        zip_path = download(self.url, self.download_directory / self.zip_file)
        verify_file(zip_path, self.checksum)
        dataset_path = extract(zip_path, self.dataset_directory / self.file)

        # Copying already preprocessed click file to output directory
        in_path = dataset_path / self.file / "clicks.parquet"
        out_path = self.output_directory / f"{self.name}-{self.fold}-{split}.parquet"
        copy_file(in_path, out_path)

        return ParquetClickDataset(out_path, batch_size)

    @property
    def folds(self) -> List[int]:
        return [1]

    @property
    def splits(self) -> List[str]:
        return ["train"]

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pyarrow import Table
from pyarrow.parquet import ParquetFile
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, IterableDataset

from src.util.tensor import scatter_rank_add

logger = logging.getLogger(__name__)


class RatingDataset(Dataset):
    def __init__(self, path: Union[str, Path]):
        self.path = path

        df = pd.read_parquet(self.path)
        assert all([c in df.columns for c in ["query_id", "doc_ids", "relevance"]])

        self.query_id = torch.tensor(df["query_id"])
        self.n = torch.tensor(df["doc_ids"].map(len))
        self.x = self.pad(df["doc_ids"])
        self.y = self.pad(df["relevance"])

    @staticmethod
    def pad(column: List[List[int]]):
        """
        Pad a list of variable-sized lists to max length
        """
        return pad_sequence([torch.tensor(y) for y in column], batch_first=True)

    def __getitem__(self, i):
        return self.query_id[i], self.x[i], self.y[i], self.n[i]

    def __len__(self):
        return len(self.query_id)


class ClickDataset(Dataset):
    def __init__(
        self,
        query_ids: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        y_click: torch.Tensor,
        n: torch.Tensor,
    ):
        self.query_ids = query_ids
        self.x = x
        self.y = y
        self.y_click = y_click
        self.n = n

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i: int):
        return self.query_ids[i], self.x[i], self.y_click[i], self.n[i]

    def get_document_rank_clicks(self, n_documents) -> torch.Tensor:
        return scatter_rank_add(self.y_click, self.x, n_documents)

    def get_document_rank_impressions(self, n_documents) -> torch.Tensor:
        impressions = (self.x > 0).float()
        return scatter_rank_add(impressions, self.x, n_documents)


class ParquetClickDataset(IterableDataset):
    """
    Loads a click dataset from a .parquet file, expecting the following columns:

    query_id, doc_ids, click
    1, [50, 51, 52], [False, False, True]

    The dataset is an iterable dataset, since it reads batches directly
    from the compressed file. Thus, automatic batching in the dataloader
    should be disabled:

    >>> dataset = ParquetClickDataset(batch_size=256)
    >>> loader = DataLoader(dataset, batch_size=None)

    When using multiple workers, the dataset is split into equal chunks to avoid
    parallel iterations as suggested in the official documentation:
    https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
    """

    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int,
        row_group_subset: Optional[Tuple[int, int]] = None,
    ):
        self.path = Path(path)
        self.batch_size = batch_size
        self.row_group_subset = row_group_subset
        self.row_groups = self._get_row_groups()

    def __iter__(self):
        file = ParquetFile(self.path)
        logger.info(f"New worker iterating {len(self.row_groups)} groups")

        return map(
            self.collate_clicks,
            file.iter_batches(self.batch_size, self.row_groups),
        )

    def split(self, train_size: float, shuffle: bool):
        train, test = train_test_split(
            self.row_groups,
            train_size=train_size,
            shuffle=shuffle,
        )
        return (
            ParquetClickDataset(self.path, self.batch_size, train),
            ParquetClickDataset(self.path, self.batch_size, test),
        )

    def _get_row_groups(
        self,
    ) -> List[int]:
        file = ParquetFile(self.path)
        n_workers, worker_id = self.get_worker_info()

        if self.row_group_subset is None:
            row_groups = np.arange(file.num_row_groups)
        else:
            row_groups = np.array(self.row_group_subset)

        return list(np.array_split(row_groups, n_workers)[worker_id])

    @staticmethod
    def get_worker_info():
        worker_info = torch.utils.data.get_worker_info()
        n_workers = 1
        worker_id = 0

        if worker_info is not None:
            n_workers = worker_info.num_workers
            worker_id = worker_info.id

        return n_workers, worker_id

    @staticmethod
    def collate_clicks(batch: Table):
        # Convert arrow table to dict of format: {"query_id": [...], ...}
        batch = batch.to_pydict()

        # Convert to torch tensors
        query_ids = torch.tensor(batch["query_id"])
        x = torch.tensor(batch["doc_ids"])
        y_click = torch.tensor(batch["click"]).int()

        n_batch, n_items = x.shape
        n = torch.full((n_batch,), n_items)

        return query_ids, x, y_click, n

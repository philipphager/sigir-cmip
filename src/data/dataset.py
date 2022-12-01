import logging
import math
from abc import ABC
from pathlib import Path
from typing import List, Union, Callable

import torch
from pyarrow import Table
from pyarrow.parquet import ParquetFile
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, DataLoader


logger = logging.getLogger(__name__)


class ParquetDataset(IterableDataset, ABC):
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

    def __init__(self, path: Union[str, Path], batch_size: int, collate_fn: Callable):
        super(ParquetDataset).__init__()
        self.path = Path(path)
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def __iter__(self):
        file = ParquetFile(self.path)

        n_workers, worker_id = self.get_worker_info()
        row_groups = self.get_row_groups(file.num_row_groups, n_workers, worker_id)
        logger.info(f"Worker with id: {worker_id} iterating {len(row_groups)} groups")

        return map(self.collate_fn, file.iter_batches(self.batch_size, row_groups))

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
    def get_row_groups(total_groups: int, n_workers: int, worker_id: id) -> List[int]:
        assert n_workers <= total_groups, (
            f"Cannot split {total_groups} row groups between {n_workers}."
            f"Use less workers or create a .parquet file with more row groups."
        )

        groups_per_worker = math.ceil(total_groups / n_workers)
        start = worker_id * groups_per_worker
        end = min(start + groups_per_worker, total_groups)

        return list(range(start, end))


class ParquetClickDataset(ParquetDataset):
    def __init__(self, path: Union[str, Path], batch_size: int):
        super().__init__(path, batch_size, self.collate_clicks)

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

        return query_ids, x, y_click, y_click, n


class ParquetRelevanceDataset(ParquetDataset):
    def __init__(self, path: Union[str, Path], batch_size: int):
        super().__init__(path, batch_size, self.collate_relevance)

    @staticmethod
    def collate_relevance(batch: Table):
        # Convert arrow table to dict of format: {"query_id": [...], ...}
        batch = batch.to_pydict()

        # Convert to torch tensors and pad all tensors in batch.
        query_ids = torch.tensor(batch["query_id"])
        n = torch.tensor([len(x) for x in batch["doc_ids"]])
        x = pad_sequence(
            [torch.tensor(x) for x in batch["doc_ids"]],
            batch_first=True,
        )
        y = pad_sequence(
            [torch.tensor(y) for y in batch["relevance"]],
            batch_first=True,
        )

        return query_ids, x, y, n

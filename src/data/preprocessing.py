import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RatingDataset(Dataset):
    def __init__(
        self,
        query_ids: torch.LongTensor,
        doc_ids: torch.LongTensor,
        y: torch.LongTensor,
        n: torch.LongTensor,
    ):
        self.query_ids = query_ids
        self.doc_ids = doc_ids
        self.y = y
        self.n = n

    def __len__(self):
        return len(self.query_ids)

    def __getitem__(self, i: int):
        return self.query_ids[i], self.doc_ids[i], self.y[i], self.n[i]


class Step(ABC):
    @abstractmethod
    def __call__(self, df) -> pd.DataFrame:
        pass


class Pipeline:
    def __init__(
        self,
        steps: List[Callable] = [],
    ):
        self.steps = steps

    def __call__(self, df: pd.DataFrame) -> RatingDataset:
        logger.info("Pre-processing:")

        for step in self.steps:
            df = step(df)

        return self._to_torch(df)

    def _to_torch(self, df: pd.DataFrame):
        logger.info("Converting DataFrame to torch tensors")
        query_df = (
            df.groupby("query_id")
            .agg(doc_ids=("doc_id", list), y=("y", list), n=("y", "count"))
            .reset_index()
        )

        n_queries = len(query_df)
        n_results = query_df["n"].max()

        query_id = torch.zeros((n_queries,), dtype=torch.long)
        doc_ids = torch.zeros((n_queries, n_results), dtype=torch.long)
        y = torch.zeros((n_queries, n_results), dtype=torch.long)
        n = torch.zeros((n_queries,), dtype=torch.long)

        for i, row in query_df.iterrows():
            query_id[i] = row["query_id"]
            doc_ids[i, : row["n"]] = torch.from_numpy(np.array(row["doc_ids"]))
            y[i, : row["n"]] = torch.from_numpy(np.array(row["y"]))
            n[i] = row["n"]

        return RatingDataset(query_id, doc_ids, y, n)


class StratifiedTruncate(Step):
    def __init__(self, max_length: int, random_state: int):
        self.max_length = max_length
        self.random_state = random_state

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            f"Stratified sampling queries to max {self.max_length} documents, "
            f"random_state: {self.random_state}"
        )
        return df.groupby(["query_id"], group_keys=False).apply(self.stratified_sample)

    def stratified_sample(self, df):
        n_results = min(self.max_length, len(df))
        return (
            df.groupby("y")
            .sample(frac=n_results / len(df), random_state=self.random_state)
            .tail(n_results)
        )


class Shuffle(Step):
    def __init__(self, random_state: int):
        self.random_state = random_state

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(
            f"Uniformly shuffle documents per query, "
            f"random_state: {self.random_state}"
        )
        return (
            df.groupby("query_id")
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )


class GenerateDocumentIds(Step):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating surrogate document ids")
        df["doc_id"] = np.arange(len(df))
        return df

import logging
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Pre-processing:")

        for step in self.steps:
            df = step(df)

        return self._aggregate_query(df)

    @staticmethod
    def _aggregate_query(df: pd.DataFrame) -> pd.DataFrame:
        agg = {"doc_id": list, "y": list}

        if "features" in df.columns:
            agg["features"] = list

        df = (
            df.groupby("query_id")
            .aggregate(agg)
            .reset_index()
            .rename(columns={"doc_id": "doc_ids", "y": "relevance"})
        )

        df["query_id"] = np.arange(len(df)) + 1

        return df


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


class DiscardShortQueries(Step):
    def __init__(self, min_length: int):
        self.min_length = min_length

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Discarding queries with less than {self.min_length} documents")
        query_df = df.groupby("query_id").agg(n_results=("y", "count")).reset_index()
        query_df = query_df[query_df.n_results >= self.min_length]

        return df[df.query_id.isin(query_df.query_id)]


class GenerateDocumentIds(Step):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Generating surrogate document ids")
        df["doc_id"] = np.arange(len(df)) + 1
        return df

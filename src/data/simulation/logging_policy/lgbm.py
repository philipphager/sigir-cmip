from typing import Optional

import torch
from lightgbm import LGBMRanker, early_stopping

from src.data.dataset import FeatureRatingDataset
from src.data.simulation.logging_policy.base import LoggingPolicy
from src.model.loss import mask_padding


class LightGBMRanker(LoggingPolicy):
    def __init__(
        self,
        objective: str,
        boosting_type: str,
        metric: str,
        name: str,
        n_estimators: int,
        n_leaves: int,
        learning_rate: float,
        early_stopping_patience: int,
        random_state: int,
    ):
        self.name = name
        self.early_stopping_patience = early_stopping_patience
        self.model = LGBMRanker(
            objective=objective,
            boosting_type=boosting_type,
            metric=metric,
            n_estimators=n_estimators,
            num_leaves=n_leaves,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(
        self, train: FeatureRatingDataset, val: Optional[FeatureRatingDataset] = None
    ):
        x_train, y_train, n_train = self.to_lightgbm(train)
        eval_set = None
        eval_group = None
        callbacks = []

        if val is not None:
            x_val, y_val, n_val = self.to_lightgbm(val)
            eval_set = [(x_val, y_val)]
            eval_group = [n_val]
            callbacks = [
                early_stopping(
                    stopping_rounds=self.early_stopping_patience, verbose=False
                )
            ]

        self.model.fit(
            X=x_train,
            y=y_train,
            group=n_train,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=callbacks,
        )

    def predict(self, dataset: FeatureRatingDataset) -> torch.Tensor:
        query_ids, doc_ids, x, y, n = dataset[:]
        n_batch, n_results, n_features = x.shape

        # (n_batch, n_results, n_features) -> (n_batch * n_results, n_features)
        x = x.reshape(-1, n_features).numpy()
        y_predict = torch.from_numpy(self.model.predict(x))
        y_predict = y_predict.reshape(n_batch, n_results)

        return mask_padding(y_predict, n, fill=-torch.inf)

    def requires_features(self) -> bool:
        return True

    @staticmethod
    def to_lightgbm(dataset: FeatureRatingDataset):
        query_ids, doc_ids, x, y, n = dataset[:]
        n_batch, n_results, n_features = x.shape

        # (n_batch, n_results, n_features) -> (n_batch * n_results, n_features)
        mask = torch.arange(n_results).repeat(n_batch, 1)
        mask = mask < n.unsqueeze(-1)
        x = x[mask].numpy()
        y = y[mask].numpy()
        n = n.numpy()

        return x, y, n

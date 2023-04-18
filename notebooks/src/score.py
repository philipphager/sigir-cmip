from itertools import combinations
from typing import Callable, List

import numpy as np
import scipy.stats as st
from sklearn.metrics import r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.tree import DecisionTreeRegressor


def confidence_interval(x: np.ndarray, interval: float = 0.95):
    low, high = st.t.interval(
        alpha=interval,
        df=len(x) - 1,
        loc=np.mean(x),
        scale=st.sem(x),
    )

    return (high - low) / 2


def adjust_r2(r2, n_samples, n_variables):
    return 1 - (((1 - r2) * (n_samples - 1)) / (n_samples - n_variables - 1))


def repeated_cross_val_r2_score(
    X: np.ndarray,
    y: np.ndarray,
    regressor: Callable = DecisionTreeRegressor,
    n_splits: int = 2,
    n_repeats: int = 100,
    use_adjusted_r2: bool = True,
):
    folds = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
    scores = []

    for train_idx, test_idx in folds.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = regressor()
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        score = r2_score(y_test, y_predict)

        if use_adjusted_r2:
            score = adjust_r2(score, X_test.shape[0], X_test.shape[1])

        scores.append(score)

    return scores


def get_metric_combinations(metrics: List[str]) -> List[List[str]]:
    return [list(j) for i in range(len(metrics)) for j in combinations(metrics, i + 1)]

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from src.evaluation.base import PolicyMetric
from src.evaluation.ci import nearest_neighbor_bootstrap
from src.evaluation.util import add_label, hstack, padding_mask, random_split


class KLDivergence(ABC):
    @abstractmethod
    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> float:
        pass


class ClassifierKLDivergence(KLDivergence):
    """
    Classifier-based estimator of the Kullback–Leibler divergence as in Eq. 3 of
    [Mukherjee et al. 2019](http://auai.org/uai2019/proceedings/papers/403.pdf).
    """

    def __init__(self, classifier: Any, n_bootstrap: int, eta: float):
        self.classifier = classifier
        self.n_bootstrap = n_bootstrap
        self.eta = eta

    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> float:
        p = add_label(p, 1)
        q = add_label(q, 0)
        kl_divergences = [self.kl_divergence(p, q) for _ in range(self.n_bootstrap)]

        return torch.tensor(kl_divergences).mean()

    def kl_divergence(self, p: torch.Tensor, q: torch.Tensor):
        p_train, p_test = random_split(p, 2)
        q_train, q_test = random_split(q, 2)

        train = torch.vstack([p_train, q_train])
        self.train(train)

        p_predict = self.predict(p_test)
        q_predict = self.predict(q_test)

        p_ratio = p_predict / (1 - p_predict)
        q_ratio = q_predict / (1 - q_predict)

        return (p_ratio.log() - q_ratio.mean().log()).mean()

    def train(self, train: torch.Tensor):
        x = train[:, :3].numpy()
        y = train[:, 3].numpy()

        self.classifier.fit(x, y)

    def predict(self, test: torch.Tensor):
        x = test[:, :3]
        y_predict = self.classifier.predict_proba(x.numpy())
        y_predict = torch.tensor(y_predict[:, 1])

        return y_predict.clip(self.eta, 1 - self.eta)


class ConditionalMutualInformation(PolicyMetric):
    """
    Classifier-based Conditional Mutual Information (CCMI) from
    [Mukherjee et al. 2019](http://auai.org/uai2019/proceedings/papers/403.pdf).

    Main idea follows the mimic and classify schema by [Sen et al., 2017].
    Given the joint distribution P(X, Y, Z), we mimic a dataset in which
    X and Y are independent given Z using the knn method in [Sen et al., 2017].

    The conditional mutual information is then the KL divergence between the original
    distribution and the marginal distribution in which X and Y are independent.
    """

    def __init__(
        self,
        name: str,
        kl_divergence: KLDivergence,
        n_bootstrap: int,
    ):
        self.name = name
        self.kl_divergence = kl_divergence
        self.n_bootstrap = n_bootstrap

    def __call__(
        self,
        y_predict: torch.Tensor,
        y_logging_policy: torch.Tensor,
        y_true: torch.LongTensor,
        n: torch.LongTensor,
    ) -> Dict:
        n_batch, n_results = y_true.shape
        mask = padding_mask(n, n_batch, n_results)
        dataset = hstack(y_predict, y_logging_policy, y_true)
        dataset = dataset[mask.ravel()]

        cmi = []

        for _ in range(self.n_bootstrap):
            split1, split2 = random_split(dataset, splits=2)
            split2 = nearest_neighbor_bootstrap(split1, split2)
            cmi.append(self.kl_divergence(split1, split2))

        return {self.name: torch.tensor(cmi).mean()}

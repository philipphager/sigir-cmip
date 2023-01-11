from abc import ABC, abstractmethod
from typing import Any, Dict

import torch

from src.evaluation.base import PolicyMetric
from src.evaluation.ccit import nearest_neighbor_bootstrap
from src.evaluation.util import add_label, hstack, padding_mask, random_split


class KLDivergence(ABC):
    @abstractmethod
    def __call__(self, p: torch.Tensor, q: torch.Tensor) -> float:
        pass


class ClassifierKLDivergence(KLDivergence):
    """
    Classifier-based estimator of the Kullback–Leibler divergence as in Eq. 3 and Alg. 1
    of [Mukherjee et al. 2019](http://auai.org/uai2019/proceedings/papers/403.pdf).
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
        classifier = self.train(self.classifier, train)

        p_predict = self.predict(classifier, p_test, self.eta)
        q_predict = self.predict(classifier, q_test, self.eta)

        # Point-wise likelihood ratio as in section 3.1
        p_ratio = p_predict / (1 - p_predict)
        q_ratio = q_predict / (1 - q_predict)

        # Estimate KL divergence using Donsker-Varadhan formulation
        return (p_ratio.log() - q_ratio.mean().log()).mean()

    @staticmethod
    def train(model: Any, train: torch.Tensor):
        x = train[:, :3].cpu().numpy()
        y = train[:, 3].cpu().numpy()
        model.fit(x, y)

        return model

    @staticmethod
    def predict(model: Any, test: torch.Tensor, eta: float):
        x = test[:, :3].cpu().numpy()
        y_predict = torch.tensor(model.predict_proba(x)[:, 1])
        # Clip predictions to avoid exploding likelihood ratios
        return y_predict.clip(eta, 1 - eta)


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

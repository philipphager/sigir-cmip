from typing import Any

import torch

from src.evaluation.base import PolicyMetric
from src.evaluation.util import add_label, hstack, padding_mask, random_split


def nearest_neighbor_bootstrap(split1: torch.Tensor, split2: torch.Tensor):
    """
    This method uses two dataset splits containing observations (x, y, z) to
    generate a third dataset in which x, y are independent conditioned on z.

    This is done by iterating over the first split u1, and finding a nearest
    neighbor in u2 based on the z values. Then we swap the y values of both entries.
    Given that z are the true relevance labels, we avoid the nearest neighbor search
    and from all entries of u2 with a given relevance label to replace the entries
    in u1. This method needs to be adjusted when moving away from int relevance
    labels between 0-4 as z / y_true.

    :param split1: A tensor containing a third of the original dataset
    :param split2: A tensor containing another third of the original dataset
    :return: Split1 with the y entries replaced with samples from split2
    """
    split1 = torch.clone(split1)
    split2 = torch.clone(split2)
    z1 = split1[:, 2]
    z2 = split2[:, 2]

    for z in range(5):
        idx1 = torch.argwhere(z1 == z).ravel()
        idx2 = torch.argwhere(z2 == z).ravel()

        sample_idx = torch.randint(len(idx2), (len(idx1),))
        split1[idx1, 1] = split2[idx2[sample_idx], 1]

    return split1


class PointwiseClassifierCITest(PolicyMetric):
    """
    Classifier Conditional Independence Test (CCIT) from [Sen et al. 2017]

    Uses two-thirds of data to generate a dataset in which X, Y (continuous random vars)
    are conditionally independent given Z. The goal of the independence test is to train
    a classifier that differentiates between the independent dataset and the original
    data. If this can be done better than chance, the original data is not independent.
    """

    def __init__(self, name: str, classifier: Any):
        self.name = name
        self.classifier = classifier

    def __call__(
        self,
        y_predict: torch.Tensor,
        y_logging_policy: torch.Tensor,
        y_true: torch.LongTensor,
        n: torch.LongTensor,
    ):
        n_batch, n_results = y_true.shape
        mask = padding_mask(n, n_batch, n_results)
        dataset = hstack(y_predict, y_logging_policy, y_true)
        dataset = dataset[mask.ravel()]

        split1, split2, split3 = random_split(dataset, splits=3)
        split2 = nearest_neighbor_bootstrap(split2, split3)

        split1 = add_label(split1, 1)
        split2 = add_label(split2, 0)
        dataset = torch.vstack([split1, split2])

        train, test = random_split(dataset, splits=2)
        classifier = self.train_classifier(self.classifier, train)
        loss = self.evaluate(classifier, test)

        threshold = self.get_threshold(len(test))
        is_independent = loss > 0.5 - threshold

        return {self.name: is_independent}

    @staticmethod
    def train_classifier(model: Any, train: torch.Tensor):
        x = train[:, :3].numpy()
        y = train[:, 3].numpy()
        model.fit(x, y)

        return model

    @staticmethod
    def evaluate(model: Any, test: torch.Tensor):
        x = test[:, :3]
        y = test[:, 3]

        y_predict = torch.tensor(model.predict(x.numpy()))
        zero_one_loss = (y_predict != y).float().mean()

        return zero_one_loss

    @staticmethod
    def get_threshold(n: int):
        """
        :param n: Number of samples in the train / test split
        :return: Upper bound on the expected variance of the test statistic
        """
        return 1 / n**0.5

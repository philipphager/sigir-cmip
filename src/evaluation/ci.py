from typing import Any

import torch

from src.evaluation.base import PolicyMetric


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
        mask = self.padding_mask(n, n_batch, n_results)
        dataset = self.hstack(y_predict, y_logging_policy, y_true)
        dataset = dataset[mask.ravel()]

        split1, split2, split3 = self.random_split(dataset, splits=3)
        split2 = self.nearest_neighbor_bootstrap(split2, split3)

        split1 = self.add_label(split1, 1)
        split2 = self.add_label(split2, 0)
        dataset = torch.vstack([split1, split2])

        train, test = self.random_split(dataset, splits=2)
        classifier = self.train_classifier(self.classifier, train)
        loss = self.evaluate(classifier, test)

        threshold = self.get_threshold(len(test))
        is_independent = loss > 0.5 - threshold

        return {self.name: is_independent}

    @staticmethod
    def padding_mask(n, n_batch, n_results):
        mask = torch.arange(n_results).repeat(n_batch, 1).type_as(n)
        return mask < n.unsqueeze(-1)

    @staticmethod
    def hstack(
        y_predict: torch.Tensor,
        y_logging_policy: torch.Tensor,
        y_true: torch.LongTensor,
    ):
        """
        Flatten all tensors to 1d and stack them into columns of a matrix of size:
        (n_results * n_queries) x 3
        Each row contains thus one observation of:
        (y_predict, y_logging_policy, y_true) or (x, y, z)
        """
        return torch.column_stack(
            [
                y_predict.ravel(),
                y_logging_policy.ravel(),
                y_true.ravel(),
            ]
        )

    @staticmethod
    def random_split(x: torch.Tensor, splits: int):
        """
        Shuffle and split a tensor into equal parts
        """
        idx = torch.randperm(len(x))
        return torch.chunk(x[idx], splits)

    @staticmethod
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
        z1 = split1[:, 2]
        z2 = split2[:, 2]

        for z in range(5):
            idx1 = torch.argwhere(z1 == z).ravel()
            idx2 = torch.argwhere(z2 == z).ravel()

            sample_idx = torch.randint(len(idx2), (len(idx1),))
            split1[idx1, 1] = split2[idx2[sample_idx], 1]

        return split1

    @staticmethod
    def add_label(x: torch.Tensor, label: int):
        """
        Adds a new column to the dataset containing the given label
        """
        labels = torch.full((len(x), 1), label)
        return torch.hstack([x, labels])

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

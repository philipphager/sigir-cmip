import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.preprocessing import RatingDataset
from src.simulation.logging_policy import LoggingPolicy
from src.simulation.user_model.base import UserModel


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
        return self.query_ids[i], self.x[i], self.y[i], self.y_click[i], self.n[i]


class Simulator:
    def __init__(
        self, logging_policy: LoggingPolicy, user_model: UserModel, n_sessions: int
    ):
        self.logging_policy = logging_policy
        self.user_model = user_model
        self.n_sessions = n_sessions

    def __call__(self, dataset: RatingDataset, eps: float = 1e-9):
        query_ids, x, y, n = dataset[:]

        # FIXME
        # Get scores from logging policy
        y_predict = self.logging_policy.predict(dataset)

        # Uniform sample queries
        sample_ids = torch.randint(len(query_ids), (self.n_sessions,))
        query_ids = query_ids[sample_ids]
        x = x[sample_ids]
        y = y[sample_ids]
        y_predict = y_predict[sample_ids]
        n = n[sample_ids]

        # Sample logging policy rankings using Gumbel Noise trick
        y_predict = F.gumbel_softmax(
            torch.log(y_predict.clip(min=0))
        )  # Fixme: Proper Gumbel trick?
        idx = torch.argsort(-y_predict)
        x = torch.gather(x, 1, idx)
        y = torch.gather(y, 1, idx)

        # Sample clicks
        click_probabilities = self.user_model(y)
        y_clicks = torch.bernoulli(click_probabilities)

        return ClickDataset(query_ids, x, y, y_clicks, n)

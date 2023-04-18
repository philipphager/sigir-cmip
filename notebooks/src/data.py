from typing import Dict, List, Optional

import pandas as pd
import wandb
from wandb.apis.public import Runs
from wandb.sdk.wandb_run import Run


class WandbLoader:
    def __init__(
        self,
        entity: str,
        project: str,
        experiment_name: str,
        run_name: str,
        model2name: Dict[str, str] = {
            "CACM_minus": "CACM",
            "RankedDCTR": "RDCTR",
        },
    ):
        self.entity = entity
        self.project = project
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.model2name = model2name

    def load_metrics(self):
        dfs = []
        runs = self._load_runs()

        for run in runs:
            try:
                metric_df = self._parse_metric(run)
                meta_df = self._parse_meta(run)
                df = meta_df.merge(metric_df, how="cross")
                df = df[sorted(df.columns)]
                dfs.append(df)
            except Exception as e:
                print(f"Failed to load run: {e}")

        return pd.concat(dfs)

    def load_policy_scores(
        self,
        user_model: str,
        train_policy: str,
        test_policy: str,
        models: List[str],
        temperature: int,
    ):
        dfs = []
        runs = self._load_runs()

        for run in runs:
            try:
                meta_df = self._parse_meta(run)
                meta_df = meta_df[
                    (meta_df["user_model"] == user_model)
                    & (meta_df["train_policy"] == train_policy)
                    & (meta_df["test_policy"] == test_policy)
                    & (meta_df["model"].isin(models))
                    & (meta_df["temperature"] == temperature)
                ]

                if len(meta_df) > 0:
                    policy_df = self._parse_policy(run)

                    if policy_df is not None:
                        df = meta_df.merge(policy_df, how="cross")
                        df = df[sorted(df.columns)]
                        dfs.append(df)
            except Exception as e:
                print(f"Failed to load run: {e}")

        return pd.concat(dfs)

    def _load_runs(self, timeout: int = 30) -> Runs:
        """
        Load runs from Weights & Biases
        """
        return wandb.Api(timeout=timeout).runs(
            f"{self.entity}/{self.project}",
            {
                "$and": [
                    {
                        "config.experiment_name": self.experiment_name,
                        "config.run_name": self.run_name,
                    }
                ]
            },
        )

    def _parse_metric(self, run: Run) -> pd.DataFrame:
        """
        Parse any logged metric starting Metrics/
        """
        metric2value = {
            k.replace("Metrics/", ""): v
            for k, v in run.summary.items()
            if k.startswith("Metrics")
        }

        return pd.DataFrame([metric2value])

    def _parse_meta(self, run: Run) -> pd.DataFrame:
        """
        Parse metadata from class names in the Hydra config
        """
        model = self._extract_target_class(run.config["model"])
        user_model = self._extract_target_class(
            run.config["data"]["train_simulator"]["user_model"]
        )
        train_policy = self._extract_target_class(run.config["data"]["train_policy"])
        test_policy = self._extract_target_class(run.config["data"]["test_policy"])
        temperature = run.config["data"]["train_simulator"]["temperature"]
        random_state = run.config["random_state"]

        # Optionally rename models
        if model in self.model2name:
            model = self.model2name[model]

        return pd.DataFrame(
            [
                {
                    "model": model,
                    "user_model": user_model,
                    "train_policy": train_policy,
                    "test_policy": test_policy,
                    "temperature": temperature,
                    "random_state": random_state,
                }
            ]
        )

    def _parse_policy(self, run: Run) -> Optional[pd.DataFrame]:
        artifacts = [a for a in run.logged_artifacts() if "policy" in a.name]

        if len(artifacts) > 0:
            table = run.use_artifact(artifacts[0]).get("Appendix/policy")
            return pd.DataFrame(table.data, columns=table.columns)
        else:
            return None

    @staticmethod
    def _extract_target_class(config):
        """
        Extract Hydra _target_: src.path.class_name -> class_name
        """
        target_path = config["_target_"]
        return target_path.split(".")[-1]

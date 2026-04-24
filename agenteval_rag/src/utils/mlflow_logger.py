"""
MLflow logger: records per-query eval scores and pipeline parameters.
"""
from typing import Any, Dict, Optional

import mlflow

from config import settings


class MLflowLogger:
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

    # ── Query logging ─────────────────────────────────────────────────────────

    def log_query(
        self,
        query: str,
        query_type: str,
        answer: str,
        eval_scores: Dict[str, float],
        num_docs: int,
        retry_count: int = 0,
        used_web: bool = False,
    ) -> None:
        """Log a complete query → answer → eval cycle as a nested MLflow run."""
        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "query": query[:256],
                    "query_type": query_type,
                    "num_retrieved_docs": num_docs,
                    "retry_count": retry_count,
                    "used_web_fallback": used_web,
                }
            )
            mlflow.log_metrics(
                {k: float(v) for k, v in eval_scores.items()}
            )

    # ── Generic helpers ───────────────────────────────────────────────────────

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params({k: str(v)[:256] for k, v in params.items()})

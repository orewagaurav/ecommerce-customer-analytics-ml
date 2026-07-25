"""Prediction history.

Every scored request is appended to a parquet log so the dashboard can show what
the system has actually been asked and what it answered — the audit trail a
production model needs and a notebook never has.

Writes are buffered and flushed in batches: a parquet rewrite per request would
put file IO on the latency path. Failures here are logged and swallowed, since
losing an audit row must never turn a successful prediction into a 500.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

HISTORY_COLUMNS = [
    "timestamp",
    "customer_id",
    "predicted_clv",
    "churn_probability",
    "cluster_label",
    "recommended_action",
    "model_version",
    "latency_ms",
]


class PredictionHistoryService:
    """Append-only store of scored predictions."""

    def __init__(self, path: Path, flush_every: int = 5, enabled: bool = True) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._enabled = enabled
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        customer_id: int,
        predicted_clv: float,
        churn_probability: float,
        cluster_label: str,
        recommended_action: str,
        model_version: str,
        latency_ms: float,
    ) -> None:
        """Buffer one prediction, flushing when the batch is full."""
        if not self._enabled:
            return

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "customer_id": int(customer_id),
            "predicted_clv": float(predicted_clv),
            "churn_probability": float(churn_probability),
            "cluster_label": str(cluster_label),
            "recommended_action": str(recommended_action),
            "model_version": str(model_version),
            "latency_ms": float(latency_ms),
        }

        with self._lock:
            self._buffer.append(entry)
            should_flush = len(self._buffer) >= self._flush_every

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Persist buffered rows. Never raises onto the request path."""
        with self._lock:
            if not self._buffer:
                return
            pending, self._buffer = self._buffer, []

        try:
            frame = pd.DataFrame(pending, columns=HISTORY_COLUMNS)
            self._path.parent.mkdir(parents=True, exist_ok=True)

            if self._path.exists():
                frame = pd.concat([pd.read_parquet(self._path), frame], ignore_index=True)

            frame.to_parquet(self._path, index=False)
            logger.debug("Prediction history flushed", extra={"rows": len(pending)})
        except Exception:
            logger.exception("Failed to persist prediction history", extra={"rows": len(pending)})

    def read(
        self,
        limit: int | None = None,
        customer_id: int | None = None,
        min_churn_probability: float | None = None,
    ) -> pd.DataFrame:
        """Return recorded predictions, most recent first."""
        self.flush()

        if not self._path.exists():
            return pd.DataFrame(columns=HISTORY_COLUMNS)

        frame = pd.read_parquet(self._path)

        if customer_id is not None:
            frame = frame[frame["customer_id"] == int(customer_id)]
        if min_churn_probability is not None:
            frame = frame[frame["churn_probability"] >= min_churn_probability]

        frame = frame.sort_values("timestamp", ascending=False)
        return frame.head(limit) if limit else frame

    def summary(self) -> dict[str, Any]:
        """Aggregate stats for the /metrics endpoint."""
        frame = self.read()
        if frame.empty:
            return {"total_predictions": 0, "avg_latency_ms": None, "avg_churn_probability": None}

        return {
            "total_predictions": int(len(frame)),
            "avg_latency_ms": round(float(frame["latency_ms"].mean()), 2),
            "p95_latency_ms": round(float(frame["latency_ms"].quantile(0.95)), 2),
            "avg_churn_probability": round(float(frame["churn_probability"].mean()), 4),
            "unique_customers": int(frame["customer_id"].nunique()),
        }

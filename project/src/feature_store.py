"""Offline feature generation and lazy-loading feature store.

Inference used to call `build_customer_aggregates` over ~800k transaction rows
on every request, then discard all but one customer. That made latency a
function of dataset size. Features are now materialised once, offline, into a
parquet table keyed by CustomerID; a request becomes a dictionary lookup.

The store is loaded lazily on first access and cached for the process lifetime,
so API workers pay the (small) load cost once at first request rather than at
import time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.feature_engineering import build_customer_aggregates
from src.logging_config import get_logger

logger = get_logger(__name__)

# Persisted alongside the table so a stale store can be detected rather than
# silently served.
METADATA_COLUMNS = ["built_at", "source_rows", "customer_count"]


class FeatureStoreError(RuntimeError):
    """Raised when the feature store is missing or unusable."""


class CustomerNotFoundError(KeyError):
    """Raised when a customer has no row in the feature store."""

    def __init__(self, customer_id: int) -> None:
        super().__init__(customer_id)
        self.customer_id = customer_id

    def __str__(self) -> str:
        return f"Customer {self.customer_id} not found in the feature store"


@dataclass(frozen=True)
class FeatureStoreStats:
    """Summary of a materialised store, surfaced through /model-info."""

    built_at: str
    source_rows: int
    customer_count: int
    path: str


def build_feature_store(processed_csv: Path, output_path: Path) -> FeatureStoreStats:
    """Materialise customer-level features to parquet.

    Runs offline (training time or a build step), never on the request path.
    """
    logger.info("Building feature store", extra={"source": str(processed_csv)})

    transactions = pd.read_csv(processed_csv)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")

    features = build_customer_aggregates(transactions)
    features["CustomerID"] = features["CustomerID"].astype("int64")

    built_at = datetime.now(timezone.utc).isoformat()
    features.attrs["built_at"] = built_at

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    stats = FeatureStoreStats(
        built_at=built_at,
        source_rows=int(len(transactions)),
        customer_count=int(len(features)),
        path=str(output_path),
    )

    # Sidecar metadata: parquet attrs do not survive a round trip reliably.
    pd.DataFrame([{
        "built_at": stats.built_at,
        "source_rows": stats.source_rows,
        "customer_count": stats.customer_count,
    }]).to_parquet(output_path.with_suffix(".meta.parquet"), index=False)

    logger.info(
        "Feature store built",
        extra={"customers": stats.customer_count, "source_rows": stats.source_rows},
    )
    return stats


class FeatureStore:
    """Lazy, cached, read-only view over the materialised feature table."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._frame: pd.DataFrame | None = None
        self._index: dict[int, int] | None = None
        self._stats: FeatureStoreStats | None = None
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def _load(self) -> None:
        """Read the table once. Double-checked under a lock for thread safety."""
        if self._frame is not None:
            return

        with self._lock:
            if self._frame is not None:
                return

            if not self._path.exists():
                raise FeatureStoreError(
                    f"Feature store not found at {self._path}. "
                    "Run: python project/src/build_feature_store.py"
                )

            frame = pd.read_parquet(self._path)
            # Positional index gives O(1) lookup without pandas index overhead.
            self._index = {int(cid): pos for pos, cid in enumerate(frame["CustomerID"])}
            self._frame = frame
            self._stats = self._read_stats(frame)

            logger.info(
                "Feature store loaded",
                extra={"customers": len(frame), "path": str(self._path)},
            )

    def _read_stats(self, frame: pd.DataFrame) -> FeatureStoreStats:
        meta_path = self._path.with_suffix(".meta.parquet")
        built_at, source_rows = "unknown", 0
        if meta_path.exists():
            meta = pd.read_parquet(meta_path).iloc[0]
            built_at = str(meta["built_at"])
            source_rows = int(meta["source_rows"])

        return FeatureStoreStats(
            built_at=built_at,
            source_rows=source_rows,
            customer_count=int(len(frame)),
            path=str(self._path),
        )

    def get_customer(self, customer_id: int) -> pd.Series:
        """Return one customer's feature row. O(1) after first load."""
        self._load()
        assert self._frame is not None and self._index is not None

        position = self._index.get(int(customer_id))
        if position is None:
            raise CustomerNotFoundError(customer_id)

        return self._frame.iloc[position]

    def customer_ids(self) -> list[int]:
        self._load()
        assert self._index is not None
        return list(self._index.keys())

    def frame(self) -> pd.DataFrame:
        """Full table, for dashboard aggregate views."""
        self._load()
        assert self._frame is not None
        return self._frame

    def stats(self) -> FeatureStoreStats:
        self._load()
        assert self._stats is not None
        return self._stats

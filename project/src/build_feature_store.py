"""Offline build step: materialise the feature store and the model registry.

Run after training. Keeps every expensive aggregation off the request path.

    python project/src/build_feature_store.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib

from src.config import get_settings
from src.analytics_store import build_analytics_store
from src.feature_store import build_feature_store
from src.logging_config import configure_logging, get_logger
from src.registry import ModelRecord, ModelRegistry, build_version, dataset_fingerprint

logger = get_logger(__name__)


def _round_metrics(metrics: dict) -> dict[str, float]:
    """Flatten and round the winning model's metrics for the registry."""
    return {key: round(float(value), 6) for key, value in metrics.items()}


def register_models(models_dir: Path, registry_path: Path, data_path: Path) -> None:
    """Write the registry manifest from the trained artifacts."""
    report_path = models_dir / "training_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"training_report.json not found in {models_dir}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    version = build_version()
    trained_at = datetime.fromtimestamp(
        report_path.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    dataset_version = dataset_fingerprint(data_path)

    clv_artifacts = joblib.load(models_dir / "clv_model_artifacts.joblib")
    churn_artifacts = joblib.load(models_dir / "churn_model_artifacts.joblib")
    rfm_artifacts = joblib.load(models_dir / "rfm_kmeans_artifacts.joblib")

    clv_best = report["clv"]["best_model"]
    churn_best = report["churn"]["best_model"]

    records = [
        ModelRecord(
            name="clv_regressor",
            version=version,
            algorithm=clv_best,
            task="clv",
            trained_at=trained_at,
            dataset_version=dataset_version,
            feature_list=list(clv_artifacts["feature_columns"]),
            metrics=_round_metrics(report["clv"]["metrics"][clv_best]),
            status="production",
            notes="Selected on Spearman rank correlation; RMSE is outlier-dominated.",
        ),
        ModelRecord(
            name="churn_classifier",
            version=version,
            algorithm=churn_best,
            task="churn",
            trained_at=trained_at,
            dataset_version=dataset_version,
            feature_list=list(churn_artifacts["feature_columns"]),
            metrics=_round_metrics(report["churn"]["metrics"][churn_best]),
            status="production",
            notes="PredictedCLV is generated out-of-fold to avoid label leakage.",
        ),
        ModelRecord(
            name="rfm_segmenter",
            version=version,
            algorithm="KMeans",
            task="segmentation",
            trained_at=trained_at,
            dataset_version=dataset_version,
            feature_list=["Recency", "Frequency", "Monetary"],
            metrics={"n_clusters": float(report["rfm"]["optimal_k"])},
            status="production",
            notes=f"Segments: {', '.join(sorted(set(rfm_artifacts['label_map'].values())))}",
        ),
    ]

    ModelRegistry(registry_path).register(records)
    logger.info("Models registered", extra={"version": version, "count": len(records)})


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, log_format="console")

    parser = argparse.ArgumentParser(description="Build feature store and model registry")
    parser.add_argument("--data_path", type=Path, default=settings.data_path)
    parser.add_argument("--models_dir", type=Path, default=settings.model_path)
    parser.add_argument("--feature_store_path", type=Path, default=settings.feature_store_path)
    parser.add_argument("--registry_path", type=Path, default=settings.registry_path)
    parser.add_argument("--skip_registry", action="store_true")
    args = parser.parse_args()

    stats = build_feature_store(args.data_path, args.feature_store_path)
    print(
        f"Feature store: {stats.customer_count:,} customers "
        f"from {stats.source_rows:,} rows -> {stats.path}"
    )

    aggregates = build_analytics_store(args.data_path, args.feature_store_path.parent)
    print(f"Analytics aggregates: {aggregates.directory}")

    if not args.skip_registry:
        register_models(args.models_dir, args.registry_path, args.data_path)
        print(f"Model registry: {args.registry_path}")


if __name__ == "__main__":
    main()

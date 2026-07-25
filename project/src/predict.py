"""Command-line scoring.

A thin wrapper over `PredictionService` so the CLI, the API and the dashboard
all resolve to one implementation. This module deliberately contains no scoring
logic of its own: it previously held a second copy that read the raw CSV and
rebuilt customer aggregates per call, which was both slow and a source of drift.

    python project/src/predict.py --customer_id 12748
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import Settings, get_settings
from src.feature_store import FeatureStore
from src.registry import ModelRegistry
from src.services.prediction_service import PredictionService


def build_service(settings: Settings | None = None) -> PredictionService:
    """Construct a scoring service from configuration.

    The same object graph the API builds, minus the HTTP layer.
    """
    settings = settings or get_settings()
    return PredictionService(
        feature_store=FeatureStore(settings.feature_store_path),
        models_dir=settings.model_path,
        registry=ModelRegistry(settings.registry_path),
    )


def predict_customer(
    customer_id: int,
    feature_store_path: Path | None = None,
    models_dir: Path | None = None,
    include_explanations: bool = True,
) -> dict:
    """Score one customer and return the dashboard-shaped dictionary.

    Raises:
        CustomerNotFoundError: no feature row for this customer.
        FeatureStoreError: the feature store has not been built.
    """
    settings = get_settings()
    if feature_store_path is not None or models_dir is not None:
        settings = settings.model_copy(
            update={
                "feature_store_path": feature_store_path or settings.feature_store_path,
                "model_path": models_dir or settings.model_path,
            }
        )

    service = build_service(settings)
    return service.predict(customer_id, include_explanations=include_explanations).to_dict()


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Score a customer from the command line")
    parser.add_argument("--customer_id", type=int, required=True, help="Customer ID to score")
    parser.add_argument(
        "--feature_store_path", type=Path, default=settings.feature_store_path,
        help="Path to features.parquet",
    )
    parser.add_argument(
        "--models_dir", type=Path, default=settings.model_path,
        help="Directory with trained artifacts",
    )
    parser.add_argument(
        "--no_explanations", action="store_true",
        help="Skip SHAP for a faster, lighter result",
    )
    args = parser.parse_args()

    result = predict_customer(
        customer_id=args.customer_id,
        feature_store_path=args.feature_store_path,
        models_dir=args.models_dir,
        include_explanations=not args.no_explanations,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

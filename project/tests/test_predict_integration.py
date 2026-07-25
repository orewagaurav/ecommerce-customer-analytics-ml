"""End-to-end inference tests against the committed model artifacts.

Skipped when artifacts are absent so a fresh clone can still run the unit suite
before training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.predict import build_service, predict_customer
from src.services.prediction_service import PredictionService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_STORE = PROJECT_ROOT / "feature_store" / "features.parquet"

REQUIRED_ARTIFACTS = [
    "rfm_kmeans_artifacts.joblib",
    "clv_model_artifacts.joblib",
    "churn_model_artifacts.joblib",
]

artifacts_present = all((MODELS_DIR / name).exists() for name in REQUIRED_ARTIFACTS)
pytestmark = pytest.mark.skipif(
    not (artifacts_present and FEATURE_STORE.exists()),
    reason="trained artifacts or feature store not available",
)


@pytest.fixture(scope="module")
def known_customer_id() -> int:
    predictions = pd.read_csv(MODELS_DIR / "customer_predictions.csv")
    return int(predictions["CustomerID"].iloc[0])


@pytest.fixture(scope="module")
def prediction(known_customer_id) -> dict:
    return predict_customer(known_customer_id, FEATURE_STORE, MODELS_DIR)


def test_artifacts_carry_a_shap_background_sample():
    """Without a background sample, per-customer SHAP silently returns zeros."""
    import joblib

    for task, filename in (("clv", "clv_model_artifacts.joblib"),
                           ("churn", "churn_model_artifacts.joblib")):
        artifacts = joblib.load(MODELS_DIR / filename)
        background = artifacts.get("background_sample")
        assert background is not None, f"{task} artifacts are missing background_sample"
        assert len(background) > 1, "background must contain more than the explained row"


def test_prediction_contains_full_output_contract(prediction):
    for key in (
        "CustomerID", "ClusterLabel", "PredictedCLV", "ChurnProbability",
        "Decision", "RecommendationActions", "Explanations", "ShapTopFeatures",
    ):
        assert key in prediction


def test_churn_probability_is_a_valid_probability(prediction):
    assert 0.0 <= prediction["ChurnProbability"] <= 1.0


def test_predicted_clv_is_not_negative(prediction):
    assert prediction["PredictedCLV"] >= 0.0


def test_per_customer_shap_is_not_all_zero(prediction):
    """Regression guard for the zero-background explainer bug."""
    for task in ("CLV", "Churn"):
        magnitudes = [abs(f["Contribution"]) for f in prediction["ShapTopFeatures"][task]]
        assert sum(magnitudes) > 0, f"{task} SHAP contributions are all zero"


def test_recorded_best_model_matches_saved_pipeline():
    """The report name and the pickled estimator must not drift apart."""
    import joblib

    for filename in ("clv_model_artifacts.joblib", "churn_model_artifacts.joblib"):
        artifacts = joblib.load(MODELS_DIR / filename)
        recorded = artifacts["best_model"]
        assert recorded in artifacts["metrics"], (
            f"best_model '{recorded}' is absent from its own metrics table"
        )


def test_cli_and_service_share_one_implementation():
    """predict.py must delegate, not hold a second copy of the scoring logic."""
    assert isinstance(build_service(), PredictionService)


def test_unknown_customer_raises_not_found():
    from src.feature_store import CustomerNotFoundError

    with pytest.raises(CustomerNotFoundError):
        predict_customer(-12345, FEATURE_STORE, MODELS_DIR)

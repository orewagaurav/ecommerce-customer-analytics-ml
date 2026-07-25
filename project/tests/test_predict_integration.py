"""End-to-end inference tests against the committed model artifacts.

Skipped when artifacts are absent so a fresh clone can still run the unit suite
before training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.predict import load_artifacts, predict_customer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "processed_online_retail_II.csv"

REQUIRED_ARTIFACTS = [
    "rfm_kmeans_artifacts.joblib",
    "clv_model_artifacts.joblib",
    "churn_model_artifacts.joblib",
]

artifacts_present = all((MODELS_DIR / name).exists() for name in REQUIRED_ARTIFACTS)
pytestmark = pytest.mark.skipif(
    not (artifacts_present and DATA_PATH.exists()),
    reason="trained artifacts or processed dataset not available",
)


@pytest.fixture(scope="module")
def known_customer_id() -> int:
    predictions = pd.read_csv(MODELS_DIR / "customer_predictions.csv")
    return int(predictions["CustomerID"].iloc[0])


@pytest.fixture(scope="module")
def prediction(known_customer_id) -> dict:
    return predict_customer(known_customer_id, DATA_PATH, MODELS_DIR)


def test_artifacts_carry_a_shap_background_sample():
    """Without a background sample, per-customer SHAP silently returns zeros."""
    artifacts = load_artifacts(MODELS_DIR)

    for task in ("clv", "churn"):
        background = artifacts[task].get("background_sample")
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
    artifacts = load_artifacts(MODELS_DIR)

    for task in ("clv", "churn"):
        recorded = artifacts[task]["best_model"]
        assert recorded in artifacts[task]["metrics"], (
            f"{task} best_model '{recorded}' is absent from its own metrics table"
        )


def test_unknown_customer_raises_value_error():
    with pytest.raises(ValueError, match="not found"):
        predict_customer(-12345, DATA_PATH, MODELS_DIR)

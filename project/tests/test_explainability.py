"""Regression tests for SHAP explanations.

The dashboard shipped for a while with every per-customer SHAP contribution
equal to zero: a single-row explain call used that same row as its background,
so each feature was compared against itself. The panel still rendered, and no
accuracy metric could have caught it. These tests pin the behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.explainability import (
    _churn_risk_band,
    _human_explanation,
    _unwrap_estimator,
    explain_churn_prediction,
    explain_clv_prediction,
)

NUMERIC = ["Recency", "Frequency", "Monetary"]
CATEGORICAL = ["ClusterLabel"]


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
        ]
    )


@pytest.fixture
def training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame(
        {
            "Recency": rng.integers(1, 365, n).astype(float),
            "Frequency": rng.integers(1, 20, n).astype(float),
            "Monetary": rng.uniform(10, 5000, n),
            "ClusterLabel": rng.choice(["Champions", "At Risk"], n),
        }
    )


@pytest.fixture
def churn_pipeline(training_frame):
    y = (training_frame["Recency"] > 180).astype(int)
    pipe = Pipeline([("preprocessor", _preprocessor()), ("model", RandomForestClassifier(
        n_estimators=25, random_state=0
    ))])
    pipe.fit(training_frame, y)
    return pipe


@pytest.fixture
def clv_pipeline(training_frame):
    y = training_frame["Monetary"] * 0.5 + training_frame["Frequency"] * 10
    pipe = Pipeline([("preprocessor", _preprocessor()), ("model", RandomForestRegressor(
        n_estimators=25, random_state=0
    ))])
    pipe.fit(training_frame, y)
    return pipe


def test_single_row_churn_shap_is_nonzero_with_background(churn_pipeline, training_frame):
    """The bug: one row explained against itself yields all-zero contributions."""
    single = training_frame.head(1)

    result = explain_churn_prediction(
        churn_pipeline, single, top_n=3, background=training_frame
    )
    contributions = [abs(f["Contribution"]) for f in result["top_features"]]

    assert sum(contributions) > 0, (
        "all per-customer SHAP contributions are zero - the explainer is using "
        "the explained row as its own background"
    )


def test_single_row_clv_shap_is_nonzero_with_background(clv_pipeline, training_frame):
    single = training_frame.head(1)

    result = explain_clv_prediction(clv_pipeline, single, top_n=3, background=training_frame)
    contributions = [abs(f["Contribution"]) for f in result["top_features"]]

    assert sum(contributions) > 0


def test_top_features_are_ordered_by_absolute_contribution(churn_pipeline, training_frame):
    result = explain_churn_prediction(
        churn_pipeline, training_frame.head(1), top_n=3, background=training_frame
    )
    magnitudes = [f["AbsContribution"] for f in result["top_features"]]

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_direction_matches_contribution_sign(churn_pipeline, training_frame):
    result = explain_churn_prediction(
        churn_pipeline, training_frame.head(1), top_n=3, background=training_frame
    )

    for feature in result["top_features"]:
        expected = "increase" if feature["Contribution"] > 0 else "decrease"
        assert feature["Direction"] == expected


def test_global_importance_covers_every_transformed_feature(clv_pipeline, training_frame):
    result = explain_clv_prediction(clv_pipeline, training_frame, top_n=3)
    importance = result["feature_importance"]

    expected = clv_pipeline.named_steps["preprocessor"].get_feature_names_out()
    assert set(importance["Feature"]) == set(expected)
    assert (importance["Importance"] >= 0).all()


def test_unwrap_estimator_reaches_inner_regressor():
    """TransformedTargetRegressor / two-stage wrappers must not reach SHAP raw."""
    class Wrapper:
        def __init__(self, inner):
            self.regressor_ = inner

    leaf = object()
    assert _unwrap_estimator(Wrapper(Wrapper(leaf))) is leaf
    assert _unwrap_estimator(leaf) is leaf


# --- explanation wording ----------------------------------------------------

@pytest.mark.parametrize(
    "probability,expected",
    [(0.05, "Low"), (0.39, "Low"), (0.4, "Moderate"), (0.69, "Moderate"), (0.7, "High"), (0.95, "High")],
)
def test_churn_risk_band_matches_probability(probability, expected):
    assert _churn_risk_band(probability) == expected


def test_low_risk_customer_is_not_described_as_high_risk():
    """The wording was hardcoded to "High churn risk" regardless of the number,
    so a 7% customer was labelled high risk beside SHAP values all pointing down."""
    rows = [
        {"Feature": "Recency", "Contribution": -0.11, "AbsContribution": 0.11, "Direction": "decrease"},
        {"Feature": "PredictedCLV", "Contribution": -0.08, "AbsContribution": 0.08, "Direction": "decrease"},
    ]
    sentence = _human_explanation(rows, task_name="churn", prediction=0.075)

    assert sentence.startswith("Low churn risk")
    assert "High churn risk" not in sentence


def test_high_risk_customer_is_described_as_high_risk():
    rows = [
        {"Feature": "Recency", "Contribution": 0.4, "AbsContribution": 0.4, "Direction": "increase"},
    ]
    sentence = _human_explanation(rows, task_name="churn", prediction=0.88)

    assert sentence.startswith("High churn risk")


def test_churn_driver_wording_follows_contribution_direction():
    """A feature pushing risk down must not be described as pushing it up."""
    downward = _human_explanation(
        [{"Feature": "Frequency", "Contribution": -0.3, "AbsContribution": 0.3, "Direction": "decrease"}],
        task_name="churn", prediction=0.1,
    )
    upward = _human_explanation(
        [{"Feature": "Frequency", "Contribution": 0.3, "AbsContribution": 0.3, "Direction": "increase"}],
        task_name="churn", prediction=0.9,
    )

    assert "regular ordering" in downward
    assert "low order frequency" in upward


def test_clv_wording_reflects_predicted_value():
    rows = [{"Feature": "Monetary", "Contribution": 2.0, "AbsContribution": 2.0, "Direction": "increase"}]

    assert _human_explanation(rows, task_name="clv", prediction=9000.0).startswith("High")
    assert _human_explanation(rows, task_name="clv", prediction=10.0).startswith("Low")

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

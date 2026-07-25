"""Guards on out-of-fold CLV predictions.

The churn model consumes PredictedCLV as a feature. The CLV model is trained to
predict FutureRevenue over the same window the churn label is derived from, and
`FutureRevenue > 0` is exactly `ChurnLabel == 0`. Scoring the CLV training rows
in-sample therefore hands the churn model a memorised copy of its own label:
measured on the real dataset, PredictedCLV alone scored churn at AUC 0.96 while
a history-only model reached 0.80.

Out-of-fold prediction is what keeps that feature honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.train_models import _out_of_fold_predictions


@pytest.fixture
def memorisable_data():
    """Features carry no signal, so any correlation with y means memorisation."""
    rng = np.random.default_rng(11)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.normal(size=n))
    return X, y


def _pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=60, random_state=0)),
    ])


def test_in_sample_predictions_memorise_pure_noise(memorisable_data):
    """Establishes the baseline the out-of-fold test is measured against."""
    X, y = memorisable_data
    model = _pipeline().fit(X, y)

    in_sample_corr = np.corrcoef(model.predict(X), y)[0, 1]
    assert in_sample_corr > 0.5, (
        "expected the model to memorise noise in-sample; if it does not, this "
        "test can no longer detect the leak it was written for"
    )


def test_out_of_fold_predictions_do_not_memorise_noise(memorisable_data):
    """Each row must be scored by a model that never saw it."""
    X, y = memorisable_data
    out_of_fold = _out_of_fold_predictions(_pipeline(), X, y, n_splits=5)

    corr = abs(np.corrcoef(out_of_fold, y)[0, 1])
    assert corr < 0.25, (
        f"out-of-fold predictions correlate {corr:.3f} with a pure-noise target, "
        "so rows are being scored by a model that saw them"
    )


def test_out_of_fold_returns_one_prediction_per_row(memorisable_data):
    X, y = memorisable_data
    assert len(_out_of_fold_predictions(_pipeline(), X, y, n_splits=5)) == len(X)


def test_out_of_fold_still_captures_real_signal():
    """The guard must not destroy genuine predictive information."""
    rng = np.random.default_rng(3)
    n = 300
    signal = rng.normal(size=n)
    X = pd.DataFrame({"f0": signal, "f1": rng.normal(size=n)})
    y = pd.Series(signal * 3 + rng.normal(scale=0.3, size=n))

    out_of_fold = _out_of_fold_predictions(_pipeline(), X, y, n_splits=5)
    assert np.corrcoef(out_of_fold, y)[0, 1] > 0.8


def test_out_of_fold_predictions_are_never_negative_for_revenue():
    """CLV is a currency amount; the churn model must not receive negatives."""
    rng = np.random.default_rng(5)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.clip(rng.normal(500, 300, n), 0, None))

    out_of_fold = _out_of_fold_predictions(_pipeline(), X, y, n_splits=5, clip_negative=True)
    assert (out_of_fold >= 0).all()

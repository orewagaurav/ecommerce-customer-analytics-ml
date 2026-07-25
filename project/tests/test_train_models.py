"""Tests for the metrics and the zero-inflated CLV estimator.

These cover the pieces that decide which model ships, so a silent change here
would alter model selection without any test noticing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.train_models import (
    TwoStageCLVRegressor,
    _rmse,
    _spearman,
    _top_decile_lift,
)


def test_rmse_is_zero_for_perfect_predictions():
    y = pd.Series([1.0, 2.0, 3.0])
    assert _rmse(y, np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_rmse_matches_hand_computation():
    y = pd.Series([0.0, 0.0])
    assert _rmse(y, np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def test_spearman_is_one_for_monotonic_predictions():
    """Rank correlation ignores scale, unlike R2."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _spearman(y, np.array([10.0, 20.0, 30.0, 40.0])) == pytest.approx(1.0)


def test_spearman_is_negative_when_ranking_is_inverted():
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _spearman(y, np.array([4.0, 3.0, 2.0, 1.0])) == pytest.approx(-1.0)


def test_spearman_is_unaffected_by_a_single_extreme_outlier():
    """The property that motivated switching selection away from RMSE."""
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    baseline = _spearman(y, pred)
    y_with_whale = pd.Series([1.0, 2.0, 3.0, 4.0, 500_000.0])
    assert _spearman(y_with_whale, pred) == pytest.approx(baseline)


def test_top_decile_lift_rewards_correct_ranking():
    """Perfect ranking concentrates all revenue in the top decile."""
    y = pd.Series([0.0] * 90 + [100.0] * 10)
    perfect = np.arange(100, dtype=float)

    # Top 10 predicted are exactly the 10 spenders; their mean is 100 vs overall 10.
    assert _top_decile_lift(y, perfect) == pytest.approx(10.0)


def test_top_decile_lift_is_about_one_for_random_ranking():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.uniform(0, 100, 1000))
    lift = _top_decile_lift(y, rng.uniform(0, 1, 1000))

    assert 0.7 < lift < 1.3


def test_top_decile_lift_handles_all_zero_revenue():
    y = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(_top_decile_lift(y, np.array([1.0, 2.0, 3.0])))


@pytest.fixture
def zero_inflated_data():
    """Half the rows spend nothing; spenders scale with the feature."""
    rng = np.random.default_rng(7)
    n = 400
    feature = rng.uniform(0, 10, n)
    buys = feature > 5
    y = np.where(buys, feature * 100 + rng.normal(0, 10, n), 0.0)
    return np.column_stack([feature, rng.normal(size=n)]), np.clip(y, 0, None)


def test_two_stage_predicts_near_zero_for_non_buyers(zero_inflated_data):
    X, y = zero_inflated_data
    model = TwoStageCLVRegressor(
        classifier=RandomForestClassifier(n_estimators=40, random_state=0),
        regressor=RandomForestRegressor(n_estimators=40, random_state=0),
    ).fit(X, y)

    non_buyers = model.predict(X)[y == 0]
    buyers = model.predict(X)[y > 0]

    assert non_buyers.mean() < buyers.mean() / 5, (
        "the classifier stage should suppress predictions for non-buyers"
    )


def test_two_stage_never_predicts_negative_revenue(zero_inflated_data):
    X, y = zero_inflated_data
    model = TwoStageCLVRegressor(
        classifier=RandomForestClassifier(n_estimators=40, random_state=0),
        regressor=RandomForestRegressor(n_estimators=40, random_state=0),
    ).fit(X, y)

    assert (model.predict(X) >= 0).all()


def test_two_stage_regressor_trains_only_on_buyers(zero_inflated_data):
    """The spend stage must not be diluted by the zero rows."""
    X, y = zero_inflated_data
    model = TwoStageCLVRegressor(
        classifier=RandomForestClassifier(n_estimators=10, random_state=0),
        regressor=RandomForestRegressor(n_estimators=10, random_state=0),
    ).fit(X, y)

    # Fitted in log space on buyers only, so inverting gives buyer-scale spend.
    predicted_spend = np.expm1(model.regressor_.predict(X))
    assert predicted_spend.min() > 0


def test_two_stage_beats_naive_ranking_on_zero_inflated_target(zero_inflated_data):
    X, y = zero_inflated_data
    model = TwoStageCLVRegressor(
        classifier=RandomForestClassifier(n_estimators=40, random_state=0),
        regressor=RandomForestRegressor(n_estimators=40, random_state=0),
    ).fit(X, y)

    assert _spearman(pd.Series(y), model.predict(X)) > 0.7

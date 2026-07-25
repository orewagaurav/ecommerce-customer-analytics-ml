"""Guards on the temporal split that makes the churn task honest.

The churn label is "did this customer purchase in the window after the cutoff".
If any feature is computed from that same window, the model reads the answer off
its own inputs and reports an accuracy it will never reproduce in production.
These tests pin that boundary.
"""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import build_churn_dataset, build_clv_dataset
from tests.conftest import CUTOFF_DATE


def test_churn_features_exclude_post_cutoff_spend(
    processed_transactions, clv_predictions, cluster_labels
):
    """Customer 1 spends 100 before the cutoff and 5000 after; only 100 may show."""
    churn_df, _ = build_churn_dataset(
        transactions=processed_transactions,
        clv_predictions=clv_predictions,
        cluster_labels=cluster_labels,
        threshold_days=90,
    )

    customer_1 = churn_df.loc[churn_df["CustomerID"] == 1].iloc[0]

    assert customer_1["Monetary"] == 100.0, (
        "Monetary includes post-cutoff revenue - the churn label window has "
        f"leaked into the features (got {customer_1['Monetary']}, expected 100.0)"
    )
    assert customer_1["Frequency"] == 1, (
        "Frequency counts the post-cutoff invoice; features must use history only"
    )


def test_churn_label_matches_future_window(
    processed_transactions, clv_predictions, cluster_labels
):
    """Customers 1-6 purchase after the cutoff (label 0); 7-12 do not (label 1)."""
    churn_df, _ = build_churn_dataset(
        transactions=processed_transactions,
        clv_predictions=clv_predictions,
        cluster_labels=cluster_labels,
        threshold_days=90,
    )
    labels = churn_df.set_index("CustomerID")["ChurnLabel"]

    for cid in range(1, 7):
        assert labels[cid] == 0, f"customer {cid} purchased after cutoff, must not be churned"
    for cid in range(7, 13):
        assert labels[cid] == 1, f"customer {cid} had no future purchase, must be churned"


def test_churn_recency_measured_against_cutoff_not_dataset_end(
    processed_transactions, clv_predictions, cluster_labels
):
    """Recency must be relative to the history window, never the global max date.

    Recency is a model feature, so anchoring it to the dataset end would encode
    how far each customer sits from the label window.
    """
    churn_df, _ = build_churn_dataset(
        transactions=processed_transactions,
        clv_predictions=clv_predictions,
        cluster_labels=cluster_labels,
        threshold_days=90,
    )

    customer_1 = churn_df.loc[churn_df["CustomerID"] == 1].iloc[0]

    # Customer 1's last in-history purchase is 2010-03-01. Measured from the
    # history snapshot the value is ~150 days; measured from the dataset end
    # (2011-01-01) it would be ~306.
    days_from_cutoff = (CUTOFF_DATE - pd.Timestamp("2010-03-01")).days
    assert customer_1["Recency"] <= days_from_cutoff + 2, (
        f"Recency {customer_1['Recency']} looks anchored to the dataset end "
        f"rather than the cutoff (~{days_from_cutoff} expected)"
    )


def test_clv_target_uses_only_future_window(processed_transactions):
    """FutureRevenue for customer 1 is the 5000 post-cutoff sale, not 5100."""
    clv_df = build_clv_dataset(processed_transactions, horizon_days=90)
    customer_1 = clv_df.loc[clv_df["CustomerID"] == 1].iloc[0]

    assert customer_1["FutureRevenue"] == 5000.0
    assert customer_1["Monetary"] == 100.0, "CLV features must not include the target window"


def test_churned_customers_get_zero_future_revenue(processed_transactions):
    """Customers absent from the future window score 0, not NaN."""
    clv_df = build_clv_dataset(processed_transactions, horizon_days=90)
    churned = clv_df.loc[clv_df["CustomerID"].between(7, 12), "FutureRevenue"]

    assert churned.notna().all(), "missing future revenue must be filled, not NaN"
    assert (churned == 0.0).all()

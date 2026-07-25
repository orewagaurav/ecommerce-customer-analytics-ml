"""Tests for customer-level feature construction and segment labelling."""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    build_customer_aggregates,
    choose_k_from_elbow,
    compute_rfm,
    make_recommendation_actions,
    map_cluster_labels,
)


def test_rfm_computed_from_hand_checked_values(processed_transactions):
    """Customer 2 has 3 invoices totalling 50*2 + 25*2 + 40*2 = 230."""
    rfm = compute_rfm(processed_transactions).set_index("CustomerID")

    assert rfm.loc[2, "Frequency"] == 3
    assert rfm.loc[2, "Monetary"] == pytest.approx(230.0)


def test_frequency_counts_invoices_not_line_items():
    """Two line items on one invoice are a single purchase occasion."""
    transactions = pd.DataFrame(
        [
            {"CustomerID": 1, "Invoice": "X1", "InvoiceDate": pd.Timestamp("2010-05-01"),
             "TotalAmount": 10.0, "Country": "United Kingdom"},
            {"CustomerID": 1, "Invoice": "X1", "InvoiceDate": pd.Timestamp("2010-05-01"),
             "TotalAmount": 15.0, "Country": "United Kingdom"},
        ]
    )
    rfm = compute_rfm(transactions).set_index("CustomerID")

    assert rfm.loc[1, "Frequency"] == 1
    assert rfm.loc[1, "Monetary"] == pytest.approx(25.0)


def test_recency_is_zero_days_for_most_recent_customer(processed_transactions):
    """Snapshot is max date + 1, so the latest purchaser has Recency 1."""
    rfm = compute_rfm(processed_transactions)
    assert rfm["Recency"].min() == 1.0


def test_average_basket_size_is_mean_invoice_total(processed_transactions):
    """Customer 2: invoices of 100, 50 and 80 -> mean 76.67."""
    aggregates = build_customer_aggregates(processed_transactions).set_index("CustomerID")
    assert aggregates.loc[2, "AverageBasketSize"] == pytest.approx((100 + 50 + 80) / 3)


def test_purchase_frequency_never_divides_by_zero():
    """A single-day customer has zero tenure; the rate must stay finite."""
    transactions = pd.DataFrame(
        [
            {"CustomerID": 9, "Invoice": "Y1", "InvoiceDate": pd.Timestamp("2010-05-01"),
             "TotalAmount": 10.0, "Country": "France"},
        ]
    )
    aggregates = build_customer_aggregates(transactions).set_index("CustomerID")

    value = aggregates.loc[9, "PurchaseFrequency"]
    assert pd.notna(value)
    assert value == pytest.approx(1.0)


def test_choose_k_from_elbow_finds_the_bend():
    """Inertia drops steeply to K=3 then flattens."""
    inertia = {2: 900.0, 3: 300.0, 4: 280.0, 5: 270.0, 6: 265.0}
    assert choose_k_from_elbow(inertia) == 3


def test_choose_k_handles_too_few_points():
    assert choose_k_from_elbow({2: 100.0}) == 2


def test_cluster_labels_rank_best_segment_as_champions():
    """Lowest recency with highest frequency and spend is the top segment."""
    clustered = pd.DataFrame(
        {
            "Cluster": [0, 0, 1, 1, 2, 2, 3, 3],
            "Recency": [5, 7, 40, 45, 200, 210, 350, 360],
            "Frequency": [30, 28, 12, 11, 3, 2, 1, 1],
            "Monetary": [9000, 8500, 3000, 2800, 400, 380, 60, 50],
        }
    )
    labels = map_cluster_labels(clustered)

    assert labels[0] == "Champions"
    assert labels[3] == "Lost Customers"


def test_cluster_labels_degrade_gracefully_below_four_clusters():
    clustered = pd.DataFrame(
        {
            "Cluster": [0, 1, 2],
            "Recency": [5, 100, 300],
            "Frequency": [20, 5, 1],
            "Monetary": [5000, 800, 50],
        }
    )
    labels = map_cluster_labels(clustered)

    assert len(labels) == 3
    assert labels[0] == "Champions"


@pytest.mark.parametrize(
    "cluster,clv,churn,expected",
    [
        ("Champions", 100.0, 0.9, "Offer Discount"),
        ("Champions", 9000.0, 0.1, "Mark as VIP"),
        ("At Risk", 100.0, 0.1, "Send Retention Campaign"),
        ("Loyal Customers", 100.0, 0.1, "Maintain Engagement"),
    ],
)
def test_recommendation_actions_cover_each_rule(cluster, clv, churn, expected):
    actions = make_recommendation_actions(
        cluster_label=cluster,
        predicted_clv=clv,
        churn_probability=churn,
        clv_high_threshold=5000.0,
    )
    assert expected in actions


def test_recommendation_actions_never_returns_empty():
    actions = make_recommendation_actions(
        cluster_label="Unknown", predicted_clv=0.0, churn_probability=0.0, clv_high_threshold=1.0
    )
    assert actions

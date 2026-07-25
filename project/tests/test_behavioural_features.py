"""Tests for the richer behavioural features used by the CLV model.

Each expected value is computed by hand from the fixture so these tests assert
intended behaviour rather than whatever the implementation happens to return.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import build_customer_aggregates


def _txn(customer: int, invoice: str, date: str, amount: float, stock: str = "SKU-A",
         quantity: int = 1) -> dict:
    return {
        "CustomerID": customer,
        "Invoice": invoice,
        "StockCode": stock,
        "Quantity": quantity,
        "InvoiceDate": pd.Timestamp(date),
        "TotalAmount": amount,
        "Country": "United Kingdom",
    }


@pytest.fixture
def spaced_purchases() -> pd.DataFrame:
    """One customer buying on day 0, 10 and 40; another buying once.

    Gaps are 10 and 30 days -> mean 20, and the second customer has no gap.
    """
    return pd.DataFrame(
        [
            _txn(1, "I1", "2010-01-01", 100.0, stock="SKU-A"),
            _txn(1, "I2", "2010-01-11", 200.0, stock="SKU-B"),
            _txn(1, "I3", "2010-02-10", 300.0, stock="SKU-A"),
            _txn(2, "I4", "2010-01-05", 50.0, stock="SKU-C"),
        ]
    )


def test_tenure_is_days_between_first_purchase_and_snapshot(spaced_purchases):
    """Snapshot is max date + 1 = 2010-02-11; customer 1 first bought 2010-01-01."""
    aggregates = build_customer_aggregates(spaced_purchases).set_index("CustomerID")
    assert aggregates.loc[1, "Tenure"] == pytest.approx(41.0)


def test_average_inter_purchase_days_uses_gaps_between_invoices(spaced_purchases):
    """Gaps of 10 and 30 days average to 20."""
    aggregates = build_customer_aggregates(spaced_purchases).set_index("CustomerID")
    assert aggregates.loc[1, "AvgInterPurchaseDays"] == pytest.approx(20.0)


def test_single_purchase_customer_gets_tenure_as_inter_purchase_fallback(spaced_purchases):
    """One purchase means no observed gap; falling back to tenure keeps the
    feature finite and monotonic in how long we have watched the customer."""
    aggregates = build_customer_aggregates(spaced_purchases).set_index("CustomerID")

    value = aggregates.loc[2, "AvgInterPurchaseDays"]
    assert pd.notna(value)
    assert value == pytest.approx(aggregates.loc[2, "Tenure"])


def test_distinct_products_counts_unique_stock_codes(spaced_purchases):
    aggregates = build_customer_aggregates(spaced_purchases).set_index("CustomerID")

    assert aggregates.loc[1, "DistinctProducts"] == 2  # SKU-A, SKU-B
    assert aggregates.loc[2, "DistinctProducts"] == 1


def test_average_items_per_invoice_counts_quantity(spaced_purchases):
    frame = pd.DataFrame(
        [
            _txn(9, "J1", "2010-01-01", 10.0, quantity=2),
            _txn(9, "J1", "2010-01-01", 10.0, quantity=4),
            _txn(9, "J2", "2010-01-05", 10.0, quantity=6),
        ]
    )
    aggregates = build_customer_aggregates(frame).set_index("CustomerID")

    # Invoice J1 has 6 items, J2 has 6 -> mean 6.
    assert aggregates.loc[9, "AvgItemsPerInvoice"] == pytest.approx(6.0)


def test_recent_revenue_share_isolates_the_last_90_days():
    """Customer spends 100 long ago and 300 recently -> recent share 0.75."""
    frame = pd.DataFrame(
        [
            _txn(5, "K1", "2010-01-01", 100.0),
            _txn(5, "K2", "2010-06-15", 300.0),
        ]
    )
    aggregates = build_customer_aggregates(frame).set_index("CustomerID")
    assert aggregates.loc[5, "RecentRevenueShare"] == pytest.approx(0.75)


def test_recent_revenue_share_is_one_when_all_spend_is_recent(spaced_purchases):
    """Every purchase inside the trailing window means the whole share is recent."""
    aggregates = build_customer_aggregates(spaced_purchases).set_index("CustomerID")
    assert aggregates.loc[1, "RecentRevenueShare"] == pytest.approx(1.0)


def test_recent_revenue_share_is_bounded_between_zero_and_one(spaced_purchases):
    aggregates = build_customer_aggregates(spaced_purchases)
    shares = aggregates["RecentRevenueShare"]

    assert (shares >= 0).all()
    assert (shares <= 1).all()


def test_new_features_are_finite_for_every_customer(spaced_purchases):
    """NaN or inf here would silently become the imputer's median downstream."""
    import numpy as np

    aggregates = build_customer_aggregates(spaced_purchases)
    new_columns = [
        "Tenure",
        "AvgInterPurchaseDays",
        "DistinctProducts",
        "AvgItemsPerInvoice",
        "RecentRevenueShare",
    ]

    for column in new_columns:
        values = aggregates[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{column} contains NaN or inf"


def test_existing_features_are_preserved(spaced_purchases):
    """The new columns must not displace what CLV and churn already consume."""
    aggregates = build_customer_aggregates(spaced_purchases)

    for column in [
        "CustomerID", "Country", "Recency", "Frequency", "Monetary",
        "AverageBasketSize", "PurchaseFrequency",
    ]:
        assert column in aggregates.columns

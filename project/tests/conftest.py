"""Shared fixtures.

Tests run on small synthetic frames rather than the real 80 MB dataset so the
suite stays fast and deterministic, and so expected values can be computed by
hand instead of copied from whatever the code currently produces.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MAX_DATE = pd.Timestamp("2011-01-01")
# build_churn_dataset / build_clv_dataset cut history at max_date - horizon.
CUTOFF_DATE = MAX_DATE - pd.Timedelta(days=90)


def _txn(customer: int, invoice: str, date: str, amount: float, country: str = "United Kingdom") -> dict:
    """One processed-format transaction row with TotalAmount already derived."""
    return {
        "Invoice": invoice,
        "StockCode": "TESTSKU",
        "Description": "Test item",
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp(date),
        "Price": amount,
        "Customer ID": float(customer),
        "Country": country,
        "CustomerID": customer,
        "TotalAmount": amount,
    }


@pytest.fixture
def processed_transactions() -> pd.DataFrame:
    """Synthetic processed transactions with a known pre/post-cutoff split.

    Customers 1-6 purchase on both sides of the cutoff (retained, ChurnLabel 0).
    Customers 7-12 purchase only before it (churned, ChurnLabel 1).
    Customer 1 additionally has a large *post*-cutoff purchase, which any
    correctly-built feature must exclude.
    """
    rows: list[dict] = []

    # Customer 1: 100 before the cutoff, 5000 after it.
    rows.append(_txn(1, "INV-1a", "2010-03-01", 100.0))
    rows.append(_txn(1, "INV-1b", "2010-12-01", 5000.0))

    # Customers 2-6: retained, modest spend on both sides.
    for cid in range(2, 7):
        rows.append(_txn(cid, f"INV-{cid}a", "2010-02-15", 50.0 * cid))
        rows.append(_txn(cid, f"INV-{cid}b", "2010-04-20", 25.0 * cid))
        rows.append(_txn(cid, f"INV-{cid}c", "2010-11-15", 40.0 * cid))

    # Customers 7-12: churned, pre-cutoff activity only.
    for cid in range(7, 13):
        rows.append(_txn(cid, f"INV-{cid}a", "2010-01-10", 30.0 * cid, country="France"))
        rows.append(_txn(cid, f"INV-{cid}b", "2010-05-05", 20.0 * cid, country="France"))

    return pd.DataFrame(rows)


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Raw-format frame containing the dirty rows preprocessing must remove."""
    return pd.DataFrame(
        [
            # Valid.
            {
                "Invoice": "A1", "StockCode": "S1", "Description": "Good",
                "Quantity": 2, "InvoiceDate": "2010-03-01 10:00:00", "Price": 5.0,
                "Customer ID": 1001.0, "Country": "United Kingdom",
            },
            # Missing Customer ID -> dropped.
            {
                "Invoice": "A2", "StockCode": "S2", "Description": "No customer",
                "Quantity": 1, "InvoiceDate": "2010-03-02 10:00:00", "Price": 3.0,
                "Customer ID": None, "Country": "United Kingdom",
            },
            # Return (negative quantity) -> dropped.
            {
                "Invoice": "A3", "StockCode": "S3", "Description": "Return",
                "Quantity": -5, "InvoiceDate": "2010-03-03 10:00:00", "Price": 4.0,
                "Customer ID": 1002.0, "Country": "France",
            },
            # Zero price -> dropped.
            {
                "Invoice": "A4", "StockCode": "S4", "Description": "Freebie",
                "Quantity": 1, "InvoiceDate": "2010-03-04 10:00:00", "Price": 0.0,
                "Customer ID": 1003.0, "Country": "France",
            },
            # Unparseable date -> dropped.
            {
                "Invoice": "A5", "StockCode": "S5", "Description": "Bad date",
                "Quantity": 1, "InvoiceDate": "not-a-date", "Price": 9.0,
                "Customer ID": 1004.0, "Country": "Germany",
            },
            # Valid, missing Description -> kept and filled.
            {
                "Invoice": "A6", "StockCode": "S6", "Description": None,
                "Quantity": 3, "InvoiceDate": "2010-03-06 10:00:00", "Price": 2.0,
                "Customer ID": 1005.0, "Country": "Germany",
            },
        ]
    )


@pytest.fixture
def cluster_labels() -> pd.DataFrame:
    """Fixed cluster assignments so churn tests avoid nondeterministic KMeans."""
    return pd.DataFrame(
        {
            "CustomerID": list(range(1, 13)),
            "ClusterLabel": ["Champions"] * 6 + ["At Risk"] * 6,
        }
    )


@pytest.fixture
def clv_predictions() -> pd.DataFrame:
    """Placeholder CLV scores for the churn feature join."""
    return pd.DataFrame(
        {
            "CustomerID": list(range(1, 13)),
            "PredictedCLV": [1000.0 * i for i in range(1, 13)],
        }
    )

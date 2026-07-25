"""Tests for the raw-data cleaning stage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_preprocessing import RAW_COLUMNS, clean_transactions, load_raw_data


def test_drops_rows_missing_customer_id(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    assert "A2" not in set(cleaned["Invoice"])


def test_drops_returns_and_zero_priced_rows(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    invoices = set(cleaned["Invoice"])

    assert "A3" not in invoices, "negative quantity (return) must be removed"
    assert "A4" not in invoices, "zero-price row must be removed"
    assert (cleaned["Quantity"] > 0).all()
    assert (cleaned["Price"] > 0).all()


def test_drops_unparseable_dates(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    assert "A5" not in set(cleaned["Invoice"])
    assert cleaned["InvoiceDate"].notna().all()


def test_keeps_valid_rows_and_fills_missing_description(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    invoices = set(cleaned["Invoice"])

    assert invoices == {"A1", "A6"}
    assert cleaned.loc[cleaned["Invoice"] == "A6", "Description"].iloc[0] == "Unknown"


def test_derives_total_amount_and_integer_customer_id(raw_transactions):
    cleaned = clean_transactions(raw_transactions)
    row = cleaned.loc[cleaned["Invoice"] == "A1"].iloc[0]

    assert row["TotalAmount"] == pytest.approx(10.0)  # 2 x 5.0
    assert row["CustomerID"] == 1001
    assert pd.api.types.is_integer_dtype(cleaned["CustomerID"])


def test_load_raw_data_rejects_missing_columns(tmp_path):
    incomplete = tmp_path / "bad.csv"
    pd.DataFrame({"Invoice": ["A1"], "Quantity": [1]}).to_csv(incomplete, index=False)

    with pytest.raises(ValueError, match="Missing expected columns"):
        load_raw_data(incomplete)


def test_load_raw_data_accepts_full_schema(tmp_path, raw_transactions):
    path = tmp_path / "ok.csv"
    raw_transactions.to_csv(path, index=False)

    loaded = load_raw_data(path)
    assert all(column in loaded.columns for column in RAW_COLUMNS)

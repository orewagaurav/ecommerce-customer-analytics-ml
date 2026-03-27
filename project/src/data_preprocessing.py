"""Data preprocessing pipeline for the Online Retail II dataset.

This module standardizes raw transactional data and prepares a cleaned dataset
used by downstream EDA, clustering, regression, and classification modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd


RAW_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Load raw Online Retail II data with tolerant CSV parsing.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        Raw dataframe with expected schema.
    """
    df = pd.read_csv(file_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    missing_cols = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize transaction records.

    Steps:
    - Keep required columns only.
    - Parse types.
    - Remove records with missing Customer ID.
    - Remove non-positive Quantity/Price rows (returns/corrections).
    - Add derived `TotalAmount` and normalized `CustomerID`.
    """
    data = df[RAW_COLUMNS].copy()

    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data["Customer ID"] = pd.to_numeric(data["Customer ID"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["Price"] = pd.to_numeric(data["Price"], errors="coerce")

    data = data.dropna(subset=["InvoiceDate", "Customer ID", "Quantity", "Price"])

    # Filter out returns and invalid purchases for customer analytics features.
    data = data[(data["Quantity"] > 0) & (data["Price"] > 0)]

    data["CustomerID"] = data["Customer ID"].astype(int)
    data["TotalAmount"] = data["Quantity"] * data["Price"]
    data["Invoice"] = data["Invoice"].astype(str)
    data["StockCode"] = data["StockCode"].astype(str)
    data["Description"] = data["Description"].fillna("Unknown").astype(str)
    data["Country"] = data["Country"].fillna("Unknown").astype(str)

    return data


def preprocessing_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Generate a compact summary table for logging/reporting."""
    summary = {
        "raw_rows": [len(raw_df)],
        "clean_rows": [len(clean_df)],
        "rows_removed": [len(raw_df) - len(clean_df)],
        "unique_customers": [clean_df["CustomerID"].nunique()],
        "date_min": [clean_df["InvoiceDate"].min()],
        "date_max": [clean_df["InvoiceDate"].max()],
        "total_revenue": [clean_df["TotalAmount"].sum()],
    }
    return pd.DataFrame(summary)


def run_preprocessing(input_csv: Path, output_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete preprocessing pipeline and save the cleaned dataset."""
    raw_df = load_raw_data(input_csv)
    clean_df = clean_transactions(raw_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)

    return clean_df, preprocessing_summary(raw_df, clean_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Online Retail II dataset")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path("project/data/online_retail_II.csv"),
        help="Path to the raw Online Retail II CSV",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("project/data/processed_online_retail_II.csv"),
        help="Path to save cleaned data",
    )
    args = parser.parse_args()

    _, summary_df = run_preprocessing(args.input_csv, args.output_csv)
    print("Preprocessing completed.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

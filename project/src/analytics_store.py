"""Precomputed dashboard aggregates.

The dashboard needs revenue trends, country splits and KPIs — none of which
require transaction-level rows at render time. Materialising them offline means
the container ships without the 80 MB CSV, and every page renders from tables
measured in kilobytes.

Each aggregate is a separate parquet under one directory so a page loads only
what it draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

MONTHLY_REVENUE = "monthly_revenue.parquet"
COUNTRY_REVENUE = "country_revenue.parquet"
TOP_PRODUCTS = "top_products.parquet"
CUSTOMER_MONTHLY = "customer_monthly.parquet"
CUSTOMER_PRODUCTS = "customer_products.parquet"
KPIS = "kpis.parquet"


@dataclass(frozen=True)
class AnalyticsAggregates:
    """Handles to the materialised aggregate tables."""

    directory: Path

    def _read(self, name: str) -> pd.DataFrame:
        path = self.directory / name
        if not path.exists():
            raise FileNotFoundError(
                f"Aggregate {name} not found. Run project/src/build_feature_store.py"
            )
        return pd.read_parquet(path)

    def exists(self) -> bool:
        return (self.directory / KPIS).exists()

    def monthly_revenue(self) -> pd.DataFrame:
        return self._read(MONTHLY_REVENUE)

    def country_revenue(self) -> pd.DataFrame:
        return self._read(COUNTRY_REVENUE)

    def top_products(self) -> pd.DataFrame:
        return self._read(TOP_PRODUCTS)

    def customer_monthly(self, customer_id: int | None = None) -> pd.DataFrame:
        frame = self._read(CUSTOMER_MONTHLY)
        if customer_id is not None:
            frame = frame[frame["CustomerID"] == int(customer_id)]
        return frame

    def customer_products(self, customer_id: int, limit: int = 10) -> pd.DataFrame:
        """A customer's most-purchased products, by revenue."""
        frame = self._read(CUSTOMER_PRODUCTS)
        frame = frame[frame["CustomerID"] == int(customer_id)]
        return frame.sort_values("Revenue", ascending=False).head(limit)

    def kpis(self) -> dict:
        return self._read(KPIS).iloc[0].to_dict()


def build_analytics_store(processed_csv: Path, output_dir: Path) -> AnalyticsAggregates:
    """Materialise every dashboard aggregate from the processed transactions."""
    logger.info("Building analytics aggregates", extra={"source": str(processed_csv)})
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = pd.read_csv(processed_csv)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")
    transactions["InvoiceMonth"] = transactions["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    # Revenue over time.
    monthly = (
        transactions.groupby("InvoiceMonth", as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Orders=("Invoice", "nunique"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("InvoiceMonth")
    )
    monthly.to_parquet(output_dir / MONTHLY_REVENUE, index=False)

    # Geographic split.
    country = (
        transactions.groupby("Country", as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Customers=("CustomerID", "nunique"),
            Orders=("Invoice", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
    )
    country.to_parquet(output_dir / COUNTRY_REVENUE, index=False)

    # Product leaderboard, capped: the tail is not rendered anywhere.
    products = (
        transactions.groupby(["StockCode", "Description"], as_index=False)
        .agg(
            Revenue=("TotalAmount", "sum"),
            Units=("Quantity", "sum"),
            Customers=("CustomerID", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(200)
    )
    products.to_parquet(output_dir / TOP_PRODUCTS, index=False)

    # Per-customer monthly spend, for the Customer 360 trend chart.
    customer_monthly = (
        transactions.groupby(["CustomerID", "InvoiceMonth"], as_index=False)
        .agg(Revenue=("TotalAmount", "sum"), Orders=("Invoice", "nunique"))
        .sort_values(["CustomerID", "InvoiceMonth"])
    )
    customer_monthly.to_parquet(output_dir / CUSTOMER_MONTHLY, index=False)

    # Per-customer product mix, capped at the top 15 per customer so the table
    # stays small enough to load per page view.
    customer_products = (
        transactions.groupby(["CustomerID", "StockCode", "Description"], as_index=False)
        .agg(Revenue=("TotalAmount", "sum"), Units=("Quantity", "sum"))
        .sort_values(["CustomerID", "Revenue"], ascending=[True, False])
        .groupby("CustomerID", as_index=False)
        .head(15)
    )
    customer_products.to_parquet(output_dir / CUSTOMER_PRODUCTS, index=False)

    kpis = pd.DataFrame([{
        "total_revenue": float(transactions["TotalAmount"].sum()),
        "unique_customers": int(transactions["CustomerID"].nunique()),
        "total_transactions": int(len(transactions)),
        "total_orders": int(transactions["Invoice"].nunique()),
        "avg_order_value": float(
            transactions["TotalAmount"].sum() / transactions["Invoice"].nunique()
        ),
        "date_min": str(transactions["InvoiceDate"].min()),
        "date_max": str(transactions["InvoiceDate"].max()),
        "countries": int(transactions["Country"].nunique()),
    }])
    kpis.to_parquet(output_dir / KPIS, index=False)

    logger.info(
        "Analytics aggregates built",
        extra={"months": len(monthly), "countries": len(country)},
    )
    return AnalyticsAggregates(output_dir)


def top_customers(feature_frame: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """Highest-spending customers, derived from the feature store."""
    columns = ["CustomerID", "Monetary", "Frequency", "Recency", "Country"]
    return feature_frame[columns].sort_values("Monetary", ascending=False).head(limit)

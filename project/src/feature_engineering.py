"""Feature engineering utilities for customer analytics models.

This module centralizes transformations used by clustering, CLV regression,
and churn classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]
DEFAULT_SEGMENTS = ["Champions", "Loyal Customers", "At Risk", "Lost Customers"]


@dataclass
class ClusteringArtifacts:
    scaler: StandardScaler
    model: KMeans
    label_map: Dict[int, str]
    inertia_by_k: Dict[int, float]


def build_customer_aggregates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Build base customer-level aggregates used across tasks."""
    work = transactions.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")

    invoice_totals = (
        work.groupby(["CustomerID", "Invoice"], as_index=False)
        .agg(InvoiceTotal=("TotalAmount", "sum"), InvoiceDate=("InvoiceDate", "max"))
    )

    snapshot_date = work["InvoiceDate"].max() + pd.Timedelta(days=1)

    base = work.groupby("CustomerID", as_index=False).agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        FirstPurchaseDate=("InvoiceDate", "min"),
        Monetary=("TotalAmount", "sum"),
        Frequency=("Invoice", "nunique"),
        Country=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    )

    basket = invoice_totals.groupby("CustomerID", as_index=False).agg(
        AverageBasketSize=("InvoiceTotal", "mean")
    )

    base = base.merge(basket, on="CustomerID", how="left")
    base["Recency"] = (snapshot_date - base["LastPurchaseDate"]).dt.days.astype(float)

    customer_age_days = (base["LastPurchaseDate"] - base["FirstPurchaseDate"]).dt.days.clip(lower=1)
    base["PurchaseFrequency"] = base["Frequency"] / customer_age_days

    return base[[
        "CustomerID",
        "Country",
        "Recency",
        "Frequency",
        "Monetary",
        "AverageBasketSize",
        "PurchaseFrequency",
        "FirstPurchaseDate",
        "LastPurchaseDate",
    ]]


def compute_rfm(transactions: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Compute RFM features at customer level."""
    work = transactions.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")

    if snapshot_date is None:
        snapshot_date = work["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = work.groupby("CustomerID", as_index=False).agg(
        LastPurchaseDate=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchaseDate"]).dt.days.astype(float)

    return rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]


def scale_rfm(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Scale RFM columns for clustering."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_df[RFM_COLUMNS])
    return scaled, scaler


def elbow_inertia(scaled_rfm: np.ndarray, min_k: int = 2, max_k: int = 10) -> Dict[int, float]:
    """Compute inertia values for elbow method."""
    inertias: Dict[int, float] = {}
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        model.fit(scaled_rfm)
        inertias[k] = float(model.inertia_)
    return inertias


def choose_k_from_elbow(inertia_by_k: Dict[int, float]) -> int:
    """Select K using the largest absolute second derivative heuristic."""
    ks = sorted(inertia_by_k.keys())
    if len(ks) < 3:
        return ks[0]

    values = np.array([inertia_by_k[k] for k in ks], dtype=float)
    second_diff = np.diff(values, n=2)
    if len(second_diff) == 0:
        return ks[0]

    # second_diff index i maps to ks[i+1]
    elbow_index = int(np.argmax(np.abs(second_diff))) + 1
    return ks[elbow_index]


def _cluster_business_score(summary: pd.DataFrame) -> pd.Series:
    """Higher score means higher-value customer segment."""
    recency_score = (-summary["Recency"]).rank(method="average")
    frequency_score = summary["Frequency"].rank(method="average")
    monetary_score = summary["Monetary"].rank(method="average")
    return recency_score + frequency_score + monetary_score


def map_cluster_labels(rfm_with_cluster: pd.DataFrame) -> Dict[int, str]:
    """Map numeric cluster IDs to business-friendly labels."""
    summary = rfm_with_cluster.groupby("Cluster", as_index=False).agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    )
    summary["Score"] = _cluster_business_score(summary)
    summary = summary.sort_values("Score", ascending=False).reset_index(drop=True)

    ordered_clusters = summary["Cluster"].tolist()
    label_map: Dict[int, str] = {}

    if len(ordered_clusters) >= 4:
        label_map[ordered_clusters[0]] = "Champions"
        label_map[ordered_clusters[1]] = "Loyal Customers"
        label_map[ordered_clusters[-2]] = "At Risk"
        label_map[ordered_clusters[-1]] = "Lost Customers"

        for cluster_id in ordered_clusters:
            if cluster_id not in label_map:
                label_map[cluster_id] = "Regular Customers"
    else:
        # Fallback if K < 4.
        fallback_labels = DEFAULT_SEGMENTS[: len(ordered_clusters)]
        for cluster_id, segment in zip(ordered_clusters, fallback_labels):
            label_map[cluster_id] = segment

    return label_map


def run_rfm_clustering(
    transactions: pd.DataFrame,
    min_k: int = 2,
    max_k: int = 10,
) -> Tuple[pd.DataFrame, ClusteringArtifacts]:
    """Train KMeans on RFM and return customer segments with artifacts."""
    rfm = compute_rfm(transactions)
    scaled_rfm, scaler = scale_rfm(rfm)
    inertia_by_k = elbow_inertia(scaled_rfm, min_k=min_k, max_k=max_k)

    k_opt = choose_k_from_elbow(inertia_by_k)
    model = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
    clusters = model.fit_predict(scaled_rfm)

    result = rfm.copy()
    result["Cluster"] = clusters

    label_map = map_cluster_labels(result)
    result["ClusterLabel"] = result["Cluster"].map(label_map)

    artifacts = ClusteringArtifacts(
        scaler=scaler,
        model=model,
        label_map=label_map,
        inertia_by_k=inertia_by_k,
    )
    return result, artifacts


def build_clv_dataset(transactions: pd.DataFrame, horizon_days: int = 90) -> pd.DataFrame:
    """Create CLV regression dataset with next `horizon_days` revenue target."""
    work = transactions.copy()
    work["InvoiceDate"] = pd.to_datetime(work["InvoiceDate"], errors="coerce")

    max_date = work["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=horizon_days)

    history = work[work["InvoiceDate"] <= cutoff_date].copy()
    future = work[(work["InvoiceDate"] > cutoff_date) & (work["InvoiceDate"] <= max_date)].copy()

    history_features = build_customer_aggregates(history)
    future_target = future.groupby("CustomerID", as_index=False).agg(FutureRevenue=("TotalAmount", "sum"))

    dataset = history_features.merge(future_target, on="CustomerID", how="left")
    dataset["FutureRevenue"] = dataset["FutureRevenue"].fillna(0.0)

    return dataset


def build_churn_dataset(
    transactions: pd.DataFrame,
    clv_predictions: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    threshold_days: int = 90,
    dynamic_threshold: bool = False,
) -> Tuple[pd.DataFrame, int]:
    """Create churn classification dataset and label.

    churn = 1 if Recency > threshold.
    If dynamic_threshold is True, threshold is set to 75th percentile recency.
    """
    base = build_customer_aggregates(transactions)

    threshold = threshold_days
    if dynamic_threshold:
        threshold = int(np.percentile(base["Recency"], 75))

    data = base.merge(clv_predictions[["CustomerID", "PredictedCLV"]], on="CustomerID", how="left")
    data = data.merge(cluster_labels[["CustomerID", "ClusterLabel"]], on="CustomerID", how="left")

    data["PredictedCLV"] = data["PredictedCLV"].fillna(data["PredictedCLV"].median())
    data["ClusterLabel"] = data["ClusterLabel"].fillna("Regular Customers")
    data["ChurnLabel"] = (data["Recency"] > threshold).astype(int)

    return data, threshold


def make_recommendation_actions(cluster_label: str, predicted_clv: float, churn_probability: float, clv_high_threshold: float) -> List[str]:
    """Business-rule recommendations from model outputs."""
    actions: List[str] = []

    if churn_probability > 0.7:
        actions.append("Offer Discount")

    if predicted_clv >= clv_high_threshold:
        actions.append("Mark as VIP")

    if cluster_label == "At Risk":
        actions.append("Send Retention Campaign")

    if not actions:
        actions.append("Maintain Engagement")

    return actions

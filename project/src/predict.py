"""Inference and recommendation utilities for customer analytics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import (
    build_customer_aggregates,
    make_recommendation_actions,
)
from src.explainability import explain_churn_prediction, explain_clv_prediction
from src.recommendation_engine import get_recommendation


def load_artifacts(models_dir: Path) -> Dict:
    """Load all trained artifacts from disk."""
    rfm_artifacts = joblib.load(models_dir / "rfm_kmeans_artifacts.joblib")
    clv_artifacts = joblib.load(models_dir / "clv_model_artifacts.joblib")
    churn_artifacts = joblib.load(models_dir / "churn_model_artifacts.joblib")

    clv_importance = pd.read_csv(models_dir / "clv_feature_importance.csv")
    churn_importance = pd.read_csv(models_dir / "churn_feature_importance.csv")

    return {
        "rfm": rfm_artifacts,
        "clv": clv_artifacts,
        "churn": churn_artifacts,
        "clv_importance": clv_importance,
        "churn_importance": churn_importance,
    }


def _build_customer_feature_row(transactions: pd.DataFrame, customer_id: int) -> pd.Series:
    """Create single customer row from processed transactions."""
    features = build_customer_aggregates(transactions)
    row = features[features["CustomerID"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer ID {customer_id} not found in dataset")
    return row.iloc[0]


def _cluster_for_customer(customer_row: pd.Series, rfm_artifacts: Dict) -> str:
    scaler = rfm_artifacts["scaler"]
    model = rfm_artifacts["model"]
    label_map = rfm_artifacts["label_map"]

    rfm_values = pd.DataFrame(
        [[customer_row["Recency"], customer_row["Frequency"], customer_row["Monetary"]]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    cluster_id = int(model.predict(scaler.transform(rfm_values))[0])
    return label_map.get(cluster_id, "Regular Customers")


def _importance_reasons(customer_row: pd.Series, importance_df: pd.DataFrame, max_reasons: int = 3) -> List[str]:
    """Produce lightweight explanation from top importance features and values."""
    reasons: List[str] = []
    top = importance_df.head(max_reasons)

    for _, imp in top.iterrows():
        feat = str(imp["Feature"])
        if "Recency" in feat:
            reasons.append(f"Recency is {customer_row['Recency']:.0f} days")
        elif "Frequency" in feat:
            reasons.append(f"Frequency is {customer_row['Frequency']:.0f} invoices")
        elif "Monetary" in feat:
            reasons.append(f"Monetary spend is {customer_row['Monetary']:.2f}")
        elif "PredictedCLV" in feat:
            reasons.append("Predicted CLV materially influences churn score")
        elif "AverageBasketSize" in feat:
            reasons.append(f"Average basket size is {customer_row['AverageBasketSize']:.2f}")
        elif "PurchaseFrequency" in feat:
            reasons.append(f"Purchase frequency is {customer_row['PurchaseFrequency']:.4f} per day")
        elif "Country" in feat:
            reasons.append(f"Country profile is {customer_row['Country']}")
        elif "ClusterLabel" in feat:
            reasons.append("Cluster segment contributes to churn likelihood")

    # Keep original order while removing repetitive messages from one-hot expansions.
    reasons = list(dict.fromkeys(reasons))

    if not reasons:
        reasons = ["Prediction primarily driven by combined historical behavior"]

    return reasons


def _format_top_shap(top_rows: List[Dict[str, float | str]]) -> List[str]:
    """Convert SHAP top features into compact readable bullet lines."""
    formatted: List[str] = []
    for row in top_rows:
        formatted.append(
            f"{row['Feature']} ({row['Direction']}, SHAP={row['Contribution']:.4f})"
        )
    return formatted


def predict_customer(customer_id: int, processed_csv: Path, models_dir: Path) -> Dict:
    """Predict cluster label, CLV, churn probability, and recommendations."""
    transactions = pd.read_csv(processed_csv)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")

    artifacts = load_artifacts(models_dir)
    row = _build_customer_feature_row(transactions, customer_id)

    cluster_label = _cluster_for_customer(row, artifacts["rfm"])

    clv_model = artifacts["clv"]["model"]
    clv_input = pd.DataFrame([
        {
            "Recency": row["Recency"],
            "Frequency": row["Frequency"],
            "Monetary": row["Monetary"],
            "AverageBasketSize": row["AverageBasketSize"],
            "PurchaseFrequency": row["PurchaseFrequency"],
            "Country": row["Country"],
        }
    ])
    clv_prediction = float(clv_model.predict(clv_input)[0])
    clv_shap = explain_clv_prediction(clv_model, clv_input, top_n=3)

    churn_model = artifacts["churn"]["model"]
    churn_input = pd.DataFrame([
        {
            "Recency": row["Recency"],
            "Frequency": row["Frequency"],
            "Monetary": row["Monetary"],
            "PredictedCLV": clv_prediction,
            "ClusterLabel": cluster_label,
        }
    ])
    churn_probability = float(churn_model.predict_proba(churn_input)[:, 1][0])
    churn_shap = explain_churn_prediction(churn_model, churn_input, top_n=3)

    clv_high_threshold = float(artifacts["clv"].get("high_clv_threshold", 0.0))
    actions = make_recommendation_actions(
        cluster_label=cluster_label,
        predicted_clv=clv_prediction,
        churn_probability=churn_probability,
        clv_high_threshold=clv_high_threshold,
    )
    decision = get_recommendation(
        {
            "PredictedCLV": clv_prediction,
            "ChurnProbability": churn_probability,
            "ClusterLabel": cluster_label,
        }
    )

    clv_reasons = [clv_shap["explanation"]] + _format_top_shap(clv_shap["top_features"])
    churn_reasons = [churn_shap["explanation"]] + _format_top_shap(churn_shap["top_features"])

    return {
        "CustomerID": int(customer_id),
        "ClusterLabel": cluster_label,
        "PredictedCLV": clv_prediction,
        "ChurnProbability": churn_probability,
        "Decision": decision,
        "RecommendationActions": actions,
        "Explanations": {
            "CLV": clv_reasons,
            "Churn": churn_reasons,
        },
        "ShapTopFeatures": {
            "CLV": clv_shap["top_features"],
            "Churn": churn_shap["top_features"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict customer analytics outputs")
    parser.add_argument("--customer_id", type=int, required=True, help="Customer ID to score")
    parser.add_argument(
        "--processed_csv",
        type=Path,
        default=Path("project/data/processed_online_retail_II.csv"),
        help="Processed dataset path",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path("project/models"),
        help="Directory with trained artifacts",
    )
    args = parser.parse_args()

    result = predict_customer(
        customer_id=args.customer_id,
        processed_csv=args.processed_csv,
        models_dir=args.models_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

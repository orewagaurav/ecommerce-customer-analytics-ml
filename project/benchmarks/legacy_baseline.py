"""Reference implementation of the pre-feature-store scoring path.

This is the code inference used to run: read the full processed CSV, rebuild
customer aggregates across every transaction, then throw away all but one row.
It exists **solely** so the "before" column of the benchmark stays reproducible
rather than being a number quoted from memory.

It is not importable from `src/` on purpose. Production has one scoring path
(`PredictionService`); keeping a second copy there would be duplicate logic
waiting to drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.explainability import explain_churn_prediction, explain_clv_prediction
from src.feature_engineering import build_customer_aggregates, make_recommendation_actions
from src.recommendation_engine import get_recommendation


def _load_artifacts(models_dir: Path) -> dict[str, Any]:
    return {
        "rfm": joblib.load(models_dir / "rfm_kmeans_artifacts.joblib"),
        "clv": joblib.load(models_dir / "clv_model_artifacts.joblib"),
        "churn": joblib.load(models_dir / "churn_model_artifacts.joblib"),
    }


def _model_input(
    row: pd.Series, feature_columns: list[str], extras: dict[str, Any] | None = None
) -> pd.DataFrame:
    extras = extras or {}
    values = {
        column: (extras[column] if column in extras else row[column])
        for column in feature_columns
    }
    return pd.DataFrame([values], columns=feature_columns)


def predict_customer_legacy(customer_id: int, processed_csv: Path, models_dir: Path) -> dict:
    """Score one customer the slow way, re-reading and re-aggregating everything."""
    transactions = pd.read_csv(processed_csv)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")

    artifacts = _load_artifacts(models_dir)

    features = build_customer_aggregates(transactions)
    matching = features[features["CustomerID"] == customer_id]
    if matching.empty:
        raise ValueError(f"Customer ID {customer_id} not found in dataset")
    row = matching.iloc[0]

    rfm = artifacts["rfm"]
    rfm_values = pd.DataFrame(
        [[row["Recency"], row["Frequency"], row["Monetary"]]],
        columns=["Recency", "Frequency", "Monetary"],
    )
    cluster_id = int(rfm["model"].predict(rfm["scaler"].transform(rfm_values))[0])
    cluster_label = rfm["label_map"].get(cluster_id, "Regular Customers")

    clv_model = artifacts["clv"]["model"]
    clv_input = _model_input(row, artifacts["clv"]["feature_columns"])
    predicted_clv = float(clv_model.predict(clv_input)[0])
    explain_clv_prediction(
        clv_model, clv_input, top_n=3,
        background=artifacts["clv"].get("background_sample"),
        prediction=predicted_clv,
    )

    churn_model = artifacts["churn"]["model"]
    churn_input = _model_input(
        row,
        artifacts["churn"]["feature_columns"],
        extras={"PredictedCLV": predicted_clv, "ClusterLabel": cluster_label},
    )
    churn_probability = float(churn_model.predict_proba(churn_input)[:, 1][0])
    explain_churn_prediction(
        churn_model, churn_input, top_n=3,
        background=artifacts["churn"].get("background_sample"),
        prediction=churn_probability,
    )

    return {
        "CustomerID": int(customer_id),
        "ClusterLabel": cluster_label,
        "PredictedCLV": predicted_clv,
        "ChurnProbability": churn_probability,
        "Decision": get_recommendation(
            {
                "PredictedCLV": predicted_clv,
                "ChurnProbability": churn_probability,
                "ClusterLabel": cluster_label,
            }
        ),
        "RecommendationActions": make_recommendation_actions(
            cluster_label=cluster_label,
            predicted_clv=predicted_clv,
            churn_probability=churn_probability,
            clv_high_threshold=float(artifacts["clv"].get("high_clv_threshold", 0.0)),
        ),
    }

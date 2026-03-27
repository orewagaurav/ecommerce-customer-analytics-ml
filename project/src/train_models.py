"""Training orchestration for clustering, CLV regression, and churn classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import (
    build_churn_dataset,
    build_clv_dataset,
    make_recommendation_actions,
    run_rfm_clustering,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


RANDOM_STATE = 42


def _save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _train_clv_model(clv_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Dict[str, float]], pd.DataFrame]:
    feature_cols = [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageBasketSize",
        "PurchaseFrequency",
        "Country",
    ]
    target_col = "FutureRevenue"

    X = clv_df[feature_cols].copy()
    y = clv_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    numeric_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "AverageBasketSize",
        "PurchaseFrequency",
    ]
    categorical_features = ["Country"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    candidates: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoostRegressor"] = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        metrics[model_name] = {
            "RMSE": _rmse(y_test, pred),
            "R2": float(r2_score(y_test, pred)),
        }
        fitted_models[model_name] = pipe

    best_name = min(metrics, key=lambda name: metrics[name]["RMSE"])
    best_model = fitted_models[best_name]

    return best_model, metrics, X


def _train_churn_model(churn_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Dict[str, float]], pd.DataFrame]:
    feature_cols = ["Recency", "Frequency", "Monetary", "PredictedCLV", "ClusterLabel"]
    target_col = "ChurnLabel"

    X = churn_df[feature_cols].copy()
    y = churn_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = ["Recency", "Frequency", "Monetary", "PredictedCLV"]
    categorical_features = ["ClusterLabel"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    candidates: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(max_iter=300),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoostClassifier"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        metrics[model_name] = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, proba)),
        }
        fitted_models[model_name] = pipe

    best_name = max(metrics, key=lambda name: metrics[name]["ROC_AUC"])
    best_model = fitted_models[best_name]

    return best_model, metrics, X


def _extract_feature_importance(model: Pipeline, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Extract built-in feature importance from final estimator when available."""
    transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0]) if getattr(estimator.coef_, "ndim", 1) > 1 else np.abs(estimator.coef_)
    else:
        values = np.zeros(len(transformed_names), dtype=float)

    if len(values) != len(transformed_names):
        values = np.resize(values, len(transformed_names))

    out = pd.DataFrame({"Feature": transformed_names, "Importance": values})
    out = out.sort_values("Importance", ascending=False).reset_index(drop=True)
    return out


def _save_elbow_and_cluster_plot(rfm_df: pd.DataFrame, inertia_by_k: Dict[int, float], models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)

    elbow_path = models_dir / "elbow_plot.png"
    plt.figure(figsize=(8, 5))
    ks = sorted(inertia_by_k.keys())
    vals = [inertia_by_k[k] for k in ks]
    plt.plot(ks, vals, marker="o")
    plt.title("Elbow Method for K-Means")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.tight_layout()
    plt.savefig(elbow_path)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    points = pca.fit_transform(rfm_df[["Recency", "Frequency", "Monetary"]])

    cluster_path = models_dir / "rfm_clusters_2d.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=rfm_df["Cluster"], cmap="viridis", s=20)
    plt.title("Customer Clusters (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(cluster_path)
    plt.close()


def train_all_models(processed_csv: Path, models_dir: Path, horizon_days: int, churn_days: int) -> Dict:
    transactions = pd.read_csv(processed_csv)
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")

    # Module 1: RFM clustering
    rfm_segments, clustering_artifacts = run_rfm_clustering(transactions)
    _save_elbow_and_cluster_plot(rfm_segments, clustering_artifacts.inertia_by_k, models_dir)

    joblib.dump(
        {
            "scaler": clustering_artifacts.scaler,
            "model": clustering_artifacts.model,
            "label_map": clustering_artifacts.label_map,
            "inertia_by_k": clustering_artifacts.inertia_by_k,
        },
        models_dir / "rfm_kmeans_artifacts.joblib",
    )
    rfm_segments.to_csv(models_dir / "customer_segments.csv", index=False)

    # Module 2: CLV regression
    clv_df = build_clv_dataset(transactions, horizon_days=horizon_days)
    clv_model, clv_metrics, clv_X = _train_clv_model(clv_df)
    clv_predictions = clv_model.predict(clv_df[["Recency", "Frequency", "Monetary", "AverageBasketSize", "PurchaseFrequency", "Country"]])

    clv_pred_df = clv_df[["CustomerID"]].copy()
    clv_pred_df["PredictedCLV"] = clv_predictions

    clv_feature_importance = _extract_feature_importance(clv_model, clv_X)
    clv_feature_importance.to_csv(models_dir / "clv_feature_importance.csv", index=False)

    best_clv_name = min(clv_metrics, key=lambda m: clv_metrics[m]["RMSE"])
    joblib.dump(
        {
            "model": clv_model,
            "feature_columns": ["Recency", "Frequency", "Monetary", "AverageBasketSize", "PurchaseFrequency", "Country"],
            "best_model": best_clv_name,
            "metrics": clv_metrics,
            "high_clv_threshold": float(np.percentile(clv_pred_df["PredictedCLV"], 75)),
        },
        models_dir / "clv_model_artifacts.joblib",
    )

    # Module 3: Churn classification
    churn_df, used_threshold = build_churn_dataset(
        transactions=transactions,
        clv_predictions=clv_pred_df,
        cluster_labels=rfm_segments[["CustomerID", "ClusterLabel"]],
        threshold_days=churn_days,
        dynamic_threshold=False,
    )

    churn_model, churn_metrics, churn_X = _train_churn_model(churn_df)
    churn_feature_importance = _extract_feature_importance(churn_model, churn_X)
    churn_feature_importance.to_csv(models_dir / "churn_feature_importance.csv", index=False)

    best_churn_name = max(churn_metrics, key=lambda m: churn_metrics[m]["ROC_AUC"])
    joblib.dump(
        {
            "model": churn_model,
            "feature_columns": ["Recency", "Frequency", "Monetary", "PredictedCLV", "ClusterLabel"],
            "best_model": best_churn_name,
            "metrics": churn_metrics,
            "threshold_days": used_threshold,
        },
        models_dir / "churn_model_artifacts.joblib",
    )

    customer_predictions = churn_df[["CustomerID", "Recency", "Frequency", "Monetary", "PredictedCLV", "ClusterLabel"]].copy()
    customer_predictions["ChurnProbability"] = churn_model.predict_proba(
        customer_predictions[["Recency", "Frequency", "Monetary", "PredictedCLV", "ClusterLabel"]]
    )[:, 1]

    high_clv_threshold = float(np.percentile(clv_pred_df["PredictedCLV"], 75))
    customer_predictions["RecommendedActions"] = customer_predictions.apply(
        lambda row: "; ".join(
            make_recommendation_actions(
                cluster_label=row["ClusterLabel"],
                predicted_clv=float(row["PredictedCLV"]),
                churn_probability=float(row["ChurnProbability"]),
                clv_high_threshold=high_clv_threshold,
            )
        ),
        axis=1,
    )
    customer_predictions.to_csv(models_dir / "customer_predictions.csv", index=False)

    report = {
        "rfm": {
            "optimal_k": int(clustering_artifacts.model.n_clusters),
            "label_map": clustering_artifacts.label_map,
        },
        "clv": {
            "best_model": best_clv_name,
            "metrics": clv_metrics,
        },
        "churn": {
            "best_model": best_churn_name,
            "threshold_days": int(used_threshold),
            "metrics": churn_metrics,
        },
    }

    _save_json(report, models_dir / "training_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train customer analytics models")
    parser.add_argument(
        "--processed_csv",
        type=Path,
        default=Path("project/data/processed_online_retail_II.csv"),
        help="Processed transactions CSV path",
    )
    parser.add_argument(
        "--models_dir",
        type=Path,
        default=Path("project/models"),
        help="Directory where model artifacts will be saved",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=90,
        help="Future horizon used for CLV target",
    )
    parser.add_argument(
        "--churn_days",
        type=int,
        default=90,
        help="Recency threshold for churn",
    )
    args = parser.parse_args()

    report = train_all_models(
        processed_csv=args.processed_csv,
        models_dir=args.models_dir,
        horizon_days=args.horizon_days,
        churn_days=args.churn_days,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

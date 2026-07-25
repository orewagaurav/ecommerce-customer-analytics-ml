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
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.feature_engineering import (
    build_churn_dataset,
    build_clv_dataset,
    make_recommendation_actions,
    run_rfm_clustering,
)
from src.explainability import (
    explain_churn_prediction,
    explain_clv_prediction,
    save_feature_importance_plot,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None


RANDOM_STATE = 42

# Forest artifacts are ~70 MB uncompressed and live in git. compress=3 cuts that
# ~4x for a negligible load-time cost, and the models are byte-identical on load.
ARTIFACT_COMPRESSION = 3

# Declared once so the trainers, the saved artifacts and inference cannot drift.
CLV_NUMERIC_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AverageBasketSize",
    "PurchaseFrequency",
    "Tenure",
    "AvgInterPurchaseDays",
    "DistinctProducts",
    "AvgItemsPerInvoice",
    "RecentRevenueShare",
]
CLV_CATEGORICAL_FEATURES = ["Country"]
CLV_FEATURES = CLV_NUMERIC_FEATURES + CLV_CATEGORICAL_FEATURES

CHURN_NUMERIC_FEATURES = CLV_NUMERIC_FEATURES + ["PredictedCLV"]
CHURN_CATEGORICAL_FEATURES = ["ClusterLabel"]
CHURN_FEATURES = CHURN_NUMERIC_FEATURES + CHURN_CATEGORICAL_FEATURES


def _save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _spearman(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Rank correlation between predicted and actual revenue.

    Future revenue is dominated by a few extreme customers, so squared-error
    metrics mostly measure how well the single largest customer was fit. Every
    downstream use of CLV here (VIP flagging, priority ordering, campaign
    targeting) consumes the *ranking*, which rank correlation measures directly
    and outliers cannot distort.
    """
    return float(spearmanr(y_true, y_pred).statistic)


def _top_decile_lift(y_true: pd.Series, y_pred: np.ndarray, decile: float = 0.1) -> float:
    """Mean actual revenue of the top-predicted decile over the overall mean.

    This is the number a marketing team acts on: if we contact the top 10% the
    model flags, how much richer is that group than a random customer?
    """
    y_true = np.asarray(y_true, dtype=float)
    overall_mean = y_true.mean()
    if overall_mean <= 0:
        return float("nan")

    cutoff = max(1, int(len(y_true) * decile))
    top_idx = np.argsort(y_pred)[::-1][:cutoff]
    return float(y_true[top_idx].mean() / overall_mean)


def _out_of_fold_predictions(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    clip_negative: bool = False,
) -> np.ndarray:
    """Predict every row using a model fitted without that row.

    The churn model consumes PredictedCLV as a feature, and the CLV model is
    trained against revenue in the same window the churn label comes from
    (`FutureRevenue > 0` is exactly `ChurnLabel == 0`). Scoring the CLV training
    rows in-sample therefore leaks the churn label through a memorised
    prediction. Out-of-fold scoring keeps the feature usable without it.
    """
    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,
    )
    if clip_negative:
        predictions = np.clip(predictions, 0.0, None)
    return predictions


class TwoStageCLVRegressor(BaseEstimator, RegressorMixin):
    """Zero-inflated CLV model: E[revenue] = P(purchase) x E[spend | purchase].

    Roughly 57% of customers generate no revenue in the forecast window, so a
    single regressor spends most of its capacity predicting zeros and fits the
    spending customers poorly. Splitting the problem lets each stage solve a
    well-posed task: a classifier estimates purchase probability, and a
    regressor fit only on purchasers estimates spend in log space, where the
    heavy revenue tail no longer dominates squared error.
    """

    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        purchased = (y > 0).astype(int)

        self.classifier_ = clone(self.classifier)
        self.classifier_.fit(X, purchased)

        self.regressor_ = clone(self.regressor)
        buyers = purchased == 1
        self.regressor_.fit(X[buyers], np.log1p(y[buyers]))

        return self

    def predict(self, X):
        purchase_prob = self.classifier_.predict_proba(X)[:, 1]
        expected_spend = np.expm1(self.regressor_.predict(X))
        return purchase_prob * np.clip(expected_spend, 0.0, None)


def _clv_preprocessor() -> ColumnTransformer:
    """Build the shared CLV feature transformer (dense output for SHAP)."""
    numeric_features = CLV_NUMERIC_FEATURES
    categorical_features = CLV_CATEGORICAL_FEATURES

    return ColumnTransformer(
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
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            ),
        ]
    )


def _train_clv_model(clv_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Dict[str, float]], str, pd.DataFrame]:
    """Train CLV candidates and return the best pipeline by holdout RMSE.

    All metrics are reported on the original currency scale, so log-target and
    two-stage models stay directly comparable to the raw-target baseline.
    """
    feature_cols = CLV_FEATURES
    target_col = "FutureRevenue"

    X = clv_df[feature_cols].copy()
    y = clv_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    def _log_target(estimator: object) -> TransformedTargetRegressor:
        """Fit the estimator on log1p(revenue) and invert predictions to currency."""
        return TransformedTargetRegressor(
            regressor=estimator, func=np.log1p, inverse_func=np.expm1
        )

    # Baseline kept deliberately: it is the raw-target model the first version
    # shipped, and it anchors the improvement reported in the README.
    candidates: Dict[str, object] = {
        "LinearRegression_raw": LinearRegression(),
        "RandomForestRegressor_log": _log_target(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    if XGBRegressor is not None:
        xgb_reg_params = dict(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        candidates["XGBoostRegressor_log"] = _log_target(XGBRegressor(**xgb_reg_params))
        candidates["TwoStageXGBoost"] = TwoStageCLVRegressor(
            classifier=XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            regressor=XGBRegressor(**xgb_reg_params),
        )

    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Pipeline] = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for model_name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", _clv_preprocessor()),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        # 5-fold CV on the training split guards against a lucky holdout.
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)

        metrics[model_name] = {
            "RMSE": _rmse(y_test, pred),
            "R2": float(r2_score(y_test, pred)),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "SpearmanR": _spearman(y_test, pred),
            "TopDecileLift": _top_decile_lift(y_test, pred),
            "CV_R2_mean": float(np.mean(cv_r2)),
            "CV_R2_std": float(np.std(cv_r2)),
        }
        fitted_models[model_name] = pipe

    # Selected on rank correlation, not RMSE: a single customer contributes ~80%
    # of the holdout sum of squares, so RMSE ranks models mostly by how they
    # happened to fit that one point. Spearman reflects the targeting quality
    # the dashboard and recommendation rules actually depend on.
    best_name = max(metrics, key=lambda name: metrics[name]["SpearmanR"])
    best_model = fitted_models[best_name]

    return best_model, metrics, best_name, X


def _train_churn_model(churn_df: pd.DataFrame) -> Tuple[Pipeline, Dict[str, Dict[str, float]], str, pd.DataFrame]:
    """Train churn candidates and return the best pipeline by ROC-AUC.

    Recency is a legitimate feature here: ChurnLabel is defined by the absence
    of purchases in the window *after* the cutoff, while every feature is built
    from history up to the cutoff only. The earlier exclusion guarded against a
    recency-threshold label definition that this pipeline no longer uses.
    """
    feature_cols = CHURN_FEATURES
    target_col = "ChurnLabel"

    X = churn_df[feature_cols].copy()
    y = churn_df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    numeric_features = CHURN_NUMERIC_FEATURES
    categorical_features = CHURN_CATEGORICAL_FEATURES

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
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for model_name, estimator in candidates.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]

        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)

        metrics[model_name] = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_test, proba)),
            "BrierScore": float(brier_score_loss(y_test, proba)),
            "CV_ROC_AUC_mean": float(np.mean(cv_auc)),
            "CV_ROC_AUC_std": float(np.std(cv_auc)),
        }
        fitted_models[model_name] = pipe

    # Single selection criterion, returned to the caller so the recorded model
    # name can never disagree with the pipeline that was actually saved.
    best_name = max(metrics, key=lambda name: metrics[name]["ROC_AUC"])
    best_model = fitted_models[best_name]

    return best_model, metrics, best_name, X


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
        compress=ARTIFACT_COMPRESSION,
    )
    rfm_segments.to_csv(models_dir / "customer_segments.csv", index=False)

    # Module 2: CLV regression
    clv_df = build_clv_dataset(transactions, horizon_days=horizon_days)
    clv_model, clv_metrics, best_clv_name, clv_X = _train_clv_model(clv_df)
    clv_feature_cols = CLV_FEATURES
    # Out-of-fold, not in-sample: PredictedCLV feeds the churn model, and an
    # in-sample score would carry a memorised copy of the churn label.
    clv_pred_df = clv_df[["CustomerID"]].copy()
    clv_pred_df["PredictedCLV"] = _out_of_fold_predictions(
        clv_model,
        clv_df[clv_feature_cols],
        clv_df["FutureRevenue"],
        clip_negative=True,
    )

    clv_shap = explain_clv_prediction(
        model=clv_model,
        X_sample=clv_df[clv_feature_cols].sample(min(600, len(clv_df)), random_state=RANDOM_STATE),
        top_n=3,
    )
    clv_feature_importance = clv_shap["feature_importance"]
    clv_feature_importance.to_csv(models_dir / "clv_feature_importance.csv", index=False)
    save_feature_importance_plot(
        clv_feature_importance,
        models_dir / "clv_shap_importance.png",
        title="CLV SHAP Feature Importance",
    )

    joblib.dump(
        {
            "model": clv_model,
            "feature_columns": clv_feature_cols,
            "best_model": best_clv_name,
            "metrics": clv_metrics,
            "high_clv_threshold": float(np.percentile(clv_pred_df["PredictedCLV"], 75)),
            # Reference distribution for single-customer SHAP at inference time.
            "background_sample": clv_df[clv_feature_cols].sample(
                min(200, len(clv_df)), random_state=RANDOM_STATE
            ),
        },
        models_dir / "clv_model_artifacts.joblib",
        compress=ARTIFACT_COMPRESSION,
    )

    # Module 3: Churn classification
    churn_df, used_threshold = build_churn_dataset(
        transactions=transactions,
        clv_predictions=clv_pred_df,
        threshold_days=churn_days,
        dynamic_threshold=False,
    )

    churn_model, churn_metrics, best_churn_name, churn_X = _train_churn_model(churn_df)
    churn_feature_cols = CHURN_FEATURES
    churn_shap = explain_churn_prediction(
        model=churn_model,
        X_sample=churn_df[churn_feature_cols].sample(min(600, len(churn_df)), random_state=RANDOM_STATE),
        top_n=3,
    )
    churn_feature_importance = churn_shap["feature_importance"]
    churn_feature_importance.to_csv(models_dir / "churn_feature_importance.csv", index=False)
    save_feature_importance_plot(
        churn_feature_importance,
        models_dir / "churn_shap_importance.png",
        title="Churn SHAP Feature Importance",
    )

    joblib.dump(
        {
            "model": churn_model,
            "feature_columns": churn_feature_cols,
            "best_model": best_churn_name,
            "metrics": churn_metrics,
            "threshold_days": used_threshold,
            # Reference distribution for single-customer SHAP at inference time.
            "background_sample": churn_df[churn_feature_cols].sample(
                min(200, len(churn_df)), random_state=RANDOM_STATE
            ),
        },
        models_dir / "churn_model_artifacts.joblib",
        compress=ARTIFACT_COMPRESSION,
    )

    customer_predictions = churn_df[["CustomerID"] + CHURN_FEATURES].copy()
    customer_predictions["ChurnProbability"] = churn_model.predict_proba(
        customer_predictions[churn_feature_cols]
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

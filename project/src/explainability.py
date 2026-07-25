"""SHAP-based explainability utilities for CLV and churn models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse
from sklearn.pipeline import Pipeline


def _to_dense(matrix: np.ndarray) -> np.ndarray:
    """Convert sparse matrix to dense numpy array."""
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _extract_binary_shap(shap_values: object) -> np.ndarray:
    """Normalize SHAP output into (n_samples, n_features) for binary tasks."""
    if isinstance(shap_values, list):
        # Older APIs sometimes return list[class0, class1].
        if len(shap_values) == 2:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        # (n_samples, n_features, n_classes)
        class_index = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_index]
    return values


def _unwrap_estimator(estimator: object) -> object:
    """Return the underlying fitted estimator SHAP can explain directly.

    CLV models are wrapped: `TransformedTargetRegressor` fits in log space, and
    `TwoStageCLVRegressor` holds a separate classifier and regressor. Both expose
    the fitted inner regressor as `regressor_`. Explaining that inner model gives
    contributions in log-revenue space, which preserves the ranking and sign of
    each feature's effect while keeping a fast tree/linear explainer in play.
    """
    inner = getattr(estimator, "regressor_", None)
    if inner is not None:
        return _unwrap_estimator(inner)
    return estimator


def _safe_explainer(
    estimator: object,
    transformed: np.ndarray,
    task: str,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Compute SHAP values with robust fallback logic.

    `background` must be a reference sample distinct from the rows being
    explained. Scoring a single customer without one makes SHAP explain that row
    against itself, which yields all-zero contributions.
    """
    dense = _to_dense(transformed)
    reference = _to_dense(background) if background is not None else dense

    try:
        explainer = shap.Explainer(estimator, reference)
        explanation = explainer(dense)
        values = explanation.values
    except Exception:
        # KernelExplainer is slower but supports generic estimators.
        fallback_bg = reference[: min(50, len(reference))]
        if task == "classification" and hasattr(estimator, "predict_proba"):
            kernel = shap.KernelExplainer(estimator.predict_proba, fallback_bg)
            values = kernel.shap_values(dense, silent=True)
        else:
            kernel = shap.KernelExplainer(estimator.predict, fallback_bg)
            values = kernel.shap_values(dense, silent=True)

    if task == "classification":
        return _extract_binary_shap(values)
    return np.asarray(values)


def _pretty_feature_name(feature: str) -> str:
    """Convert transformed feature names into dashboard-friendly labels."""
    cleaned = feature.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("ClusterLabel_", "Cluster=")
    cleaned = cleaned.replace("Country_", "Country=")
    return cleaned


def _top_features_for_sample(
    shap_sample: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> List[Dict[str, float | str]]:
    """Return top SHAP contributors for one sample."""
    order = np.argsort(np.abs(shap_sample))[::-1][:top_n]
    top_rows: List[Dict[str, float | str]] = []

    for idx in order:
        contribution = float(shap_sample[idx])
        top_rows.append(
            {
                "Feature": _pretty_feature_name(feature_names[idx]),
                "Contribution": contribution,
                "AbsContribution": float(abs(contribution)),
                "Direction": "increase" if contribution > 0 else "decrease",
            }
        )

    return top_rows


def _churn_risk_band(probability: float) -> str:
    """Bucket a churn probability into the wording used in explanations."""
    if probability >= 0.7:
        return "High"
    if probability >= 0.4:
        return "Moderate"
    return "Low"


def _human_explanation(
    top_rows: List[Dict[str, float | str]],
    task_name: str,
    raw_sample: pd.Series | None = None,
    prediction: float | None = None,
) -> str:
    """Build a concise explanation sentence that agrees with the prediction.

    `prediction` is the model's own output (churn probability, or predicted
    revenue). Without it the wording was fixed at "High churn risk" for every
    customer, so a 7% risk was described as high while the SHAP values beside it
    all pointed downwards.
    """
    lowered = [str(row["Feature"]).lower() for row in top_rows]

    if task_name == "churn":
        band = _churn_risk_band(prediction) if prediction is not None else None

        signals: List[str] = []
        # Name the driver by the direction its contribution actually points.
        for row in top_rows:
            feature = str(row["Feature"]).lower()
            increases = float(row.get("Contribution", 0.0)) > 0
            if "recency" in feature and "recentrevenue" not in feature:
                signals.append("high recency" if increases else "recent activity")
            elif "frequency" in feature:
                signals.append("low order frequency" if increases else "regular ordering")
            elif "monetary" in feature or "predictedclv" in feature:
                signals.append("customer value profile")
            elif "tenure" in feature:
                signals.append("account tenure")

        signals = list(dict.fromkeys(signals))

        if band and signals:
            return f"{band} churn risk, driven by " + " and ".join(signals) + "."
        if band:
            return f"{band} churn risk based on the customer's overall behaviour."
        if signals:
            return "Churn risk driven by " + " and ".join(signals) + "."
        return "Churn prediction is driven by the strongest behavioural features."

    if task_name == "clv":
        level = None
        if prediction is not None:
            level = "High" if prediction >= 1000 else "Moderate" if prediction >= 200 else "Low"

        signals = []
        if any("monetary" in f for f in lowered):
            signals.append("historical spend")
        if any("frequency" in f for f in lowered):
            signals.append("purchase frequency")
        if any("recency" in f and "recentrevenue" not in f for f in lowered):
            signals.append("recency pattern")
        if any("recentrevenueshare" in f for f in lowered):
            signals.append("recent revenue momentum")
        if any("country" in f for f in lowered):
            signals.append("country profile")

        if level and signals:
            return f"{level} predicted value, driven by " + " and ".join(signals) + "."
        if signals:
            return "Predicted CLV is primarily influenced by " + " and ".join(signals) + "."
        return "CLV prediction is driven by the strongest transactional features."

    return "Prediction is driven by the top contributing features."


def _explain_pipeline_prediction(
    model: Pipeline,
    X_sample: pd.DataFrame,
    task: str,
    top_n: int = 3,
    background: pd.DataFrame | None = None,
    prediction: float | None = None,
) -> Dict:
    """Generic SHAP explanation for pipeline model predictions."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = _unwrap_estimator(model.named_steps["model"])

    transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out().tolist()

    transformed_background = (
        preprocessor.transform(background) if background is not None else None
    )
    shap_values = _safe_explainer(
        estimator, transformed, task=task, background=transformed_background
    )
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    sample_values = shap_values[0]
    top_rows = _top_features_for_sample(sample_values, feature_names, top_n=top_n)

    global_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    raw_sample = X_sample.iloc[0] if len(X_sample) > 0 else None
    explanation = _human_explanation(
        top_rows,
        task_name="churn" if task == "classification" else "clv",
        raw_sample=raw_sample,
        prediction=prediction,
    )

    return {
        "top_features": top_rows,
        "explanation": explanation,
        "feature_importance": global_importance,
    }


def explain_clv_prediction(
    model: Pipeline,
    X_sample: pd.DataFrame,
    top_n: int = 3,
    background: pd.DataFrame | None = None,
    prediction: float | None = None,
) -> Dict:
    """Explain CLV model prediction using SHAP values.

    Returns a dictionary containing:
    - top_features: top SHAP contributors for first sample
    - explanation: human-readable text explanation
    - feature_importance: global SHAP importance over provided sample set

    Pass `background` when scoring a single customer, otherwise contributions
    collapse to zero.
    """
    return _explain_pipeline_prediction(
        model=model, X_sample=X_sample, task="regression", top_n=top_n,
        background=background, prediction=prediction,
    )


def explain_churn_prediction(
    model: Pipeline,
    X_sample: pd.DataFrame,
    top_n: int = 3,
    background: pd.DataFrame | None = None,
    prediction: float | None = None,
) -> Dict:
    """Explain churn model prediction using SHAP values.

    Returns a dictionary containing:
    - top_features: top SHAP contributors for first sample
    - explanation: human-readable text explanation
    - feature_importance: global SHAP importance over provided sample set

    Pass `background` when scoring a single customer, otherwise contributions
    collapse to zero.
    """
    return _explain_pipeline_prediction(
        model=model, X_sample=X_sample, task="classification", top_n=top_n,
        background=background, prediction=prediction,
    )


def save_feature_importance_plot(feature_df: pd.DataFrame, plot_path: Path, title: str) -> None:
    """Save a horizontal bar chart for top feature importance values."""
    top = feature_df.head(15).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top["Feature"], top["Importance"], color="#1f77b4")
    plt.title(title)
    plt.xlabel("Mean |SHAP Value|")
    plt.ylabel("Feature")
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

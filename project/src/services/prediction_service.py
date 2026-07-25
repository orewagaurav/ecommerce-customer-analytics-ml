"""Prediction service.

Owns the scoring workflow: features in, segment + CLV + churn + explanation +
recommendation out. Collaborators (feature store, artifacts, registry) are
injected rather than constructed here, so the API can share one warm instance
across requests and tests can substitute fakes without touching disk.

Artifacts are loaded once and held; the previous implementation reloaded three
joblib files and re-read an 80 MB CSV on every single call.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.explainability import explain_churn_prediction, explain_clv_prediction
from src.feature_store import CustomerNotFoundError, FeatureStore
from src.feature_engineering import make_recommendation_actions
from src.logging_config import get_logger
from src.recommendation_engine import get_recommendation
from src.registry import ModelRegistry

logger = get_logger(__name__)


class ModelArtifactsError(RuntimeError):
    """Raised when trained artifacts are missing or incomplete."""


@dataclass
class PredictionResult:
    """Scoring outcome for a single customer."""

    customer_id: int
    cluster_label: str
    predicted_clv: float
    churn_probability: float
    decision: dict[str, str]
    recommendation_actions: list[str]
    explanations: dict[str, list[str]]
    shap_top_features: dict[str, list[dict[str, Any]]]
    model_version: str
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "CustomerID": self.customer_id,
            "ClusterLabel": self.cluster_label,
            "PredictedCLV": self.predicted_clv,
            "ChurnProbability": self.churn_probability,
            "Decision": self.decision,
            "RecommendationActions": self.recommendation_actions,
            "Explanations": self.explanations,
            "ShapTopFeatures": self.shap_top_features,
            "ModelVersion": self.model_version,
            "LatencyMs": self.latency_ms,
        }


@dataclass
class _Artifacts:
    rfm: dict[str, Any]
    clv: dict[str, Any]
    churn: dict[str, Any]
    clv_importance: pd.DataFrame = field(repr=False)
    churn_importance: pd.DataFrame = field(repr=False)


class PredictionService:
    """Scores a single customer against the production model set."""

    def __init__(
        self,
        feature_store: FeatureStore,
        models_dir: Path,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._feature_store = feature_store
        self._models_dir = models_dir
        self._registry = registry
        self._artifacts: _Artifacts | None = None
        self._lock = threading.Lock()

    # --- artifact lifecycle -------------------------------------------------

    def _load_artifacts(self) -> _Artifacts:
        if self._artifacts is not None:
            return self._artifacts

        with self._lock:
            if self._artifacts is not None:
                return self._artifacts

            required = [
                "rfm_kmeans_artifacts.joblib",
                "clv_model_artifacts.joblib",
                "churn_model_artifacts.joblib",
            ]
            missing = [name for name in required if not (self._models_dir / name).exists()]
            if missing:
                raise ModelArtifactsError(
                    f"Missing model artifacts in {self._models_dir}: {missing}. "
                    "Run project/src/train_models.py first."
                )

            self._artifacts = _Artifacts(
                rfm=joblib.load(self._models_dir / "rfm_kmeans_artifacts.joblib"),
                clv=joblib.load(self._models_dir / "clv_model_artifacts.joblib"),
                churn=joblib.load(self._models_dir / "churn_model_artifacts.joblib"),
                clv_importance=pd.read_csv(self._models_dir / "clv_feature_importance.csv"),
                churn_importance=pd.read_csv(self._models_dir / "churn_feature_importance.csv"),
            )
            logger.info("Model artifacts loaded", extra={"models_dir": str(self._models_dir)})

        return self._artifacts

    def warm_up(self) -> None:
        """Preload artifacts and the feature store so the first request is fast."""
        self._load_artifacts()
        self._feature_store.stats()

    def is_ready(self) -> bool:
        try:
            self._load_artifacts()
            return self._feature_store.exists()
        except (ModelArtifactsError, OSError):
            return False

    @property
    def model_version(self) -> str:
        return self._registry.production_version() if self._registry else "unregistered"

    # --- scoring ------------------------------------------------------------

    @staticmethod
    def _model_input(
        row: pd.Series, feature_columns: list[str], extras: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Assemble a one-row frame in the column order the model was fitted on."""
        extras = extras or {}
        missing = [
            column
            for column in feature_columns
            if column not in extras and column not in row.index
        ]
        if missing:
            raise ModelArtifactsError(
                f"Feature store is missing columns the model expects: {missing}. "
                "Rebuild the feature store after retraining."
            )

        values = {
            column: (extras[column] if column in extras else row[column])
            for column in feature_columns
        }
        return pd.DataFrame([values], columns=feature_columns)

    def _cluster_for(self, row: pd.Series, rfm_artifacts: dict[str, Any]) -> str:
        rfm_values = pd.DataFrame(
            [[row["Recency"], row["Frequency"], row["Monetary"]]],
            columns=["Recency", "Frequency", "Monetary"],
        )
        scaled = rfm_artifacts["scaler"].transform(rfm_values)
        cluster_id = int(rfm_artifacts["model"].predict(scaled)[0])
        return rfm_artifacts["label_map"].get(cluster_id, "Regular Customers")

    def predict(self, customer_id: int, include_explanations: bool = True) -> PredictionResult:
        """Score one customer.

        Raises:
            CustomerNotFoundError: no feature row for this customer.
            ModelArtifactsError: artifacts missing or schema drifted.
        """
        started = time.perf_counter()
        artifacts = self._load_artifacts()
        row = self._feature_store.get_customer(customer_id)

        cluster_label = self._cluster_for(row, artifacts.rfm)

        clv_input = self._model_input(row, artifacts.clv["feature_columns"])
        predicted_clv = float(artifacts.clv["model"].predict(clv_input)[0])

        churn_input = self._model_input(
            row,
            artifacts.churn["feature_columns"],
            extras={"PredictedCLV": predicted_clv, "ClusterLabel": cluster_label},
        )
        churn_probability = float(artifacts.churn["model"].predict_proba(churn_input)[:, 1][0])

        explanations: dict[str, list[str]] = {"CLV": [], "Churn": []}
        shap_top: dict[str, list[dict[str, Any]]] = {"CLV": [], "Churn": []}

        if include_explanations:
            clv_shap = explain_clv_prediction(
                artifacts.clv["model"], clv_input, top_n=3,
                background=artifacts.clv.get("background_sample"),
            )
            churn_shap = explain_churn_prediction(
                artifacts.churn["model"], churn_input, top_n=3,
                background=artifacts.churn.get("background_sample"),
            )
            explanations = {
                "CLV": [clv_shap["explanation"]] + _format_shap(clv_shap["top_features"]),
                "Churn": [churn_shap["explanation"]] + _format_shap(churn_shap["top_features"]),
            }
            shap_top = {"CLV": clv_shap["top_features"], "Churn": churn_shap["top_features"]}

        clv_threshold = float(artifacts.clv.get("high_clv_threshold", 0.0))
        actions = make_recommendation_actions(
            cluster_label=cluster_label,
            predicted_clv=predicted_clv,
            churn_probability=churn_probability,
            clv_high_threshold=clv_threshold,
        )
        decision = get_recommendation(
            {
                "PredictedCLV": predicted_clv,
                "ChurnProbability": churn_probability,
                "ClusterLabel": cluster_label,
            }
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PredictionResult(
            customer_id=int(customer_id),
            cluster_label=cluster_label,
            predicted_clv=predicted_clv,
            churn_probability=churn_probability,
            decision=decision,
            recommendation_actions=actions,
            explanations=explanations,
            shap_top_features=shap_top,
            model_version=self.model_version,
            latency_ms=round(latency_ms, 2),
        )


def _format_shap(top_rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"{row['Feature']} ({row['Direction']}, SHAP={row['Contribution']:.4f})"
        for row in top_rows
    ]


__all__ = [
    "CustomerNotFoundError",
    "ModelArtifactsError",
    "PredictionResult",
    "PredictionService",
]

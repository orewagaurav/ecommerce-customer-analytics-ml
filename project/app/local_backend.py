"""In-process scoring backend.

Mirrors `AnalyticsApiClient`'s interface exactly, but calls `PredictionService`
directly instead of going over HTTP. The dashboard picks whichever is available,
so pages are written against one shape and never know which is in use.

Why this exists: the dashboard is deployed to Streamlit Community Cloud, which
runs a single process and cannot host the FastAPI service alongside it. Without
a local backend, six of the nine pages would show "API unreachable" on the
public demo.

The API remains the primary path — under `docker compose` or locally the HTTP
client is used, and this module is never touched. Both routes call the same
`PredictionService`, so behaviour cannot diverge between them; only the
transport differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.analytics_store import AnalyticsAggregates
from src.config import Settings
from src.feature_store import CustomerNotFoundError, FeatureStore
from src.registry import ModelRegistry
from src.services.history_service import PredictionHistoryService
from src.services.prediction_service import PredictionService
from src.services.report_service import ReportService


@dataclass
class LocalBackend:
    """Scoring backend that runs in the dashboard's own process."""

    settings: Settings

    def __post_init__(self) -> None:
        self._feature_store = FeatureStore(self.settings.feature_store_path)
        self._registry = ModelRegistry(self.settings.registry_path)
        self._aggregates = AnalyticsAggregates(self.settings.analytics_store_path)
        self._service = PredictionService(
            feature_store=self._feature_store,
            models_dir=self.settings.model_path,
            registry=self._registry,
        )
        self._history = PredictionHistoryService(
            path=self.settings.history_path,
            enabled=self.settings.enable_prediction_history,
        )
        self._reports = ReportService(
            prediction_service=self._service,
            feature_store=self._feature_store,
            aggregates=self._aggregates,
        )

    # --- parity with AnalyticsApiClient -------------------------------------

    def is_available(self) -> bool:
        return self._service.is_ready()

    def health(self) -> dict:
        return {
            "status": "healthy" if self._service.is_ready() else "degraded",
            "version": self.settings.api_version,
            "environment": self.settings.environment,
            "models_loaded": self._service.is_ready(),
            "feature_store_available": self._feature_store.exists(),
        }

    def predict(self, customer_id: int, include_explanations: bool = True) -> dict:
        """Returns the API's snake_case wire shape, not the internal dict."""
        result = self._service.predict(int(customer_id), include_explanations)

        self._history.record(
            customer_id=result.customer_id,
            predicted_clv=result.predicted_clv,
            churn_probability=result.churn_probability,
            cluster_label=result.cluster_label,
            recommended_action=result.decision.get("RecommendedAction", ""),
            model_version=result.model_version,
            latency_ms=result.latency_ms,
        )

        return {
            "customer_id": result.customer_id,
            "cluster_label": result.cluster_label,
            "predicted_clv": result.predicted_clv,
            "churn_probability": result.churn_probability,
            "decision": {
                "customer_segment": result.decision.get("CustomerSegment", ""),
                "priority_level": result.decision.get("PriorityLevel", ""),
                "recommended_action": result.decision.get("RecommendedAction", ""),
            },
            "recommendation_actions": result.recommendation_actions,
            "explanations": result.explanations,
            "shap_top_features": {
                task: [
                    {
                        "feature": row["Feature"],
                        "contribution": row["Contribution"],
                        "abs_contribution": row["AbsContribution"],
                        "direction": row["Direction"],
                    }
                    for row in rows
                ]
                for task, rows in result.shap_top_features.items()
            },
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
        }

    def model_info(self) -> dict:
        manifest = self._registry.load()
        store = None
        if self._feature_store.exists():
            stats = self._feature_store.stats()
            store = {
                "built_at": stats.built_at,
                "source_rows": stats.source_rows,
                "customer_count": stats.customer_count,
                "path": stats.path,
            }
        return {
            "production_version": self._registry.production_version(),
            "registry_updated_at": manifest.updated_at,
            "models": [record.to_dict() for record in manifest.models],
            "feature_store": store,
        }

    def metrics(self) -> dict:
        summary = self._history.summary()
        return {
            "process": {
                "uptime_seconds": 0.0,
                "requests": 0,
                "predictions": summary.get("total_predictions", 0),
                "failed_requests": 0,
            },
            "lifetime": {
                "predictions": summary.get("total_predictions", 0),
                "avg_latency_ms": summary.get("avg_latency_ms"),
                "p95_latency_ms": summary.get("p95_latency_ms"),
                "avg_churn_probability": summary.get("avg_churn_probability"),
                "unique_customers": summary.get("unique_customers"),
            },
            "model_version": self._registry.production_version(),
        }

    def customer_ids(self, limit: int = 500) -> list[int]:
        return self._feature_store.customer_ids()[:limit]

    def customer_profile(self, customer_id: int) -> dict:
        return {
            "customer_id": int(customer_id),
            "features": self._service.customer_profile(int(customer_id)),
        }

    def simulate(self, customer_id: int, overrides: dict[str, float]) -> dict:
        baseline, simulated = self._service.simulate(int(customer_id), overrides)

        def _outcome(result: Any) -> dict:
            return {
                "cluster_label": result.cluster_label,
                "predicted_clv": result.predicted_clv,
                "churn_probability": result.churn_probability,
                "recommended_action": result.decision.get("RecommendedAction", ""),
                "priority_level": result.decision.get("PriorityLevel", ""),
            }

        return {
            "customer_id": int(customer_id),
            "applied_overrides": overrides,
            "baseline": _outcome(baseline),
            "simulated": _outcome(simulated),
            "clv_delta": round(simulated.predicted_clv - baseline.predicted_clv, 2),
            "churn_delta": round(simulated.churn_probability - baseline.churn_probability, 6),
            "segment_changed": baseline.cluster_label != simulated.cluster_label,
            "model_version": baseline.model_version,
        }

    def history(
        self,
        limit: int = 100,
        customer_id: int | None = None,
        min_churn_probability: float | None = None,
    ) -> dict:
        frame = self._history.read(
            limit=limit,
            customer_id=customer_id,
            min_churn_probability=min_churn_probability,
        )
        return {
            "total": int(len(self._history.read())),
            "returned": int(len(frame)),
            "entries": frame.to_dict(orient="records"),
        }

    # --- reports ------------------------------------------------------------

    def customer_pdf(self, customer_id: int) -> bytes:
        return self._reports.customer_pdf(int(customer_id))

    def customers_excel(self, limit: int = 2000) -> bytes:
        return self._reports.customers_workbook(limit=limit)

    def history_excel(self, limit: int = 5000) -> bytes:
        return self._reports.history_workbook(self._history.read(limit=limit))


__all__ = ["LocalBackend", "CustomerNotFoundError", "pd"]

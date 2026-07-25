"""Dependency injection wiring.

Collaborators are built once at startup and stored on `app.state`, then handed
to routes through FastAPI `Depends`. Routes therefore never import a singleton
or touch the filesystem directly, which is what makes them testable: a test
overrides one provider and the rest of the graph is untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request

from src.config import Settings, get_settings
from src.feature_store import FeatureStore
from src.registry import ModelRegistry
from src.services.history_service import PredictionHistoryService
from src.analytics_store import AnalyticsAggregates
from src.services.prediction_service import PredictionService
from src.services.report_service import ReportService


@dataclass
class ServiceMetrics:
    """In-process counters exposed by /metrics."""

    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    total_predictions: int = 0
    failed_requests: int = 0

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self.started_at, 2)


@dataclass
class AppContainer:
    """Everything the API needs, constructed once at startup."""

    settings: Settings
    feature_store: FeatureStore
    registry: ModelRegistry
    prediction_service: PredictionService
    history_service: PredictionHistoryService
    report_service: ReportService
    aggregates: AnalyticsAggregates
    metrics: ServiceMetrics


def build_container(settings: Settings | None = None) -> AppContainer:
    """Compose the object graph. The only place collaborators are constructed."""
    settings = settings or get_settings()

    feature_store = FeatureStore(settings.feature_store_path)
    registry = ModelRegistry(settings.registry_path)
    aggregates = AnalyticsAggregates(settings.analytics_store_path)
    prediction_service = PredictionService(
        feature_store=feature_store,
        models_dir=settings.model_path,
        registry=registry,
    )

    return AppContainer(
        settings=settings,
        feature_store=feature_store,
        registry=registry,
        prediction_service=prediction_service,
        report_service=ReportService(
            prediction_service=prediction_service,
            feature_store=feature_store,
            aggregates=aggregates,
        ),
        aggregates=aggregates,
        history_service=PredictionHistoryService(
            path=settings.history_path,
            enabled=settings.enable_prediction_history,
        ),
        metrics=ServiceMetrics(),
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings_dep(container: Annotated[AppContainer, Depends(get_container)]) -> Settings:
    return container.settings


def get_prediction_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> PredictionService:
    return container.prediction_service


def get_history_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> PredictionHistoryService:
    return container.history_service


def get_registry(container: Annotated[AppContainer, Depends(get_container)]) -> ModelRegistry:
    return container.registry


def get_feature_store(container: Annotated[AppContainer, Depends(get_container)]) -> FeatureStore:
    return container.feature_store


def get_metrics(container: Annotated[AppContainer, Depends(get_container)]) -> ServiceMetrics:
    return container.metrics


def get_report_service(
    container: Annotated[AppContainer, Depends(get_container)],
) -> ReportService:
    return container.report_service


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ReportServiceDep = Annotated[ReportService, Depends(get_report_service)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
HistoryServiceDep = Annotated[PredictionHistoryService, Depends(get_history_service)]
RegistryDep = Annotated[ModelRegistry, Depends(get_registry)]
FeatureStoreDep = Annotated[FeatureStore, Depends(get_feature_store)]
MetricsDep = Annotated[ServiceMetrics, Depends(get_metrics)]

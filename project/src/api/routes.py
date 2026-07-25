"""Versioned API routes.

Scoring is CPU-bound (sklearn + SHAP), so `/predict` is a `def` endpoint:
FastAPI runs it in a threadpool, keeping the event loop free. The genuinely
async endpoints below only touch in-memory state. Declaring a CPU-bound handler
`async def` would block every other request for its duration.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, status

from src.api.dependencies import (
    FeatureStoreDep,
    HistoryServiceDep,
    MetricsDep,
    PredictionServiceDep,
    RegistryDep,
    SettingsDep,
)
from src.api.schemas import (
    ErrorResponse,
    FeatureStoreInfo,
    HealthResponse,
    MetricsResponse,
    ModelDetail,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/predict/{customer_id}",
    response_model=PredictionResponse,
    summary="Score a single customer",
    # Aliases exist to ingest the internal PascalCase dicts; the wire format
    # stays snake_case so the whole API reads consistently.
    response_model_by_alias=False,
    responses={
        404: {"model": ErrorResponse, "description": "Customer not in the feature store"},
        503: {"model": ErrorResponse, "description": "Models or feature store unavailable"},
    },
)
def predict_customer(
    service: PredictionServiceDep,
    history: HistoryServiceDep,
    metrics: MetricsDep,
    customer_id: int = Path(..., ge=0, description="Customer identifier to score."),
    request: PredictionRequest | None = None,
) -> PredictionResponse:
    """Return segment, CLV, churn probability, SHAP explanation and next action."""
    options = request or PredictionRequest()

    result = service.predict(customer_id, include_explanations=options.include_explanations)
    metrics.total_predictions += 1

    history.record(
        customer_id=result.customer_id,
        predicted_clv=result.predicted_clv,
        churn_probability=result.churn_probability,
        cluster_label=result.cluster_label,
        recommended_action=result.decision.get("RecommendedAction", ""),
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )

    logger.info(
        "Prediction served",
        extra={
            "customer_id": result.customer_id,
            "churn_probability": round(result.churn_probability, 4),
            "predicted_clv": round(result.predicted_clv, 2),
            "segment": result.cluster_label,
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
        },
    )

    return PredictionResponse(
        customer_id=result.customer_id,
        cluster_label=result.cluster_label,
        predicted_clv=result.predicted_clv,
        churn_probability=result.churn_probability,
        decision=result.decision,
        recommendation_actions=result.recommendation_actions,
        explanations=result.explanations,
        shap_top_features=result.shap_top_features,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )


@router.get("/health", response_model=HealthResponse, summary="Liveness and readiness")
async def health(
    service: PredictionServiceDep,
    feature_store: FeatureStoreDep,
    settings: SettingsDep,
) -> HealthResponse:
    """Report whether models and the feature store are actually usable.

    Returns 200 with status "degraded" rather than an error code so an
    orchestrator can distinguish "process is alive" from "ready for traffic".
    """
    models_loaded = service.is_ready()
    store_available = feature_store.exists()

    return HealthResponse(
        status="healthy" if (models_loaded and store_available) else "degraded",
        version=settings.api_version,
        environment=settings.environment,
        models_loaded=models_loaded,
        feature_store_available=store_available,
    )


@router.get("/model-info", response_model=ModelInfoResponse, summary="Deployed model metadata")
async def model_info(registry: RegistryDep, feature_store: FeatureStoreDep) -> ModelInfoResponse:
    """Expose the registry: versions, algorithms, metrics and feature lists."""
    manifest = registry.load()

    store_info = None
    if feature_store.exists():
        stats = feature_store.stats()
        store_info = FeatureStoreInfo(
            built_at=stats.built_at,
            source_rows=stats.source_rows,
            customer_count=stats.customer_count,
            path=stats.path,
        )

    return ModelInfoResponse(
        production_version=registry.production_version(),
        registry_updated_at=manifest.updated_at,
        models=[ModelDetail(**record.to_dict()) for record in manifest.models],
        feature_store=store_info,
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Operational metrics")
async def service_metrics(
    metrics: MetricsDep, history: HistoryServiceDep, registry: RegistryDep
) -> MetricsResponse:
    """Throughput, latency and error counters for the running process."""
    summary = history.summary()

    return MetricsResponse(
        uptime_seconds=metrics.uptime_seconds,
        total_requests=metrics.total_requests,
        total_predictions=metrics.total_predictions,
        failed_requests=metrics.failed_requests,
        avg_latency_ms=summary.get("avg_latency_ms"),
        p95_latency_ms=summary.get("p95_latency_ms"),
        avg_churn_probability=summary.get("avg_churn_probability"),
        unique_customers=summary.get("unique_customers"),
        model_version=registry.production_version(),
    )


@router.get("/customers", summary="Known customer identifiers", status_code=status.HTTP_200_OK)
async def list_customers(feature_store: FeatureStoreDep, limit: int = 100) -> dict:
    """Sample of scoreable customer IDs, so a client need not guess one."""
    ids = feature_store.customer_ids()
    return {"total": len(ids), "limit": limit, "customer_ids": ids[:limit]}

"""Versioned API routes.

Scoring is CPU-bound (sklearn + SHAP), so `/predict` is a `def` endpoint:
FastAPI runs it in a threadpool, keeping the event loop free. The genuinely
async endpoints below only touch in-memory state. Declaring a CPU-bound handler
`async def` would block every other request for its duration.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status
from fastapi.responses import Response, StreamingResponse

from src.api.dependencies import (
    FeatureStoreDep,
    ReportServiceDep,
    HistoryServiceDep,
    MetricsDep,
    PredictionServiceDep,
    RegistryDep,
    SettingsDep,
)
from src.api.schemas import (
    ErrorResponse,
    LifetimeMetrics,
    FeatureStoreInfo,
    HealthResponse,
    MetricsResponse,
    ModelDetail,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ProcessMetrics,
    CustomerProfileResponse,
    HistoryEntry,
    HistoryResponse,
    SimulationOutcome,
    SimulationRequest,
    SimulationResponse,
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
    """Throughput, latency and error counters.

    Split into process-scoped counters and lifetime aggregates read back from
    the persisted history, so a restart cannot make the response contradict
    itself.
    """
    summary = history.summary()

    return MetricsResponse(
        process=ProcessMetrics(
            uptime_seconds=metrics.uptime_seconds,
            requests=metrics.total_requests,
            predictions=metrics.total_predictions,
            failed_requests=metrics.failed_requests,
        ),
        lifetime=LifetimeMetrics(
            predictions=summary.get("total_predictions", 0),
            avg_latency_ms=summary.get("avg_latency_ms"),
            p95_latency_ms=summary.get("p95_latency_ms"),
            avg_churn_probability=summary.get("avg_churn_probability"),
            unique_customers=summary.get("unique_customers"),
        ),
        model_version=registry.production_version(),
    )


@router.get("/customers", summary="Known customer identifiers", status_code=status.HTTP_200_OK)
async def list_customers(feature_store: FeatureStoreDep, limit: int = 100) -> dict:
    """Sample of scoreable customer IDs, so a client need not guess one."""
    ids = feature_store.customer_ids()
    return {"total": len(ids), "limit": limit, "customer_ids": ids[:limit]}


@router.post(
    "/simulate/{customer_id}",
    response_model=SimulationResponse,
    summary="What-if: rescore a customer with overridden features",
    responses={
        404: {"model": ErrorResponse, "description": "Customer not in the feature store"},
        422: {"model": ErrorResponse, "description": "Unknown feature in overrides"},
    },
)
def simulate_customer(
    service: PredictionServiceDep,
    payload: SimulationRequest,
    customer_id: int = Path(..., ge=0),
) -> SimulationResponse:
    """Score a customer as-is and with overrides, returning both plus deltas.

    Read-only: nothing is written to the feature store or the audit log, because
    a hypothetical is not a prediction the system actually made.
    """
    try:
        baseline, simulated = service.simulate(customer_id, payload.overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def _outcome(result) -> SimulationOutcome:
        return SimulationOutcome(
            cluster_label=result.cluster_label,
            predicted_clv=result.predicted_clv,
            churn_probability=result.churn_probability,
            recommended_action=result.decision.get("RecommendedAction", ""),
            priority_level=result.decision.get("PriorityLevel", ""),
        )

    logger.info(
        "Simulation served",
        extra={
            "customer_id": customer_id,
            "overrides": payload.overrides,
            "clv_delta": round(simulated.predicted_clv - baseline.predicted_clv, 2),
        },
    )

    return SimulationResponse(
        customer_id=int(customer_id),
        applied_overrides=payload.overrides,
        baseline=_outcome(baseline),
        simulated=_outcome(simulated),
        clv_delta=round(simulated.predicted_clv - baseline.predicted_clv, 2),
        churn_delta=round(simulated.churn_probability - baseline.churn_probability, 6),
        segment_changed=baseline.cluster_label != simulated.cluster_label,
        model_version=baseline.model_version,
    )


@router.get("/history", response_model=HistoryResponse, summary="Prediction audit log")
async def prediction_history(
    history: HistoryServiceDep,
    limit: int = Query(default=100, ge=1, le=5000),
    customer_id: int | None = Query(default=None, ge=0),
    min_churn_probability: float | None = Query(default=None, ge=0.0, le=1.0),
) -> HistoryResponse:
    """Recorded predictions, most recent first, with optional filters."""
    frame = history.read(
        limit=limit,
        customer_id=customer_id,
        min_churn_probability=min_churn_probability,
    )
    unfiltered_total = len(history.read())

    return HistoryResponse(
        total=unfiltered_total,
        returned=int(len(frame)),
        entries=[HistoryEntry(**record) for record in frame.to_dict(orient="records")],
    )


@router.get(
    "/customers/{customer_id}/profile",
    response_model=CustomerProfileResponse,
    summary="Stored features for one customer",
    responses={404: {"model": ErrorResponse, "description": "Customer not found"}},
)
async def customer_profile(
    service: PredictionServiceDep, customer_id: int = Path(..., ge=0)
) -> CustomerProfileResponse:
    """Feature values behind a prediction, for the Customer 360 view."""
    return CustomerProfileResponse(
        customer_id=int(customer_id), features=service.customer_profile(customer_id)
    )


# --- reports ----------------------------------------------------------------

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get(
    "/reports/customer/{customer_id}/pdf",
    summary="Per-customer PDF briefing",
    response_class=Response,
    responses={
        200: {"content": {"application/pdf": {}}, "description": "PDF report"},
        404: {"model": ErrorResponse, "description": "Customer not found"},
    },
)
def customer_pdf_report(
    reports: ReportServiceDep, customer_id: int = Path(..., ge=0)
) -> Response:
    """One-page briefing: profile, model output, SHAP drivers and top products."""
    payload = reports.customer_pdf(customer_id)
    logger.info("PDF report served", extra={"customer_id": customer_id, "bytes": len(payload)})

    return Response(
        content=payload,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="customer_{customer_id}_report.pdf"'
        },
    )


@router.get(
    "/reports/customers/excel",
    summary="Customer workbook (multi-sheet Excel)",
    response_class=StreamingResponse,
    responses={200: {"content": {XLSX_MEDIA_TYPE: {}}, "description": "Excel workbook"}},
)
def customers_excel_report(reports: ReportServiceDep, limit: int = Query(2000, ge=1, le=20000)):
    """Summary, customer features, monthly and country revenue, top products."""
    payload = reports.customers_workbook(limit=limit)
    return StreamingResponse(
        iter([payload]),
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="customer_analytics.xlsx"'},
    )


@router.get(
    "/reports/history/excel",
    summary="Prediction history workbook",
    response_class=StreamingResponse,
    responses={200: {"content": {XLSX_MEDIA_TYPE: {}}, "description": "Excel workbook"}},
)
def history_excel_report(
    reports: ReportServiceDep,
    history: HistoryServiceDep,
    limit: int = Query(5000, ge=1, le=50000),
):
    """The audit log, plus a per-segment summary sheet."""
    payload = reports.history_workbook(history.read(limit=limit))
    return StreamingResponse(
        iter([payload]),
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": 'attachment; filename="prediction_history.xlsx"'},
    )

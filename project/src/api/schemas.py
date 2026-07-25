"""Pydantic request and response models.

These are the API contract. Field constraints are declared here so malformed
input is rejected at the boundary with a 422 rather than surfacing as a pandas
error three layers down.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Optional body for a scoring call."""

    model_config = ConfigDict(protected_namespaces=())

    include_explanations: bool = Field(
        default=True,
        description="Set false to skip SHAP and return a faster, lighter response.",
    )


class ShapFeature(BaseModel):
    feature: str = Field(alias="Feature")
    contribution: float = Field(alias="Contribution")
    abs_contribution: float = Field(alias="AbsContribution")
    direction: Literal["increase", "decrease"] = Field(alias="Direction")

    model_config = ConfigDict(populate_by_name=True)


class Decision(BaseModel):
    customer_segment: str = Field(alias="CustomerSegment")
    priority_level: str = Field(alias="PriorityLevel")
    recommended_action: str = Field(alias="RecommendedAction")

    model_config = ConfigDict(populate_by_name=True)


class PredictionResponse(BaseModel):
    """Full scoring result for one customer."""

    model_config = ConfigDict(protected_namespaces=())

    customer_id: int = Field(description="Customer identifier that was scored.")
    cluster_label: str = Field(description="RFM segment, e.g. Champions.")
    predicted_clv: float = Field(description="Expected revenue over the next 90 days.")
    churn_probability: float = Field(ge=0.0, le=1.0)
    decision: Decision
    recommendation_actions: list[str]
    explanations: dict[str, list[str]]
    shap_top_features: dict[str, list[ShapFeature]]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Liveness and readiness signal."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["healthy", "degraded"]
    version: str
    environment: str
    models_loaded: bool
    feature_store_available: bool


class ModelDetail(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str
    version: str
    algorithm: str
    task: str
    trained_at: str
    dataset_version: str
    status: str
    feature_list: list[str]
    metrics: dict[str, float]
    notes: str = ""


class FeatureStoreInfo(BaseModel):
    built_at: str
    source_rows: int
    customer_count: int
    path: str


class ModelInfoResponse(BaseModel):
    """Everything the registry knows about what is deployed."""

    model_config = ConfigDict(protected_namespaces=())

    production_version: str
    registry_updated_at: str
    models: list[ModelDetail]
    feature_store: FeatureStoreInfo | None = None


class MetricsResponse(BaseModel):
    """Operational counters, not model quality metrics."""

    model_config = ConfigDict(protected_namespaces=())

    uptime_seconds: float
    total_requests: int
    total_predictions: int
    failed_requests: int
    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    avg_churn_probability: float | None = None
    unique_customers: int | None = None
    model_version: str


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    error: str
    detail: str
    request_id: str | None = None
    context: dict[str, Any] | None = None

"""The two scoring backends must stay interchangeable.

The dashboard talks to FastAPI when it is reachable and falls back to in-process
scoring otherwise (Streamlit Community Cloud runs one process and cannot host
the API). Pages are written against one interface, so if the HTTP client gains a
method and the local backend does not, the fallback breaks silently on the
public demo — where it is least likely to be noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.api_client import AnalyticsApiClient
from app.local_backend import LocalBackend
from src.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_STORE = PROJECT_ROOT / "feature_store" / "features.parquet"

# Everything the dashboard pages invoke on whichever backend answered.
BACKEND_METHODS = [
    "predict",
    "model_info",
    "metrics",
    "health",
    "customer_ids",
    "customer_profile",
    "simulate",
    "history",
    "customer_pdf",
    "customers_excel",
    "history_excel",
    "is_available",
]


def test_local_backend_implements_every_client_method():
    """Structural parity, checked without needing artifacts on disk."""
    missing = [name for name in BACKEND_METHODS if not hasattr(LocalBackend, name)]
    assert not missing, f"LocalBackend is missing: {missing}"


def test_http_client_implements_every_backend_method():
    """The reverse direction, so neither side silently grows past the other."""
    missing = [name for name in BACKEND_METHODS if not hasattr(AnalyticsApiClient, name)]
    assert not missing, f"AnalyticsApiClient is missing: {missing}"


artifacts_ready = FEATURE_STORE.exists() and (MODELS_DIR / "churn_model_artifacts.joblib").exists()


@pytest.fixture(scope="module")
def backend(tmp_path_factory) -> LocalBackend:
    settings = Settings(
        feature_store_path=FEATURE_STORE,
        model_path=MODELS_DIR,
        registry_path=MODELS_DIR / "model_registry.json",
        history_path=tmp_path_factory.mktemp("local") / "history.parquet",
        log_level="WARNING",
        log_format="console",
    )
    return LocalBackend(settings)


@pytest.mark.skipif(not artifacts_ready, reason="artifacts or feature store not built")
def test_local_predict_returns_the_api_wire_shape(backend):
    """Pages read snake_case keys, so the local backend must emit them too."""
    customer_id = backend.customer_ids(limit=1)[0]
    payload = backend.predict(customer_id)

    for key in (
        "customer_id", "cluster_label", "predicted_clv", "churn_probability",
        "decision", "recommendation_actions", "explanations", "shap_top_features",
        "model_version", "latency_ms",
    ):
        assert key in payload, f"missing '{key}' - pages read this key"

    assert set(payload["decision"]) == {
        "customer_segment", "priority_level", "recommended_action"
    }


@pytest.mark.skipif(not artifacts_ready, reason="artifacts or feature store not built")
def test_local_shap_entries_use_snake_case_keys(backend):
    customer_id = backend.customer_ids(limit=1)[0]
    rows = backend.predict(customer_id)["shap_top_features"]["Churn"]

    assert rows, "expected SHAP contributors"
    assert set(rows[0]) == {"feature", "contribution", "abs_contribution", "direction"}


@pytest.mark.skipif(not artifacts_ready, reason="artifacts or feature store not built")
def test_local_backend_generates_real_reports(backend):
    customer_id = backend.customer_ids(limit=1)[0]

    assert backend.customer_pdf(customer_id)[:4] == b"%PDF"
    assert backend.customers_excel(limit=25)[:2] == b"PK"


@pytest.mark.skipif(not artifacts_ready, reason="artifacts or feature store not built")
def test_local_simulate_matches_api_response_shape(backend):
    customer_id = backend.customer_ids(limit=1)[0]
    outcome = backend.simulate(customer_id, {"Frequency": 50.0})

    for key in (
        "customer_id", "applied_overrides", "baseline", "simulated",
        "clv_delta", "churn_delta", "segment_changed", "model_version",
    ):
        assert key in outcome


@pytest.mark.skipif(not artifacts_ready, reason="artifacts or feature store not built")
def test_local_metrics_has_the_process_lifetime_split(backend):
    metrics = backend.metrics()

    assert "process" in metrics and "lifetime" in metrics

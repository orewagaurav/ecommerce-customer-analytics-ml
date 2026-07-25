"""Application settings.

Every configurable path, port and threshold is resolved here so that nothing
downstream hardcodes a filesystem layout. Values come from the environment (or
a local .env), which is what lets the same image run under docker compose,
Streamlit Cloud and a developer's laptop without code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


class Settings(BaseSettings):
    """Runtime configuration, overridable by environment variable."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # Guards against pydantic reserving the `model_` prefix on our fields.
        protected_namespaces=(),
    )

    # --- Identity -----------------------------------------------------------
    app_name: str = "ecommerce-customer-analytics"
    api_version: str = "v1"
    environment: Literal["local", "docker", "production"] = "local"

    # --- Paths --------------------------------------------------------------
    data_path: Path = Field(
        default=PROJECT_ROOT / "data" / "processed_online_retail_II.csv",
        description="Processed transactions CSV used for offline feature builds.",
    )
    model_path: Path = Field(
        default=PROJECT_ROOT / "models",
        description="Directory holding the trained joblib artifacts.",
    )
    feature_store_path: Path = Field(
        default=PROJECT_ROOT / "feature_store" / "features.parquet",
        description="Precomputed customer feature table used at inference time.",
    )
    analytics_store_path: Path = Field(
        default=PROJECT_ROOT / "feature_store",
        description="Directory of precomputed dashboard aggregates.",
    )
    registry_path: Path = Field(
        default=PROJECT_ROOT / "models" / "model_registry.json",
        description="Model registry manifest.",
    )
    history_path: Path = Field(
        default=PROJECT_ROOT / "prediction_history" / "history.parquet",
        description="Append-only prediction history.",
    )

    # --- Service ------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    api_base_url: str = Field(
        default="http://localhost:8000",
        description="Where the Streamlit dashboard reaches the API.",
    )
    request_timeout_seconds: float = 30.0

    # --- Observability ------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    # --- Behaviour ----------------------------------------------------------
    enable_prediction_history: bool = True
    high_churn_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

    @field_validator(
        "data_path", "model_path", "feature_store_path",
        "analytics_store_path", "registry_path", "history_path",
    )
    @classmethod
    def _expand(cls, value: Path) -> Path:
        """Accept relative paths and ~ from the environment."""
        return Path(value).expanduser()

    @property
    def api_prefix(self) -> str:
        return f"/{self.api_version}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor so settings are parsed once per process.

    FastAPI dependencies override this in tests via `app.dependency_overrides`.
    """
    return Settings()

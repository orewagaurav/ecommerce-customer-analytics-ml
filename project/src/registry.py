"""Lightweight model registry.

Records what shipped: version, algorithm, training date, dataset fingerprint,
metrics, and the exact feature list each model was fitted on. Written at the end
of training and served read-only through `GET /model-info`, so the running
service can always answer "which model is this and how good was it?" without
anyone opening a notebook.

Deliberately a JSON manifest rather than MLflow: it version-controls with the
code, needs no server, and is honest about the scale of this project.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from src.logging_config import get_logger

logger = get_logger(__name__)

ModelStatus = Literal["production", "staging", "archived"]


@dataclass
class ModelRecord:
    """One trained model as promoted into the registry."""

    name: str
    version: str
    algorithm: str
    task: str
    trained_at: str
    dataset_version: str
    feature_list: list[str]
    metrics: dict[str, float]
    status: ModelStatus = "production"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegistryManifest:
    """The full registry document."""

    registry_version: str = "1.0"
    updated_at: str = ""
    models: list[ModelRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": self.registry_version,
            "updated_at": self.updated_at,
            "models": [model.to_dict() for model in self.models],
        }


def dataset_fingerprint(path: Path, sample_bytes: int = 1_000_000) -> str:
    """Short content hash identifying which dataset a model was trained on.

    Hashes a bounded prefix plus the file size rather than the whole 80 MB file:
    enough to detect a changed dataset, cheap enough to run every training pass.
    """
    if not path.exists():
        return "unknown"

    digest = hashlib.sha256()
    digest.update(str(path.stat().st_size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))

    return digest.hexdigest()[:12]


def build_version(trained_at: datetime | None = None) -> str:
    """Timestamp-derived version, e.g. v20260725.1420."""
    moment = trained_at or datetime.now(timezone.utc)
    return f"v{moment.strftime('%Y%m%d.%H%M')}"


class ModelRegistry:
    """Read/write access to the registry manifest."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> RegistryManifest:
        if not self._path.exists():
            return RegistryManifest(updated_at="")

        payload = json.loads(self._path.read_text(encoding="utf-8"))
        return RegistryManifest(
            registry_version=payload.get("registry_version", "1.0"),
            updated_at=payload.get("updated_at", ""),
            models=[ModelRecord(**record) for record in payload.get("models", [])],
        )

    def save(self, manifest: RegistryManifest) -> None:
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

        logger.info(
            "Model registry written",
            extra={"models": len(manifest.models), "path": str(self._path)},
        )

    def register(self, records: list[ModelRecord]) -> RegistryManifest:
        """Replace the manifest with a freshly trained set of models.

        Training produces all models together from one dataset, so they are
        promoted as a unit; there is no partial-update path to get wrong.
        """
        manifest = RegistryManifest(models=records)
        self.save(manifest)
        return manifest

    def get(self, task: str) -> ModelRecord | None:
        for record in self.load().models:
            if record.task == task:
                return record
        return None

    def production_version(self) -> str:
        """Single version string identifying the deployed model set."""
        models = self.load().models
        return models[0].version if models else "unregistered"

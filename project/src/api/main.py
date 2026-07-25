"""FastAPI application factory.

Assembles the app: configure logging, build the dependency container, register
middleware and handlers, mount the versioned router. Exposed as a factory rather
than a module-level singleton so tests can build isolated instances pointing at
temporary directories.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import build_container
from src.api.errors import register_exception_handlers
from src.api.routes import router
from src.config import Settings, get_settings
from src.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

DESCRIPTION = """
Customer intelligence scoring service.

Returns RFM segment, 90-day CLV forecast, churn probability, a SHAP explanation
and a recommended next action for a given customer.

Models are trained offline; features are served from a precomputed store, so
scoring latency is independent of dataset size.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured application instance."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, log_format=settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = build_container(settings)
        logger.info(
            "Service starting",
            extra={"environment": settings.environment, "api_version": settings.api_version},
        )

        # Warm the caches so the first real request is not the slow one.
        try:
            app.state.container.prediction_service.warm_up()
            logger.info("Warm-up complete")
        except Exception as exc:  # noqa: BLE001 - degraded start is intentional
            logger.warning(
                "Warm-up skipped; service starts degraded",
                extra={"reason": str(exc)},
            )

        yield

        app.state.container.history_service.flush()
        logger.info("Service stopped")

    app = FastAPI(
        title="E-Commerce Customer Analytics API",
        description=DESCRIPTION,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag each request with an ID and log its outcome and latency."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = request_id
        started = time.perf_counter()

        container = getattr(app.state, "container", None)
        if container is not None:
            container.metrics.total_requests += 1

        try:
            response = await call_next(request)
        except Exception:
            if container is not None:
                container.metrics.failed_requests += 1
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if container is not None and response.status_code >= 500:
            container.metrics.failed_requests += 1

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(duration_ms)

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    register_exception_handlers(app)
    app.include_router(router, prefix=settings.api_prefix, tags=["predictions"])

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": settings.api_version,
            "docs": "/docs",
            "health": f"{settings.api_prefix}/health",
        }

    return app


app = create_app()

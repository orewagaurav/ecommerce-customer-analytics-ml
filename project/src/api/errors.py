"""Exception handlers.

Domain exceptions are translated into HTTP responses in one place, so services
raise meaningful Python errors and never import `HTTPException`. Every failure
leaves the process through the same `ErrorResponse` envelope and carries the
request ID, which is what makes a user-reported error traceable in the logs.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import ErrorResponse
from src.feature_store import CustomerNotFoundError, FeatureStoreError
from src.logging_config import get_logger
from src.services.prediction_service import ModelArtifactsError

logger = get_logger(__name__)


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _envelope(
    request: Request, status_code: int, error: str, detail: str, context: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error, detail=detail, request_id=_request_id(request), context=context
        ).model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every domain-to-HTTP translation."""

    @app.exception_handler(CustomerNotFoundError)
    async def _customer_not_found(request: Request, exc: CustomerNotFoundError) -> JSONResponse:
        logger.warning(
            "Customer not found",
            extra={"customer_id": exc.customer_id, "request_id": _request_id(request)},
        )
        return _envelope(
            request,
            status.HTTP_404_NOT_FOUND,
            "customer_not_found",
            str(exc),
            {"customer_id": exc.customer_id},
        )

    @app.exception_handler(FeatureStoreError)
    async def _feature_store_unavailable(request: Request, exc: FeatureStoreError) -> JSONResponse:
        # 503, not 500: the service is correctly built but a build step is missing.
        logger.error("Feature store unavailable", extra={"request_id": _request_id(request)})
        return _envelope(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "feature_store_unavailable",
            str(exc),
        )

    @app.exception_handler(ModelArtifactsError)
    async def _artifacts_unavailable(request: Request, exc: ModelArtifactsError) -> JSONResponse:
        logger.error("Model artifacts unavailable", extra={"request_id": _request_id(request)})
        return _envelope(
            request,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "model_artifacts_unavailable",
            str(exc),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _envelope(
            request,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "validation_error",
            "Request failed validation.",
            {"errors": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the traceback, return a generic message: internals never leak.
        logger.exception(
            "Unhandled error",
            extra={"request_id": _request_id(request), "path": request.url.path},
        )
        return _envelope(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected error occurred.",
        )

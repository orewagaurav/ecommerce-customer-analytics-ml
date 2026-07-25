"""HTTP client the dashboard uses to reach the scoring API.

The dashboard holds no model code and imports nothing from `src.predict`; it
talks to the service over HTTP exactly as any other consumer would. That is what
makes the API/UI split real, and it means the dashboard keeps working unchanged
when the API moves to another host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ApiError(RuntimeError):
    """A request to the scoring service failed."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class CustomerNotFound(ApiError):
    """The requested customer is not in the feature store."""


@dataclass
class AnalyticsApiClient:
    """Thin typed wrapper over the scoring API."""

    base_url: str
    api_version: str = "v1"
    timeout: float = 30.0

    @property
    def _prefix(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.api_version}"

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        url = f"{self._prefix}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(method, url, **kwargs)
        except httpx.RequestError as exc:
            raise ApiError(
                f"Could not reach the scoring API at {self.base_url}. "
                "Is it running? (docker compose up, or uvicorn locally)"
            ) from exc

        if response.status_code == 404:
            body = _safe_json(response)
            raise CustomerNotFound(body.get("detail", "Customer not found"), 404)

        if response.status_code >= 400:
            body = _safe_json(response)
            raise ApiError(
                body.get("detail", f"API returned {response.status_code}"),
                response.status_code,
            )

        return response.json()

    # --- endpoints ----------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def predict(self, customer_id: int, include_explanations: bool = True) -> dict:
        return self._request(
            "POST",
            f"/predict/{int(customer_id)}",
            json={"include_explanations": include_explanations},
        )

    def model_info(self) -> dict:
        return self._request("GET", "/model-info")

    def metrics(self) -> dict:
        return self._request("GET", "/metrics")

    def customer_ids(self, limit: int = 500) -> list[int]:
        return self._request("GET", f"/customers?limit={limit}")["customer_ids"]

    def simulate(self, customer_id: int, overrides: dict[str, float]) -> dict:
        """Rescore a customer with overridden features (read-only what-if)."""
        return self._request(
            "POST", f"/simulate/{int(customer_id)}", json={"overrides": overrides}
        )

    def history(
        self,
        limit: int = 100,
        customer_id: int | None = None,
        min_churn_probability: float | None = None,
    ) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if customer_id is not None:
            params["customer_id"] = int(customer_id)
        if min_churn_probability is not None:
            params["min_churn_probability"] = min_churn_probability
        return self._request("GET", "/history", params=params)

    def customer_profile(self, customer_id: int) -> dict:
        return self._request("GET", f"/customers/{int(customer_id)}/profile")

    def _download(self, path: str) -> bytes:
        """Fetch a binary artefact (PDF/XLSX) rather than a JSON body."""
        url = f"{self._prefix}{path}"
        try:
            with httpx.Client(timeout=max(self.timeout, 60.0)) as client:
                response = client.get(url)
        except httpx.RequestError as exc:
            raise ApiError(f"Could not reach the scoring API at {self.base_url}.") from exc

        if response.status_code == 404:
            raise CustomerNotFound("Customer not found", 404)
        if response.status_code >= 400:
            raise ApiError(f"Report generation failed ({response.status_code})", response.status_code)

        return response.content

    def customer_pdf(self, customer_id: int) -> bytes:
        return self._download(f"/reports/customer/{int(customer_id)}/pdf")

    def customers_excel(self, limit: int = 2000) -> bytes:
        return self._download(f"/reports/customers/excel?limit={int(limit)}")

    def history_excel(self, limit: int = 5000) -> bytes:
        return self._download(f"/reports/history/excel?limit={int(limit)}")

    def is_available(self) -> bool:
        """Non-raising probe used to render a connection banner."""
        try:
            self.health()
            return True
        except ApiError:
            return False


def _safe_json(response: httpx.Response) -> dict:
    try:
        return response.json()
    except ValueError:
        return {}

"""Typed API client used by the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import requests
from pydantic import BaseModel, ValidationError

from sec_copilot.api.models import (
    BuildInfoResponse,
    CoverageFailureResponse,
    HealthResponse,
    IngestRunRequest,
    IngestRunResponse,
    QuerySuccessResponse,
    RetrievalDebugResponse,
    ServiceNotReadyResponse,
)
from sec_copilot.schemas.retrieval import QueryRequest


@dataclass(frozen=True)
class ApiNetworkError:
    """Transport failure reaching the backend."""

    endpoint: str
    message: str


@dataclass(frozen=True)
class ApiBackendError:
    """Unexpected backend failure that did not match a typed contract."""

    endpoint: str
    status_code: int
    message: str
    raw_body: object | str | None = None


@dataclass(frozen=True)
class ApiMalformedResponse:
    """Response could be reached, but did not match the expected schema."""

    endpoint: str
    status_code: int
    message: str
    raw_body: object | str | None = None


class ApiClient:
    """Thin typed client over the read-oriented FastAPI surfaces used by V6."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 20.0,
        session: requests.Session | Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def health(self) -> HealthResponse | ApiNetworkError | ApiBackendError | ApiMalformedResponse:
        return self._request_json("GET", "/health", success_model=HealthResponse)

    def build_info(self) -> BuildInfoResponse | ApiNetworkError | ApiBackendError | ApiMalformedResponse:
        return self._request_json("GET", "/build-info", success_model=BuildInfoResponse)

    def query(
        self,
        request: QueryRequest,
    ) -> (
        QuerySuccessResponse
        | ServiceNotReadyResponse
        | CoverageFailureResponse
        | ApiNetworkError
        | ApiBackendError
        | ApiMalformedResponse
    ):
        return self._request_json(
            "POST",
            "/query",
            success_model=QuerySuccessResponse,
            json_body=request.model_dump(mode="json"),
            error_models={
                409: CoverageFailureResponse,
                503: ServiceNotReadyResponse,
            },
        )

    def ingest_run(
        self,
        request: IngestRunRequest,
    ) -> IngestRunResponse | ApiNetworkError | ApiBackendError | ApiMalformedResponse:
        return self._request_json(
            "POST",
            "/ingest/run",
            success_model=IngestRunResponse,
            json_body=request.model_dump(mode="json"),
        )

    def retrieve_debug(
        self,
        request: QueryRequest,
    ) -> (
        RetrievalDebugResponse
        | ServiceNotReadyResponse
        | CoverageFailureResponse
        | ApiNetworkError
        | ApiBackendError
        | ApiMalformedResponse
    ):
        return self._request_json(
            "POST",
            "/retrieve/debug",
            success_model=RetrievalDebugResponse,
            json_body=request.model_dump(mode="json"),
            error_models={
                409: CoverageFailureResponse,
                503: ServiceNotReadyResponse,
            },
        )

    def _request_json(
        self,
        method: str,
        endpoint: str,
        *,
        success_model: type[BaseModel],
        json_body: dict[str, Any] | None = None,
        error_models: dict[int, type[BaseModel]] | None = None,
    ):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, json=json_body, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            return ApiNetworkError(endpoint=endpoint, message=str(exc))

        payload, raw_body = _decode_response_body(response)
        if response.status_code < 400:
            return _parse_model_or_malformed(
                endpoint=endpoint,
                status_code=response.status_code,
                payload=payload,
                raw_body=raw_body,
                model=success_model,
            )

        typed_error_model = (error_models or {}).get(response.status_code)
        if typed_error_model is not None:
            return _parse_model_or_malformed(
                endpoint=endpoint,
                status_code=response.status_code,
                payload=payload,
                raw_body=raw_body,
                model=typed_error_model,
            )

        message = f"Backend returned unexpected status {response.status_code} for {endpoint}."
        return ApiBackendError(
            endpoint=endpoint,
            status_code=response.status_code,
            message=message,
            raw_body=payload if payload is not None else raw_body,
        )


def _decode_response_body(response) -> tuple[object | None, object | str | None]:
    try:
        payload = response.json()
        return payload, payload
    except (ValueError, json.JSONDecodeError):
        text = getattr(response, "text", None)
        return None, text


def _parse_model_or_malformed(
    *,
    endpoint: str,
    status_code: int,
    payload: object | None,
    raw_body: object | str | None,
    model: type[BaseModel],
):
    if payload is None:
        return ApiMalformedResponse(
            endpoint=endpoint,
            status_code=status_code,
            message=f"{endpoint} returned a non-JSON response.",
            raw_body=raw_body,
        )

    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        return ApiMalformedResponse(
            endpoint=endpoint,
            status_code=status_code,
            message=f"{endpoint} response did not match {model.__name__}: {exc}",
            raw_body=raw_body,
        )


__all__ = [
    "ApiBackendError",
    "ApiClient",
    "ApiMalformedResponse",
    "ApiNetworkError",
]

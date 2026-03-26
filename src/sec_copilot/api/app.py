"""FastAPI application for the SEC Filing Intelligence Copilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sec_copilot import __version__
from sec_copilot.api.models import (
    BuildInfoResponse,
    CoverageFailureResponse,
    EvalRunRequest,
    EvalRunResponse,
    HealthResponse,
    IngestRunRequest,
    IngestRunResponse,
    QueryRequest,
    QuerySuccessResponse,
    RetrievalDebugResponse,
    ServiceNotReadyResponse,
)
from sec_copilot.api.service import CopilotApiService
from sec_copilot.ingest.pipeline import IngestionPreflightError
from sec_copilot.utils.logging import configure_logging, log_api_event


def create_app(
    service: CopilotApiService | None = None,
    *,
    include_admin_routes: bool = True,
    title: str = "SEC Filing Intelligence Copilot",
    description: str = "Typed FastAPI backend for grounded SEC filing retrieval, query, ingestion, and evaluation.",
) -> FastAPI:
    """Create the FastAPI app with injectable service state."""

    api_service = service or CopilotApiService()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging()
        api_service.initialize()
        app.state.service = api_service
        yield

    app = FastAPI(
        title=title,
        version=__version__,
        description=description,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = str(uuid4())
        started_at = perf_counter()
        response = await call_next(request)
        total_ms = (perf_counter() - started_at) * 1000.0
        response.headers["x-request-id"] = request_id
        log_api_event(
            {
                "event": "http_request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "total_ms": total_ms,
            }
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return api_service.health()

    @app.get("/build-info", response_model=BuildInfoResponse)
    def build_info() -> BuildInfoResponse:
        return api_service.build_info()

    @app.post(
        "/query",
        response_model=QuerySuccessResponse,
        responses={
            409: {"model": CoverageFailureResponse},
            503: {"model": ServiceNotReadyResponse},
        },
    )
    def query(request: QueryRequest):
        result = api_service.query(request)
        if isinstance(result, ServiceNotReadyResponse):
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        if isinstance(result, CoverageFailureResponse):
            return JSONResponse(status_code=409, content=result.model_dump(mode="json"))
        return result

    @app.post(
        "/retrieve/debug",
        response_model=RetrievalDebugResponse,
        responses={
            409: {"model": CoverageFailureResponse},
            503: {"model": ServiceNotReadyResponse},
        },
    )
    def retrieve_debug(request: QueryRequest):
        result = api_service.retrieve_debug(request)
        if isinstance(result, ServiceNotReadyResponse):
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        if isinstance(result, CoverageFailureResponse):
            return JSONResponse(status_code=409, content=result.model_dump(mode="json"))
        return result

    if include_admin_routes:

        @app.post("/ingest/run", response_model=IngestRunResponse)
        def ingest_run(request: IngestRunRequest) -> IngestRunResponse:
            try:
                return api_service.run_ingest(request)
            except IngestionPreflightError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @app.post("/eval/run", response_model=EvalRunResponse)
        def eval_run(request: EvalRunRequest) -> EvalRunResponse:
            try:
                return api_service.run_eval(request)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()


__all__ = ["app", "create_app"]

"""FastAPI application for the SEC Filing Intelligence Copilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from sec_copilot import __version__
from sec_copilot.api.metrics import (
    CONTENT_TYPE_LATEST,
    increment_query_abstention,
    increment_query_error,
    observe_eval,
    observe_http_request,
    observe_ingest,
    observe_query,
    observe_retrieval_debug,
    render_metrics,
)
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
    ReadinessResponse,
    RetrievalDebugResponse,
    ServiceNotReadyResponse,
)
from sec_copilot.api.service import CopilotApiService
from sec_copilot.config.runtime import load_api_runtime_settings_from_env
from sec_copilot.ingest.pipeline import IngestionPreflightError
from sec_copilot.utils.logging import configure_logging, log_api_event

load_dotenv()


def _metrics_path(request: Request) -> str:
    """Return a low-cardinality route label for Prometheus metrics."""

    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path or "unmatched"


def _seconds_from_ms(duration_ms: float) -> float:
    """Convert milliseconds to seconds for Prometheus observations."""

    return max(duration_ms, 0.0) / 1000.0


def create_app(
    service: CopilotApiService | None = None,
    *,
    include_admin_routes: bool | None = None,
    title: str = "SEC Filing Intelligence Copilot",
    description: str = "Typed FastAPI backend for grounded SEC filing retrieval, query, ingestion, and evaluation.",
) -> FastAPI:
    """Create the FastAPI app with injectable service state."""

    runtime_settings = load_api_runtime_settings_from_env()
    api_service = service or CopilotApiService()
    admin_routes_enabled = runtime_settings.enable_admin_routes if include_admin_routes is None else include_admin_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(api_service.settings.log_level)
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
        try:
            response = await call_next(request)
        except Exception:
            total_ms = (perf_counter() - started_at) * 1000.0
            if request.url.path != "/metrics":
                observe_http_request(
                    method=request.method,
                    path=_metrics_path(request),
                    status_code=500,
                    duration_seconds=_seconds_from_ms(total_ms),
                )
            log_api_event(
                {
                    "event": "http_request_completed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "total_ms": total_ms,
                }
            )
            raise

        total_ms = (perf_counter() - started_at) * 1000.0
        response.headers["x-request-id"] = request_id
        if request.url.path != "/metrics":
            observe_http_request(
                method=request.method,
                path=_metrics_path(request),
                status_code=response.status_code,
                duration_seconds=_seconds_from_ms(total_ms),
            )
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

    @app.get(
        "/readyz",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    def readyz():
        readiness = api_service.readiness()
        if not readiness.query_ready:
            return JSONResponse(status_code=503, content=readiness.model_dump(mode="json"))
        return readiness

    @app.get("/build-info", response_model=BuildInfoResponse)
    def build_info() -> BuildInfoResponse:
        return api_service.build_info()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=render_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/query",
        response_model=QuerySuccessResponse,
        responses={
            409: {"model": CoverageFailureResponse},
            503: {"model": ServiceNotReadyResponse},
        },
    )
    def query(request: QueryRequest):
        started_at = perf_counter()
        try:
            result = api_service.query(request)
        except Exception:
            observe_query(
                outcome="server_error",
                duration_seconds=perf_counter() - started_at,
            )
            increment_query_error("server_error")
            raise

        if isinstance(result, ServiceNotReadyResponse):
            observe_query(
                outcome=result.error_type,
                duration_seconds=perf_counter() - started_at,
            )
            increment_query_error(result.error_type)
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        if isinstance(result, CoverageFailureResponse):
            observe_query(
                outcome=result.error_type,
                duration_seconds=perf_counter() - started_at,
            )
            increment_query_error(result.error_type)
            return JSONResponse(status_code=409, content=result.model_dump(mode="json"))
        observe_query(
            outcome="abstained" if result.abstained else "ok",
            duration_seconds=perf_counter() - started_at,
        )
        if result.abstained:
            increment_query_abstention(result.reason_code)
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
        started_at = perf_counter()
        try:
            result = api_service.retrieve_debug(request)
        except Exception:
            observe_retrieval_debug(
                outcome="server_error",
                duration_seconds=perf_counter() - started_at,
            )
            raise

        if isinstance(result, ServiceNotReadyResponse):
            observe_retrieval_debug(
                outcome=result.error_type,
                duration_seconds=perf_counter() - started_at,
            )
            return JSONResponse(status_code=503, content=result.model_dump(mode="json"))
        if isinstance(result, CoverageFailureResponse):
            observe_retrieval_debug(
                outcome=result.error_type,
                duration_seconds=perf_counter() - started_at,
            )
            return JSONResponse(status_code=409, content=result.model_dump(mode="json"))
        observe_retrieval_debug(
            outcome=result.reason_code,
            duration_seconds=perf_counter() - started_at,
        )
        return result

    if admin_routes_enabled:

        @app.post("/ingest/run", response_model=IngestRunResponse)
        def ingest_run(request: IngestRunRequest) -> IngestRunResponse:
            started_at = perf_counter()
            try:
                result = api_service.run_ingest(request)
            except IngestionPreflightError as exc:
                observe_ingest(
                    outcome="bad_request",
                    duration_seconds=perf_counter() - started_at,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except ValueError as exc:
                observe_ingest(
                    outcome="bad_request",
                    duration_seconds=perf_counter() - started_at,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                observe_ingest(
                    outcome="server_error",
                    duration_seconds=perf_counter() - started_at,
                )
                raise
            observe_ingest(
                outcome="ok",
                duration_seconds=perf_counter() - started_at,
            )
            return result

        @app.post("/eval/run", response_model=EvalRunResponse)
        def eval_run(request: EvalRunRequest) -> EvalRunResponse:
            started_at = perf_counter()
            try:
                result = api_service.run_eval(request)
            except ValueError as exc:
                observe_eval(
                    outcome="bad_request",
                    duration_seconds=perf_counter() - started_at,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except RuntimeError as exc:
                observe_eval(
                    outcome="bad_request",
                    duration_seconds=perf_counter() - started_at,
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception:
                observe_eval(
                    outcome="server_error",
                    duration_seconds=perf_counter() - started_at,
                )
                raise
            observe_eval(
                outcome="ok",
                duration_seconds=perf_counter() - started_at,
            )
            return result

    return app


app = create_app()
public_app = create_app(include_admin_routes=False)
admin_app = create_app(include_admin_routes=True)


__all__ = ["admin_app", "app", "create_app", "public_app"]

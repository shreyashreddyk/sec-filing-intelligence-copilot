"""Prometheus metrics exposed by the FastAPI layer."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest


LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    900.0,
)

METRICS_REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS_TOTAL = Counter(
    "sec_copilot_http_requests_total",
    "Total HTTP requests handled by the FastAPI app.",
    labelnames=("method", "path", "status_code"),
    registry=METRICS_REGISTRY,
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "sec_copilot_http_request_duration_seconds",
    "End-to-end HTTP request latency for the FastAPI app.",
    labelnames=("method", "path", "status_code"),
    buckets=LATENCY_BUCKETS_SECONDS,
    registry=METRICS_REGISTRY,
)
QUERY_DURATION_SECONDS = Histogram(
    "sec_copilot_query_duration_seconds",
    "Latency for the grounded /query endpoint.",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS_SECONDS,
    registry=METRICS_REGISTRY,
)
RETRIEVAL_DEBUG_DURATION_SECONDS = Histogram(
    "sec_copilot_retrieval_debug_duration_seconds",
    "Latency for the /retrieve/debug endpoint.",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS_SECONDS,
    registry=METRICS_REGISTRY,
)
INGEST_DURATION_SECONDS = Histogram(
    "sec_copilot_ingest_duration_seconds",
    "Latency for the admin /ingest/run endpoint.",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS_SECONDS,
    registry=METRICS_REGISTRY,
)
EVAL_DURATION_SECONDS = Histogram(
    "sec_copilot_eval_duration_seconds",
    "Latency for the admin /eval/run endpoint.",
    labelnames=("outcome",),
    buckets=LATENCY_BUCKETS_SECONDS,
    registry=METRICS_REGISTRY,
)
QUERY_ERRORS_TOTAL = Counter(
    "sec_copilot_query_errors_total",
    "Query failures grouped by error type.",
    labelnames=("error_type",),
    registry=METRICS_REGISTRY,
)
QUERY_ABSTENTIONS_TOTAL = Counter(
    "sec_copilot_query_abstentions_total",
    "Successful query responses that abstained, grouped by reason code.",
    labelnames=("reason_code",),
    registry=METRICS_REGISTRY,
)


def render_metrics() -> bytes:
    """Render the custom metrics registry using Prometheus text format."""

    return generate_latest(METRICS_REGISTRY)


def observe_http_request(*, method: str, path: str, status_code: int, duration_seconds: float) -> None:
    """Record one completed FastAPI request."""

    labels = {"method": method, "path": path, "status_code": str(status_code)}
    HTTP_REQUESTS_TOTAL.labels(**labels).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(**labels).observe(duration_seconds)


def observe_query(*, outcome: str, duration_seconds: float) -> None:
    """Record one `/query` latency observation."""

    QUERY_DURATION_SECONDS.labels(outcome=outcome).observe(duration_seconds)


def observe_retrieval_debug(*, outcome: str, duration_seconds: float) -> None:
    """Record one `/retrieve/debug` latency observation."""

    RETRIEVAL_DEBUG_DURATION_SECONDS.labels(outcome=outcome).observe(duration_seconds)


def observe_ingest(*, outcome: str, duration_seconds: float) -> None:
    """Record one `/ingest/run` latency observation."""

    INGEST_DURATION_SECONDS.labels(outcome=outcome).observe(duration_seconds)


def observe_eval(*, outcome: str, duration_seconds: float) -> None:
    """Record one `/eval/run` latency observation."""

    EVAL_DURATION_SECONDS.labels(outcome=outcome).observe(duration_seconds)


def increment_query_error(error_type: str) -> None:
    """Increment the query error counter."""

    QUERY_ERRORS_TOTAL.labels(error_type=error_type).inc()


def increment_query_abstention(reason_code: str) -> None:
    """Increment the abstention counter for a successful `/query` response."""

    QUERY_ABSTENTIONS_TOTAL.labels(reason_code=reason_code).inc()


__all__ = [
    "CONTENT_TYPE_LATEST",
    "increment_query_abstention",
    "increment_query_error",
    "observe_eval",
    "observe_http_request",
    "observe_ingest",
    "observe_query",
    "observe_retrieval_debug",
    "render_metrics",
]

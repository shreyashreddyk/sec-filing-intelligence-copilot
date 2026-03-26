from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from fastapi.testclient import TestClient
import requests
import yaml

from sec_copilot.api.app import create_app
from sec_copilot.api.models import IngestRunRequest
from sec_copilot.api.service import ApiSettings, CopilotApiService
from sec_copilot.config import load_retrieval_config
from sec_copilot.eval.offline_runtime import DeterministicEmbeddingAdapter, TokenOverlapReranker
from sec_copilot.frontend.client import ApiBackendError, ApiClient, ApiMalformedResponse, ApiNetworkError
from sec_copilot.schemas.retrieval import QueryRequest


@dataclass
class FakeResponse:
    status_code: int
    payload: object | None = None
    text: str | None = None

    def json(self):
        if self.payload is None:
            raise ValueError("No JSON payload")
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse] | None = None, *, error: Exception | None = None) -> None:
        self.responses = list(responses or [])
        self.error = error

    def request(self, method: str, url: str, json=None, timeout: float = 0.0):
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("No fake responses queued.")
        return self.responses.pop(0)


def _query_request() -> QueryRequest:
    return QueryRequest(
        question="What export control risks does NVIDIA describe?",
        filters={
            "tickers": ["NVDA"],
            "form_types": ["10-K"],
        },
    )


def test_frontend_client_parses_ingest_and_query_success_with_shared_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sec_copilot.ingest.pipeline.SecClient", FakeSecClient)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = ApiSettings(
        service_name="live-client-test",
        data_dir=tmp_path / "data",
        companies_config_path="configs/companies.yaml",
        retrieval_config_path=_write_retrieval_config(tmp_path),
        prompts_config_path="configs/prompts.yaml",
        eval_config_path="configs/eval.yaml",
        strict_coverage=True,
        mock_fallback_when_openai_missing=True,
        default_annual_limit=2,
        default_quarterly_limit=4,
        default_form_types=["10-K", "10-Q"],
    )
    service = _build_service(settings)
    app = create_app(service=service)

    with TestClient(app) as client:
        ingest_payload = client.post(
            "/ingest/run",
            json={
                "companies": ["NVDA"],
                "form_types": ["10-K"],
                "annual_limit": 1,
                "quarterly_limit": 0,
                "index_mode": "rebuild",
                "user_agent": "Shreyash Reddy shreyash@sec-copilot.dev",
            },
        ).json()
        query_payload = client.post("/query", json=_query_request().model_dump(mode="json")).json()

    frontend = ApiClient(
        "http://api.local",
        session=FakeSession(
            [
                FakeResponse(status_code=200, payload=ingest_payload),
                FakeResponse(status_code=200, payload=query_payload),
            ]
        ),
    )
    ingest_result = frontend.ingest_run(
        IngestRunRequest(
            companies=["NVDA"],
            form_types=["10-K"],
            annual_limit=1,
            quarterly_limit=0,
            index_mode="rebuild",
            user_agent="Shreyash Reddy shreyash@sec-copilot.dev",
        )
    )
    result = frontend.query(_query_request())

    assert ingest_result.run_summary.successful_filings == 1
    assert ingest_result.coverage_state.index_status == "fresh"
    assert result.answer.startswith("Mock grounded answer based on")
    assert result.citations


def test_frontend_client_parses_typed_not_ready_and_coverage_errors() -> None:
    not_ready_payload = {
        "error_type": "service_not_ready",
        "message": "Grounded query execution is not ready.",
        "retrieve_ready": False,
        "query_ready": False,
        "index_status": "missing",
        "coverage_status": "uninitialized",
        "indexed_scope": {
            "companies": [],
            "form_types": [],
            "entries": [],
            "document_count": 0,
            "chunk_count": 0,
        },
        "last_ingest_completed_at": None,
        "last_index_refresh_at": None,
        "warnings": ["Processed chunks are present but the Chroma index is missing."],
    }
    coverage_payload = {
        "error_type": "coverage_error",
        "message": "The requested scope is not fully covered by the indexed corpus.",
        "coverage_status": "uncovered",
        "indexed_scope": {
            "companies": ["AMD", "INTC", "NVDA"],
            "form_types": ["10-K"],
            "entries": [],
            "document_count": 6,
            "chunk_count": 57,
        },
        "missing_scope": {
            "tickers": ["QCOM"],
            "form_types": ["10-Q"],
            "pairs": [{"ticker": "QCOM", "form_type": "10-Q"}],
            "date_ranges": [],
        },
        "last_index_refresh_at": "2026-03-26T14:04:21.823901Z",
    }
    frontend = ApiClient(
        "http://api.local",
        session=FakeSession(
            [
                FakeResponse(status_code=503, payload=not_ready_payload),
                FakeResponse(status_code=409, payload=coverage_payload),
            ]
        ),
    )

    not_ready = frontend.query(_query_request())
    coverage = frontend.query(_query_request())

    assert not_ready.error_type == "service_not_ready"
    assert coverage.error_type == "coverage_error"
    assert "QCOM" in coverage.missing_scope.tickers


def test_frontend_client_surfaces_ingest_backend_errors() -> None:
    client = ApiClient(
        "http://api.local",
        session=FakeSession([FakeResponse(status_code=400, payload={"detail": "SEC user agent is required"})]),
    )
    result = client.ingest_run(
        IngestRunRequest(
            companies=["NVDA"],
            form_types=["10-K"],
            annual_limit=1,
            quarterly_limit=0,
            index_mode="rebuild",
        )
    )
    assert isinstance(result, ApiBackendError)
    assert result.status_code == 400
    assert result.raw_body == {"detail": "SEC user agent is required"}


def test_frontend_client_surfaces_malformed_and_network_failures() -> None:
    malformed_client = ApiClient(
        "http://api.local",
        session=FakeSession([FakeResponse(status_code=200, payload={"status": "ok"})]),
    )
    malformed = malformed_client.build_info()
    assert isinstance(malformed, ApiMalformedResponse)
    assert "/build-info" in malformed.endpoint

    network_client = ApiClient(
        "http://api.local",
        session=FakeSession(error=requests.ConnectionError("connection refused")),
    )
    network = network_client.health()
    assert isinstance(network, ApiNetworkError)
    assert "connection refused" in network.message


def test_frontend_client_surfaces_unexpected_backend_errors() -> None:
    client = ApiClient(
        "http://api.local",
        session=FakeSession([FakeResponse(status_code=500, payload={"detail": "boom"})]),
    )
    result = client.health()
    assert isinstance(result, ApiBackendError)
    assert result.status_code == 500


def _fixture_text(relative_path: str) -> str:
    return (Path("tests/fixtures/sec") / relative_path).read_text(encoding="utf-8")


def _fixture_payloads() -> dict[str, str]:
    return {
        "https://data.sec.gov/submissions/CIK0001045810.json": _fixture_text("submissions/CIK0001045810.json"),
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/nvda-20250126x10k.htm": _fixture_text(
            "filings/nvda_10k_primary.html"
        ),
    }


class FakeSecClient:
    def __init__(
        self,
        user_agent: str,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.user_agent = user_agent

    def get_json(self, url: str, cache_path: Path, force_refresh: bool = False):
        payload = _fixture_payloads()[url]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return json.loads(payload)

    def get_text(self, url: str, cache_path: Path, force_refresh: bool = False) -> str:
        payload = _fixture_payloads()[url]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(payload, encoding="utf-8")
        return payload


def _write_retrieval_config(tmp_path: Path) -> str:
    config = load_retrieval_config("configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": "sec_frontend_client_v6",
                    "default_mode": "rebuild",
                }
            ),
            "provider": config.provider.model_copy(update={"default_name": "openai"}),
            "abstention": config.abstention.model_copy(
                update={"weak_top_rerank_score_threshold": 0.5}
            ),
        }
    )
    path = tmp_path / "retrieval.test.yaml"
    path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return str(path)


def _build_service(settings: ApiSettings) -> CopilotApiService:
    return CopilotApiService(
        settings,
        embedding_adapter_factory=lambda cfg: DeterministicEmbeddingAdapter(
            normalize_embeddings=cfg.normalize_embeddings
        ),
        reranker_factory=lambda cfg: TokenOverlapReranker(rerank_top_k=cfg.rerank_top_k),
    )

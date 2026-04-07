from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from sec_copilot.api.app import create_app
from sec_copilot.api.service import ApiSettings, CopilotApiService
from sec_copilot.config import load_retrieval_config
from sec_copilot.eval.offline_runtime import DeterministicEmbeddingAdapter, TokenOverlapReranker


def _fixture_text(relative_path: str) -> str:
    return (Path("tests/fixtures/sec") / relative_path).read_text(encoding="utf-8")


class FakeSecClient:
    def __init__(self, user_agent: str, rate_limit_seconds: float = 1.0, timeout_seconds: float = 30.0, max_retries: int = 3) -> None:
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


def _fixture_payloads() -> dict[str, str]:
    return {
        "https://data.sec.gov/submissions/CIK0001045810.json": _fixture_text("submissions/CIK0001045810.json"),
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/nvda-20250126x10k.htm": _fixture_text(
            "filings/nvda_10k_primary.html"
        ),
    }


def _write_retrieval_config(tmp_path: Path) -> Path:
    config = load_retrieval_config("configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": "sec_api_test_v5",
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
    return path


def _build_service(tmp_path: Path, *, data_dir: Path) -> CopilotApiService:
    settings = ApiSettings(
        data_dir=data_dir,
        retrieval_config_path=_write_retrieval_config(tmp_path),
        prompts_config_path=Path("configs/prompts.yaml"),
        eval_config_path=Path("configs/eval.yaml"),
        companies_config_path=Path("configs/companies.yaml"),
    )
    return CopilotApiService(
        settings,
        embedding_adapter_factory=lambda cfg: DeterministicEmbeddingAdapter(
            normalize_embeddings=cfg.normalize_embeddings
        ),
        reranker_factory=lambda cfg: TokenOverlapReranker(rerank_top_k=cfg.rerank_top_k),
    )


def test_openapi_exposes_v5_routes(tmp_path: Path) -> None:
    service = _build_service(tmp_path, data_dir=tmp_path / "data")
    app = create_app(service, include_admin_routes=True)

    with TestClient(app) as client:
        payload = client.get("/openapi.json").json()

    paths = payload["paths"]
    assert "/health" in paths
    assert "/readyz" in paths
    assert "/build-info" in paths
    assert "/query" in paths
    assert "/retrieve/debug" in paths
    assert "/ingest/run" in paths
    assert "/eval/run" in paths


def test_service_starts_not_ready_and_query_returns_typed_503(tmp_path: Path) -> None:
    service = _build_service(tmp_path, data_dir=tmp_path / "data")
    app = create_app(service, include_admin_routes=True)

    with TestClient(app) as client:
        health_payload = client.get("/health").json()
        readiness_response = client.get("/readyz")
        build_info_payload = client.get("/build-info").json()
        query_response = client.post(
            "/query",
            json={"question": "What export control risks does NVIDIA describe?", "filters": {"tickers": ["NVDA"]}},
        )

    assert health_payload["retrieve_ready"] is False
    assert health_payload["query_ready"] is False
    assert health_payload["index_status"] == "missing"
    assert readiness_response.status_code == 503
    assert readiness_response.json()["status"] == "not_ready"
    assert build_info_payload["coverage_status"] == "uninitialized"
    assert query_response.status_code == 503
    assert query_response.json()["error_type"] == "service_not_ready"


def test_ingest_bootstrap_updates_readiness_and_query_and_debug_endpoints(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SEC_USER_AGENT", "Shreyash Reddy shreyash@sec-copilot.dev")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sec_copilot.ingest.pipeline.SecClient", FakeSecClient)

    service = _build_service(tmp_path, data_dir=data_dir)
    app = create_app(service, include_admin_routes=True)

    with TestClient(app) as client:
        ingest_response = client.post(
            "/ingest/run",
            json={
                "companies": ["NVDA"],
                "form_types": ["10-K"],
                "annual_limit": 1,
                "quarterly_limit": 0,
                "index_mode": "rebuild",
            },
        )
        health_payload = client.get("/health").json()
        readiness_response = client.get("/readyz")
        query_response = client.post(
            "/query",
            json={
                "question": "What export control risks does NVIDIA describe?",
                "filters": {"tickers": ["NVDA"], "form_types": ["10-K"]},
            },
        )
        debug_response = client.post(
            "/retrieve/debug",
            json={
                "question": "What export control risks does NVIDIA describe?",
                "filters": {"tickers": ["NVDA"], "form_types": ["10-K"]},
            },
        )
        coverage_failure = client.post(
            "/query",
            json={
                "question": "What manufacturing strategy does AMD describe?",
                "filters": {"tickers": ["AMD"], "form_types": ["10-K"]},
            },
        )

    ingest_payload = ingest_response.json()
    query_payload = query_response.json()
    debug_payload = debug_response.json()

    assert ingest_response.status_code == 200
    assert ingest_payload["index_build"]["collection_mode"] == "rebuild"
    assert ingest_payload["coverage_state"]["index_status"] == "fresh"
    assert health_payload["retrieve_ready"] is True
    assert health_payload["query_ready"] is True
    assert readiness_response.status_code == 200
    assert readiness_response.json()["status"] == "ready"

    assert query_response.status_code == 200
    assert query_payload["coverage_status"] == "covered"
    assert query_payload["citations"]
    assert query_payload["retrieved_chunks"]
    assert query_payload["timings"]["total_ms"] >= 0.0

    assert debug_response.status_code == 200
    assert debug_payload["coverage_status"] == "covered"
    assert debug_payload["stage_counts"]["filtered_parent_count"] > 0
    assert debug_payload["timings"]["dense_ms"] >= 0.0

    assert coverage_failure.status_code == 409
    assert coverage_failure.json()["error_type"] == "coverage_error"


def test_eval_run_endpoint_smoke(tmp_path: Path) -> None:
    service = _build_service(tmp_path, data_dir=tmp_path / "data")
    app = create_app(service, include_admin_routes=True)

    with TestClient(app) as client:
        response = client.post(
            "/eval/run",
            json={
                "subset": "ci_smoke",
                "mode": "full",
                "provider": "reference",
                "score_backend": "deterministic",
                "output_dir": str(tmp_path / "eval_output"),
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["result"]["schema_version"] == "sec_eval_results.v1"
    assert payload["result"]["subset"] == "ci_smoke"
    assert payload["timings"]["total_ms"] >= 0.0


def test_metrics_endpoint_exposes_custom_metric_families(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("SEC_USER_AGENT", "Shreyash Reddy shreyash@sec-copilot.dev")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sec_copilot.ingest.pipeline.SecClient", FakeSecClient)

    service = _build_service(tmp_path, data_dir=data_dir)
    app = create_app(service, include_admin_routes=True)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"

        not_ready_query = client.post(
            "/query",
            json={"question": "What export control risks does NVIDIA describe?", "filters": {"tickers": ["NVDA"]}},
        )
        assert not_ready_query.status_code == 503

        ingest_response = client.post(
            "/ingest/run",
            json={
                "companies": ["NVDA"],
                "form_types": ["10-K"],
                "annual_limit": 1,
                "quarterly_limit": 0,
                "index_mode": "rebuild",
            },
        )
        assert ingest_response.status_code == 200

        ready_response = client.get("/readyz")
        assert ready_response.status_code == 200
        assert ready_response.json()["status"] == "ready"

        debug_response = client.post(
            "/retrieve/debug",
            json={
                "question": "What export control risks does NVIDIA describe?",
                "filters": {"tickers": ["NVDA"], "form_types": ["10-K"]},
            },
        )
        assert debug_response.status_code == 200

        abstained_query = client.post(
            "/query",
            json={
                "question": "How many penguins are mentioned in the filing?",
                "filters": {"tickers": ["NVDA"], "form_types": ["10-K"]},
            },
        )
        assert abstained_query.status_code == 200
        assert abstained_query.json()["abstained"] is True

        eval_response = client.post(
            "/eval/run",
            json={
                "subset": "ci_smoke",
                "mode": "full",
                "provider": "reference",
                "score_backend": "deterministic",
                "output_dir": str(tmp_path / "eval_output"),
            },
        )
        assert eval_response.status_code == 200

        metrics_response = client.get("/metrics")

    metrics_text = metrics_response.text
    assert metrics_response.status_code == 200
    assert metrics_response.headers["content-type"].startswith("text/plain")
    assert "# HELP sec_copilot_http_requests_total" in metrics_text
    assert "# TYPE sec_copilot_http_request_duration_seconds histogram" in metrics_text
    assert "# TYPE sec_copilot_query_duration_seconds histogram" in metrics_text
    assert "# TYPE sec_copilot_retrieval_debug_duration_seconds histogram" in metrics_text
    assert "# TYPE sec_copilot_ingest_duration_seconds histogram" in metrics_text
    assert "# TYPE sec_copilot_eval_duration_seconds histogram" in metrics_text
    assert 'sec_copilot_query_errors_total{error_type="service_not_ready"}' in metrics_text
    assert 'sec_copilot_query_abstentions_total{reason_code="' in metrics_text
    assert 'path="/metrics"' not in metrics_text


def test_public_app_excludes_admin_routes(tmp_path: Path) -> None:
    service = _build_service(tmp_path, data_dir=tmp_path / "data")
    app = create_app(service, include_admin_routes=False)

    with TestClient(app) as client:
        payload = client.get("/openapi.json").json()
        health_response = client.get("/health")
        readiness_response = client.get("/readyz")
        ingest_response = client.post("/ingest/run", json={})
        eval_response = client.post("/eval/run", json={})

    paths = payload["paths"]
    assert "/health" in paths
    assert "/readyz" in paths
    assert "/build-info" in paths
    assert "/query" in paths
    assert "/retrieve/debug" in paths
    assert "/ingest/run" not in paths
    assert "/eval/run" not in paths
    assert health_response.status_code == 200
    assert readiness_response.status_code == 503
    assert ingest_response.status_code == 404
    assert eval_response.status_code == 404

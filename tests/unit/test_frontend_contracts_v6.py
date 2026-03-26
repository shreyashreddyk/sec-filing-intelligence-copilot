from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from sec_copilot.api.app import create_app
from sec_copilot.api.service import ApiSettings, CopilotApiService
from sec_copilot.config import load_retrieval_config
from sec_copilot.eval.offline_runtime import DeterministicEmbeddingAdapter, TokenOverlapReranker


def test_frontend_consumed_routes_are_present_in_openapi(tmp_path) -> None:
    config = load_retrieval_config("configs/retrieval.yaml")
    config = config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(tmp_path / "chroma"),
                    "collection_name": "sec_frontend_contract_v6",
                    "default_mode": "rebuild",
                }
            )
        }
    )
    retrieval_config_path = tmp_path / "retrieval.test.yaml"
    retrieval_config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    service = _build_service(
        ApiSettings(
            service_name="frontend-contract-test",
            data_dir=tmp_path / "data",
            companies_config_path="configs/companies.yaml",
            retrieval_config_path=retrieval_config_path,
            prompts_config_path="configs/prompts.yaml",
            eval_config_path="configs/eval.yaml",
            strict_coverage=True,
            mock_fallback_when_openai_missing=True,
            default_annual_limit=2,
            default_quarterly_limit=4,
            default_form_types=["10-K", "10-Q"],
        )
    )
    app = create_app(service=service)

    with TestClient(app) as client:
        payload = client.get("/openapi.json").json()

    paths = payload["paths"]
    assert "/health" in paths
    assert "/build-info" in paths
    assert "/ingest/run" in paths
    assert "/query" in paths
    assert "/retrieve/debug" in paths

    assert "200" in paths["/ingest/run"]["post"]["responses"]
    assert "200" in paths["/query"]["post"]["responses"]
    assert "409" in paths["/query"]["post"]["responses"]
    assert "503" in paths["/query"]["post"]["responses"]
    assert "200" in paths["/retrieve/debug"]["post"]["responses"]
    assert "409" in paths["/retrieve/debug"]["post"]["responses"]
    assert "503" in paths["/retrieve/debug"]["post"]["responses"]


def _build_service(settings: ApiSettings) -> CopilotApiService:
    return CopilotApiService(
        settings,
        embedding_adapter_factory=lambda cfg: DeterministicEmbeddingAdapter(
            normalize_embeddings=cfg.normalize_embeddings
        ),
        reranker_factory=lambda cfg: TokenOverlapReranker(rerank_top_k=cfg.rerank_top_k),
    )

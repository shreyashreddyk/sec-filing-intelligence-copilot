from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.eval.runner import run_eval
from sec_copilot.eval.schemas import EvalConfig, EvalDataset, EvalExample, EvalRagasConfig, EvalThresholdGroups
from sec_copilot.schemas.retrieval import Citation, QueryResponse, RetrievedChunk, RetrievalFilters, RetrievalStageCounts


def test_run_eval_passes_dedicated_ragas_config(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        document_id="doc_1",
        ticker="NVDA",
        company_name="NVIDIA",
        form_type="10-K",
        filing_date="2026-02-25",
        accession_number="0001045810-26-000021",
        section_title="Risk Factors",
        source_url="https://example.com",
        text="NVIDIA described export controls affecting China.",
        dense_rank=1,
        bm25_rank=1,
        rrf_score=1.0,
        rerank_rank=1,
        rerank_score=0.95,
    )
    stage_counts = RetrievalStageCounts(
        filtered_parent_count=1,
        dense_subchunk_hit_count=1,
        dense_parent_candidate_count=1,
        bm25_candidate_count=1,
        fused_candidate_count=1,
        reranked_candidate_count=1,
    )
    retrieval_outcome = SimpleNamespace(
        reason_code="ok",
        retrieved_chunks=[chunk],
        stage_counts=stage_counts,
        reranker_applied=True,
        reranker_skipped_reason=None,
    )

    class FakeStore(dict):
        def get(self, key, default=None):
            return super().get(key, default)

        def values(self):
            return super().values()

    store = FakeStore({"chunk_1": SimpleNamespace(document_id="doc_1", text=chunk.text)})

    class FakeRetriever:
        def retrieve(self, request):
            return retrieval_outcome

    class FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def answer_with_trace(self, request, retrieval):
            response = QueryResponse(
                answer="NVIDIA describes export-control risks tied to China.",
                citations=[
                    Citation(
                        citation_id="chunk_1",
                        ticker="NVDA",
                        form_type="10-K",
                        filing_date="2026-02-25",
                        accession_number="0001045810-26-000021",
                        section_title="Risk Factors",
                        source_url="https://example.com",
                        snippet="export controls affecting China",
                    )
                ],
                abstained=False,
                retrieved_chunks=[chunk],
                reason_code="ok",
            )
            prompt = SimpleNamespace(
                prompt_name="grounded_answer_v3",
                prompt_version="v3",
                context_chunk_ids=["chunk_1"],
                truncated_chunk_ids=[],
                used_context_tokens=42,
            )
            return response, retrieval_outcome, prompt

    monkeypatch.setattr("sec_copilot.eval.runner.build_offline_eval_runtime", lambda **kwargs: SimpleNamespace(
        config=kwargs["config"],
        retriever=FakeRetriever(),
        prompt_builder=object(),
        store=store,
    ))
    monkeypatch.setattr("sec_copilot.eval.runner.ProcessedChunkStore.load", lambda path: store)
    monkeypatch.setattr("sec_copilot.eval.runner.build_eval_provider", lambda *args, **kwargs: SimpleNamespace(name="openai"))
    monkeypatch.setattr("sec_copilot.eval.runner.GroundedAnswerPipeline", FakePipeline)
    def fake_score_with_ragas(rows, *, ragas_config, api_key, score_backend):
        captured["call"] = {
            "rows": rows,
            "ragas_config": ragas_config,
            "api_key": api_key,
            "score_backend": score_backend,
        }
        return [{"ragas_faithfulness": 0.9, "ragas_response_relevancy": 0.8, "ragas_context_precision": 0.7}]

    monkeypatch.setattr("sec_copilot.eval.runner.score_with_ragas", fake_score_with_ragas)

    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text("dataset: stub\n", encoding="utf-8")
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()

    result = run_eval(
        eval_config=EvalConfig(
            schema_version="sec_eval_config.v1",
            dataset_path=str(dataset_path),
            corpus_path=str(corpus_path),
            default_subset="full",
            default_mode="full",
            default_provider="reference",
            default_score_backend="deterministic",
            output_root=str(tmp_path / "artifacts"),
            ragas=EvalRagasConfig(model_name="gpt-4.1-mini", answer_relevancy_strictness=1),
            thresholds=EvalThresholdGroups(),
        ),
        retrieval_config=load_retrieval_config("configs/retrieval.yaml"),
        prompt_catalog=load_prompt_catalog("configs/prompts.yaml"),
        dataset=EvalDataset(
            schema_version="sec_eval_dataset.v1",
            dataset_name="test",
            examples=[
                EvalExample(
                    example_id="fact_nvda_export_controls",
                    question="What export control risks does NVIDIA describe?",
                    category="fact_lookup",
                    filters=RetrievalFilters(tickers=["NVDA"], form_types=["10-K"]),
                    expected_abstention=False,
                    gold_chunk_ids=["chunk_1"],
                    required_citation_chunk_ids=["chunk_1"],
                    reference_answer="NVIDIA describes export-control risks tied to China.",
                    reference_key_points=["export-control risks", "China"],
                    tags=["full"],
                )
            ],
        ),
        dataset_path=dataset_path,
        corpus_path=corpus_path,
        ragas_config=EvalRagasConfig(model_name="gpt-4.1-mini", answer_relevancy_strictness=1),
        subset="full",
        mode="full",
        provider="openai",
        score_backend="both",
        output_dir=tmp_path / "output",
    )

    call = captured["call"]
    assert call["ragas_config"].model_name == "gpt-4.1-mini"
    assert call["ragas_config"].answer_relevancy_strictness == 1
    assert call["score_backend"] == "both"
    assert result.config["ragas"]["model_name"] == "gpt-4.1-mini"

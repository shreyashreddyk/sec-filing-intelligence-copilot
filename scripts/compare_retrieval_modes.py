"""Compare dense-only, hybrid, and reranked retrieval on curated local examples."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path

from sec_copilot.eval.curated_examples import (
    CuratedEmbeddingAdapter,
    CuratedReranker,
    build_curated_config,
    build_curated_store,
    curated_examples,
)
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptManager
from sec_copilot.generation.providers import MockLLMProvider
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas import QueryRequest, RetrievalFilters, RetrievalResponse, RetrievalStageCounts
from sec_copilot.config import load_prompt_catalog


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    store = build_curated_store()
    config = build_curated_config(output_dir)
    adapter = CuratedEmbeddingAdapter()
    index_manager = ChromaIndexManager(config, adapter)
    index_manager.build(store, mode="rebuild")
    dense_retriever = DenseRetriever(config, adapter, store, index_manager.get_collection())

    prompt_catalog = load_prompt_catalog("configs/prompts.yaml")
    prompt_template = PromptManager(prompt_catalog).get_prompt(
        config.prompting.prompt_name,
        expected_version=config.prompting.prompt_version,
    )
    prompt_builder = GroundedPromptBuilder(config.retrieval, config.prompting, prompt_template)

    examples = list(curated_examples())
    runs = []
    for example in examples:
        filters = RetrievalFilters(tickers=list(example.tickers), form_types=list(example.form_types))
        request = QueryRequest(question=example.query, filters=filters)
        runs.append(
            _dense_run(
                dense_retriever=dense_retriever,
                request=request,
                example_name=example.name,
            )
        )
        runs.append(
            _hybrid_run(
                config=config.model_copy(update={"reranking": config.reranking.model_copy(update={"enabled": False})}),
                store=store,
                dense_retriever=dense_retriever,
                request=request,
                example_name=example.name,
                include_generation=False,
                prompt_builder=prompt_builder,
            )
        )
        runs.append(
            _hybrid_run(
                config=config,
                store=store,
                dense_retriever=dense_retriever,
                request=request,
                example_name=example.name,
                include_generation=args.include_generation,
                prompt_builder=prompt_builder,
            )
        )

    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    csv_path = output_dir / "comparison.csv"

    json_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(runs), encoding="utf-8")
    if args.write_csv:
        csv_path.write_text(_render_csv(runs), encoding="utf-8")

    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved comparison Markdown: {markdown_path}")
    if args.write_csv:
        print(f"Saved comparison CSV: {csv_path}")
    return 0


def _dense_run(*, dense_retriever: DenseRetriever, request: QueryRequest, example_name: str) -> dict:
    dense_result = dense_retriever.retrieve(request.question, request.filters)
    response = RetrievalResponse(
        reason_code="ok" if dense_result.results else "no_hits",
        retrieved_chunks=list(dense_result.results),
        stage_counts=RetrievalStageCounts(
            filtered_parent_count=len(dense_retriever.store.filtered_values(request.filters)),
            dense_subchunk_hit_count=dense_result.subchunk_hit_count,
            dense_parent_candidate_count=dense_result.parent_candidate_count,
            bm25_candidate_count=0,
            fused_candidate_count=len(dense_result.results),
            reranked_candidate_count=0,
        ),
        reranker_applied=False,
        reranker_skipped_reason="dense_only_mode",
    )
    return {
        "example_name": example_name,
        "query_text": request.question,
        "filters": request.filters.model_dump(mode="json"),
        "retrieval_mode": "dense",
        "reason_code": response.reason_code,
        "retrieved_chunks": response.model_dump(mode="json")["retrieved_chunks"],
        "stage_counts": response.stage_counts.model_dump(mode="json"),
        "reranker_applied": response.reranker_applied,
        "reranker_skipped_reason": response.reranker_skipped_reason,
        "answer": None,
        "abstained": None,
        "citations": [],
    }


def _hybrid_run(
    *,
    config,
    store,
    dense_retriever: DenseRetriever,
    request: QueryRequest,
    example_name: str,
    include_generation: bool,
    prompt_builder: GroundedPromptBuilder,
) -> dict:
    reranker = CuratedReranker() if config.reranking.enabled else None
    hybrid_retriever = HybridRetriever(config, store, dense_retriever, BM25Retriever(store), reranker)
    retrieval_response = hybrid_retriever.retrieve(request).to_response()

    answer = None
    abstained = None
    citations: list[dict] = []
    if include_generation and config.reranking.enabled:
        pipeline = GroundedAnswerPipeline(config, hybrid_retriever, prompt_builder, MockLLMProvider())
        answer_response = pipeline.answer(request)
        answer = answer_response.answer
        abstained = answer_response.abstained
        citations = [citation.model_dump(mode="json") for citation in answer_response.citations]

    return {
        "example_name": example_name,
        "query_text": request.question,
        "filters": request.filters.model_dump(mode="json"),
        "retrieval_mode": "reranked" if config.reranking.enabled else "hybrid",
        "reason_code": retrieval_response.reason_code,
        "retrieved_chunks": retrieval_response.model_dump(mode="json")["retrieved_chunks"],
        "stage_counts": retrieval_response.stage_counts.model_dump(mode="json"),
        "reranker_applied": retrieval_response.reranker_applied,
        "reranker_skipped_reason": retrieval_response.reranker_skipped_reason,
        "answer": answer,
        "abstained": abstained,
        "citations": citations,
    }


def _render_markdown(runs: list[dict]) -> str:
    lines = ["# Retrieval Comparison Summary", ""]
    for run in runs:
        lines.append(f"## {run['example_name']} - {run['retrieval_mode']}")
        lines.append(f"- query: `{run['query_text']}`")
        lines.append(f"- reason_code: `{run['reason_code']}`")
        lines.append(f"- reranker_applied: `{run['reranker_applied']}`")
        top_chunks = ", ".join(chunk["chunk_id"] for chunk in run["retrieved_chunks"][:3]) or "none"
        lines.append(f"- top_chunks: {top_chunks}")
        if run["answer"] is not None:
            lines.append(f"- answer: {run['answer']}")
            lines.append(f"- abstained: `{run['abstained']}`")
        lines.append("")
    return "\n".join(lines)


def _render_csv(runs: list[dict]) -> str:
    lines = ["example_name,retrieval_mode,reason_code,top_chunk,abstained"]
    for run in runs:
        top_chunk = run["retrieved_chunks"][0]["chunk_id"] if run["retrieved_chunks"] else ""
        abstained = "" if run["abstained"] is None else str(run["abstained"]).lower()
        lines.append(
            f"{run['example_name']},{run['retrieval_mode']},{run['reason_code']},{top_chunk},{abstained}"
        )
    return "\n".join(lines) + "\n"


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "comparisons" / f"retrieval_modes_{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare dense, hybrid, and reranked retrieval on curated examples.")
    parser.add_argument("--output-dir", help="Optional output directory for comparison artifacts.")
    parser.add_argument(
        "--include-generation",
        action="store_true",
        help="Include mock grounded-answer generation for reranked runs.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Also save a CSV summary alongside JSON and Markdown.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

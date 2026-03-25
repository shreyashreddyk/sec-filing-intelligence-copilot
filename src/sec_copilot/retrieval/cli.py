"""CLI for V3 indexing, retrieval inspection, and grounded answers."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptManager
from sec_copilot.generation.providers import MockLLMProvider, OpenAILLMProvider
from sec_copilot.rerank.cross_encoder import CrossEncoderReranker
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import SentenceTransformerEmbeddingAdapter
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas.retrieval import QueryRequest, RetrievalFilters


def main(argv: list[str] | None = None) -> int:
    """Run the V3 retrieval CLI."""

    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2

    config = load_retrieval_config(args.retrieval_config)
    if args.persist_directory:
        config = config.model_copy(
            update={"index": config.index.model_copy(update={"persist_directory": args.persist_directory})}
        )
    if args.collection_name:
        config = config.model_copy(
            update={"index": config.index.model_copy(update={"collection_name": args.collection_name})}
        )

    if args.command == "index":
        return _run_index(config, args)

    prompt_catalog = load_prompt_catalog(args.prompts_config)
    prompt_manager = PromptManager(prompt_catalog)
    prompt_template = prompt_manager.get_prompt(
        config.prompting.prompt_name,
        expected_version=config.prompting.prompt_version,
    )
    store = ProcessedChunkStore.load(args.data_dir)
    adapter = SentenceTransformerEmbeddingAdapter(config.embedding)
    index_manager = ChromaIndexManager(config, adapter)
    collection = index_manager.get_collection()
    dense_retriever = DenseRetriever(config, adapter, store, collection)
    bm25_retriever = BM25Retriever(store)
    reranker = CrossEncoderReranker(config.reranking) if config.reranking.enabled else None
    hybrid_retriever = HybridRetriever(config, store, dense_retriever, bm25_retriever, reranker)

    if args.command == "retrieve":
        return _run_retrieve(hybrid_retriever, args)
    if args.command == "answer":
        prompt_builder = GroundedPromptBuilder(config.retrieval, config.prompting, prompt_template)
        provider = _build_provider(config, args)
        pipeline = GroundedAnswerPipeline(config, hybrid_retriever, prompt_builder, provider)
        return _run_answer(pipeline, args)

    parser.print_help()
    return 2


def _run_index(config, args: argparse.Namespace) -> int:
    store = ProcessedChunkStore.load(args.data_dir)
    if len(store) == 0:
        print("No processed chunks found under the supplied data directory.", file=sys.stderr)
        return 1
    adapter = SentenceTransformerEmbeddingAdapter(config.embedding)
    index_manager = ChromaIndexManager(config, adapter)
    result = index_manager.build(store=store, mode=args.mode)
    print(result.model_dump_json(indent=2))
    if result.stale_id_count:
        print("Warning: stale vector IDs remain in the collection. Run rebuild to prune them.", file=sys.stderr)
    return 0


def _run_retrieve(retriever: HybridRetriever, args: argparse.Namespace) -> int:
    request = _build_request(args)
    response = retriever.retrieve(request).to_response()
    if request.debug:
        _print_retrieved_chunks(response.retrieved_chunks)
        _print_stage_counts(response.stage_counts, response.reranker_applied, response.reranker_skipped_reason)
    print(response.model_dump_json(indent=2))
    return 0


def _run_answer(pipeline: GroundedAnswerPipeline, args: argparse.Namespace) -> int:
    request = _build_request(args)
    response = pipeline.answer(request)
    if request.debug:
        _print_retrieved_chunks(response.retrieved_chunks)
    print(response.model_dump_json(indent=2))
    return 0


def _build_request(args: argparse.Namespace) -> QueryRequest:
    return QueryRequest(
        question=args.question,
        filters=RetrievalFilters(
            tickers=args.ticker or [],
            form_types=args.form_type or [],
            filing_date_from=args.date_from,
            filing_date_to=args.date_to,
        ),
        debug=args.debug,
    )


def _build_provider(config, args: argparse.Namespace):
    provider_name = args.provider or config.provider.default_name
    if provider_name == "mock":
        return MockLLMProvider()
    model_name = args.provider_model or os.getenv("SEC_COPILOT_OPENAI_MODEL", config.provider.openai_model)
    return OpenAILLMProvider(model_name=model_name)


def _print_retrieved_chunks(results) -> None:
    print("Retrieved parent chunks:")
    for index, chunk in enumerate(results, start=1):
        print(
            f"rank={index} "
            f"chunk_id={chunk.chunk_id} "
            f"dense_rank={chunk.dense_rank} dense_score={_fmt(chunk.dense_score)} "
            f"bm25_rank={chunk.bm25_rank} bm25_score={_fmt(chunk.bm25_score)} "
            f"rrf_score={_fmt(chunk.rrf_score)} "
            f"rerank_rank={chunk.rerank_rank} rerank_score={_fmt(chunk.rerank_score)} "
            f"section_title={chunk.section_title}"
        )


def _print_stage_counts(stage_counts, reranker_applied: bool, reranker_skipped_reason: str | None) -> None:
    print(
        "Stage counts:",
        stage_counts.model_dump(mode="json"),
        f"reranker_applied={reranker_applied}",
        f"reranker_skipped_reason={reranker_skipped_reason}",
    )


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid retrieval and grounded answer CLI.")
    parser.add_argument(
        "--retrieval-config",
        default="configs/retrieval.yaml",
        help="Path to retrieval config YAML.",
    )
    parser.add_argument(
        "--prompts-config",
        default="configs/prompts.yaml",
        help="Path to prompts config YAML.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.getenv("SEC_COPILOT_DATA_DIR", "data"),
        help="Base data directory containing processed chunk artifacts.",
    )
    parser.add_argument(
        "--persist-directory",
        default=os.getenv("SEC_COPILOT_CHROMA_DIR"),
        help="Optional override for the local Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection-name",
        help="Optional override for the Chroma collection name.",
    )

    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build or update the dense Chroma index.")
    index_parser.add_argument("--mode", choices=("rebuild", "upsert"), default=None, help="Index lifecycle mode.")

    retrieve_parser = subparsers.add_parser("retrieve", help="Run hybrid retrieval only.")
    _add_query_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser("answer", help="Run hybrid retrieval plus grounded answer generation.")
    _add_query_arguments(answer_parser)
    answer_parser.add_argument("--provider", choices=("mock", "openai"), help="Provider backend for answer generation.")
    answer_parser.add_argument("--provider-model", help="Optional provider model override.")

    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--question", required=True, help="Natural-language query over the indexed corpus.")
    parser.add_argument("--ticker", action="append", help="Ticker filter. Repeat for multiple values.")
    parser.add_argument("--form-type", action="append", help="Form type filter. Repeat for multiple values.")
    parser.add_argument("--date-from", help="Inclusive filing-date lower bound in YYYY-MM-DD format.")
    parser.add_argument("--date-to", help="Inclusive filing-date upper bound in YYYY-MM-DD format.")
    parser.add_argument("--debug", action="store_true", help="Print retrieval debug output.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

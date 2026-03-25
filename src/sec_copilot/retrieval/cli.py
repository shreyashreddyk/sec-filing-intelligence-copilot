"""CLI for V2 dense indexing, retrieval inspection, and grounded answers."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder
from sec_copilot.generation.providers import MockLLMProvider, OpenAILLMProvider
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import SentenceTransformerEmbeddingAdapter
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas.retrieval import QueryRequest, RetrievalFilters


def main(argv: list[str] | None = None) -> int:
    """Run the V2 retrieval CLI."""

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
    prompt_template = prompt_catalog.get("grounded_answer_baseline")
    store = ProcessedChunkStore.load(args.data_dir)
    adapter = SentenceTransformerEmbeddingAdapter(config.embedding)
    index_manager = ChromaIndexManager(config, adapter)
    collection = index_manager.get_collection()
    retriever = DenseRetriever(config, adapter, store, collection)

    if args.command == "retrieve":
        return _run_retrieve(retriever, args)
    if args.command == "answer":
        prompt_builder = GroundedPromptBuilder(config.prompting, prompt_template)
        provider = _build_provider(config, args)
        pipeline = GroundedAnswerPipeline(retriever, prompt_builder, provider)
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


def _run_retrieve(retriever: DenseRetriever, args: argparse.Namespace) -> int:
    request = _build_request(args)
    outcome = retriever.retrieve(request)
    if request.debug:
        _print_debug_results(outcome.debug.results)
    return 0 if outcome.status in {"ok", "weak_results"} else 1


def _run_answer(pipeline: GroundedAnswerPipeline, args: argparse.Namespace) -> int:
    request = _build_request(args)
    response = pipeline.answer(request)
    if request.debug and response.retrieval_debug is not None:
        _print_debug_results(response.retrieval_debug.results)
    print(response.model_dump_json(indent=2))
    return 0 if response.status in {"ok", "weak_results", "no_results"} else 1


def _build_request(args: argparse.Namespace) -> QueryRequest:
    return QueryRequest(
        question=args.question,
        filters=RetrievalFilters(
            tickers=args.ticker or [],
            form_types=args.form_type or [],
            filing_date_from=args.date_from,
            filing_date_to=args.date_to,
        ),
        retrieval_top_k=args.top_k,
        prompt_top_n=args.prompt_top_n,
        debug=args.debug,
    )


def _build_provider(config, args: argparse.Namespace):
    provider_name = args.provider or config.provider.default_name
    if provider_name == "mock":
        return MockLLMProvider()
    model_name = args.provider_model or os.getenv("SEC_COPILOT_OPENAI_MODEL", config.provider.openai_model)
    return OpenAILLMProvider(model_name=model_name)


def _print_debug_results(results) -> None:
    print("Dense retrieval debug results (pre-LLM score = 1.0 - cosine distance):")
    for chunk in results:
        print(
            f"rank={chunk.rank} "
            f"chunk_id={chunk.chunk_id} "
            f"score={chunk.score:.4f} "
            f"ticker={chunk.ticker} "
            f"form_type={chunk.form_type} "
            f"filing_date={chunk.filing_date.isoformat()} "
            f"section_title={chunk.section_title} "
            f"source_url={chunk.source_url}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense retrieval and grounded answer CLI.")
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

    retrieve_parser = subparsers.add_parser("retrieve", help="Run dense retrieval only.")
    _add_query_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser("answer", help="Run dense retrieval plus grounded answer generation.")
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
    parser.add_argument("--top-k", type=int, help="Override the parent retrieval top-k.")
    parser.add_argument("--prompt-top-n", type=int, help="Override the prompt context top-n when answering.")
    parser.add_argument("--debug", action="store_true", help="Print dense retrieval debug output.")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

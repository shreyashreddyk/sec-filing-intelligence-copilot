"""Small device-aware walkthrough for the V2 dense retrieval pipeline."""

from __future__ import annotations

import argparse
import platform
import sys

import chromadb
import numpy
import pydantic
import sentence_transformers
import torch
import transformers

from sec_copilot.config import load_prompt_catalog, load_retrieval_config
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder
from sec_copilot.generation.providers import MockLLMProvider
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import SentenceTransformerEmbeddingAdapter
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas.retrieval import QueryRequest, RetrievalFilters


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    print("=" * 80)
    print("SEC Filing Intelligence Copilot V2 Device-Aware Walkthrough")
    print("=" * 80)
    _print_step_header(1, "Environment")
    print(f"[OK] Python: {sys.version.split()[0]}")
    print(f"[OK] Platform: {platform.platform()}")
    print(f"[OK] numpy: {numpy.__version__}")
    print(f"[OK] torch: {torch.__version__}")
    print(f"[OK] transformers: {transformers.__version__}")
    print(f"[OK] sentence-transformers: {sentence_transformers.__version__}")
    print(f"[OK] chromadb: {chromadb.__version__}")
    print(f"[OK] pydantic: {pydantic.__version__}")

    _print_step_header(2, "Config and Corpus")
    config = load_retrieval_config(args.retrieval_config)
    if args.device:
        config = config.model_copy(
            update={"embedding": config.embedding.model_copy(update={"device": args.device})}
        )
    prompt_catalog = load_prompt_catalog(args.prompts_config)
    prompt_template = prompt_catalog.get("grounded_answer_baseline")
    config = config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "collection_name": args.collection_name,
                    "persist_directory": args.persist_directory,
                }
            )
        }
    )
    store = ProcessedChunkStore.load(args.data_dir)
    if len(store) == 0:
        print("[FAIL] No processed chunks found under the supplied data directory.")
        return 1

    ticker_chunks = [
        chunk
        for chunk in store.values()
        if chunk.ticker.upper() == args.ticker.upper()
    ]
    non_cover_chunks = [chunk for chunk in ticker_chunks if chunk.section_title.strip().lower() != "cover page"]
    source_chunks = non_cover_chunks or ticker_chunks
    selected_chunks = _evenly_sample(source_chunks, args.max_chunks)
    if not selected_chunks:
        print(f"[FAIL] No chunks found for ticker={args.ticker!r}.")
        return 1

    print(f"[OK] Loaded {len(store)} parent chunks from {args.data_dir}/processed/chunks")
    print(f"[OK] Selected {len(selected_chunks)} {args.ticker.upper()} chunks for the walkthrough")
    for chunk in selected_chunks[:3]:
        print(f"  - {chunk.chunk_id} | {chunk.form_type} | {chunk.section_title}")

    _print_step_header(3, "Embedding Adapter and Subchunks")
    adapter = SentenceTransformerEmbeddingAdapter(config.embedding)
    print(f"[OK] Model: {adapter.model_name}")
    print(f"[OK] Requested device: {adapter.requested_device}")
    print(f"[OK] Resolved device: {adapter.resolved_device}")
    print(f"[OK] Embedding dimension: {adapter.embedding_dimension}")
    print(f"[OK] Max sequence length: {adapter.max_seq_length}")
    sample_subchunks = adapter.build_subchunks(selected_chunks[0])
    if not sample_subchunks:
        print("[FAIL] No embedding subchunks were produced for the first chunk.")
        return 1
    print(f"[OK] First parent chunk produced {len(sample_subchunks)} embedding subchunk(s)")
    print(f"[OK] Sample subchunk ID: {sample_subchunks[0].subchunk_id}")
    print("Sample subchunk text preview:")
    print(_preview(sample_subchunks[0].text))

    sample_embeddings = adapter.embed_texts([subchunk.text for subchunk in sample_subchunks[:2]])
    print(f"[OK] Sample embedding matrix shape: ({len(sample_embeddings)}, {len(sample_embeddings[0])})")
    print("First embedding vector preview (first 8 dims):")
    print([round(value, 6) for value in sample_embeddings[0][:8]])

    _print_step_header(4, "Index Build")
    subset_store = ProcessedChunkStore({chunk.chunk_id: chunk for chunk in selected_chunks})
    index_manager = ChromaIndexManager(config, adapter)
    build_result = index_manager.build(store=subset_store, mode="rebuild")
    print(f"[OK] Built collection: {build_result.collection_name}")
    print(
        f"[OK] Index embedding device: requested={build_result.requested_embedding_device} "
        f"resolved={build_result.resolved_embedding_device}"
    )
    print(f"[OK] Parent chunk count: {build_result.parent_chunk_count}")
    print(f"[OK] Embedding subchunk count: {build_result.embedding_subchunk_count}")
    print(f"[OK] Build metadata sidecar: {build_result.sidecar_path}")

    _print_step_header(5, "Dense Retrieval")
    retriever = DenseRetriever(config, adapter, subset_store, index_manager.get_collection())
    request = QueryRequest(
        question=args.question,
        filters=RetrievalFilters(tickers=[args.ticker], form_types=args.form_type or ["10-K"]),
        debug=True,
    )
    retrieval = retriever.retrieve(request)
    print(f"[OK] Retrieval status: {retrieval.status} ({retrieval.reason_code})")
    print("Retrieved parent chunks:")
    for chunk in retrieval.results:
        print(
            f"  - rank={chunk.rank} chunk_id={chunk.chunk_id} score={chunk.score:.4f} "
            f"best_subchunk_id={chunk.best_subchunk_id} section={chunk.section_title}"
        )
    if not retrieval.results:
        print("[FAIL] No retrieval results were returned for the walkthrough query.")
        return 1

    _print_step_header(6, "Prompt Assembly")
    prompt_builder = GroundedPromptBuilder(config.prompting, prompt_template)
    prompt = prompt_builder.build(question=args.question, retrieved_chunks=list(retrieval.results))
    print(f"[OK] Prompt version: {prompt.prompt_version}")
    print(f"[OK] Context chunk IDs: {prompt.context_chunk_ids}")
    print(f"[OK] Truncated chunk IDs: {prompt.truncated_chunk_ids}")
    print(f"[OK] Used context chars: {prompt.used_context_chars}")
    print("Prompt user message preview:")
    print(_preview(prompt.messages[-1].content, limit=900))

    _print_step_header(7, "Grounded Answer with Mock Provider")
    pipeline = GroundedAnswerPipeline(retriever, prompt_builder, MockLLMProvider())
    answer = pipeline.answer(request)
    print(f"[OK] Answer status: {answer.status} ({answer.reason_code})")
    print(f"[OK] Provider: {answer.provider_name}")
    print("Answer text:")
    print(answer.answer_text)
    print("Citations:")
    for citation in answer.citations:
        print(f"  - {citation.citation_id} | {citation.form_type} | {citation.section_title}")

    print("\nWalkthrough completed successfully.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a device-aware walkthrough of the V2 dense retrieval pipeline.")
    parser.add_argument("--data-dir", default="data", help="Base data directory containing processed chunks.")
    parser.add_argument("--retrieval-config", default="configs/retrieval.yaml", help="Retrieval config path.")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml", help="Prompt config path.")
    parser.add_argument("--ticker", default="NVDA", help="Ticker to use for the walkthrough subset.")
    parser.add_argument("--device", help="Optional embedding device override such as auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--form-type", action="append", default=None, help="Optional form-type filter.")
    parser.add_argument("--max-chunks", type=int, default=8, help="Maximum parent chunks to index for the demo.")
    parser.add_argument(
        "--question",
        default="What export control risks does NVIDIA describe?",
        help="Question to run through dense retrieval and grounded answering.",
    )
    parser.add_argument(
        "--persist-directory",
        default="artifacts/chroma_walkthrough",
        help="Temporary local Chroma directory for the walkthrough.",
    )
    parser.add_argument(
        "--collection-name",
        default="sec_semis_v2_walkthrough_cpu",
        help="Chroma collection name for the walkthrough run.",
    )
    return parser


def _preview(text: str, limit: int = 320) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + "..."


def _evenly_sample(items: list, limit: int) -> list:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]

    indices = {
        round(index * (len(items) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [items[index] for index in sorted(indices)]


def _print_step_header(step: int, title: str) -> None:
    print(f"\nStep {step}: {title}")
    print("-" * 80)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

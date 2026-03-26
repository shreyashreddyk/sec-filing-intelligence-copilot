"""Deterministic offline runtime helpers for fixture-backed evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Sequence

from sec_copilot.config.retrieval import PromptCatalog, RetrievalConfig
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptManager
from sec_copilot.retrieval.bm25 import BM25Retriever, tokenize_bm25_text
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingSubchunk
from sec_copilot.retrieval.indexer import ChromaIndexManager
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas.ingestion import ChunkRecord
from sec_copilot.schemas.retrieval import RetrievedChunk


_SIGNAL_STOPWORDS = {
    "a",
    "across",
    "and",
    "are",
    "did",
    "do",
    "does",
    "from",
    "how",
    "in",
    "its",
    "it",
    "of",
    "or",
    "the",
    "these",
    "this",
    "to",
    "what",
    "which",
}


@dataclass(frozen=True)
class OfflineEvalRuntime:
    """Fixture-backed retrieval and prompt-building runtime."""

    config: RetrievalConfig
    store: ProcessedChunkStore
    retriever: HybridRetriever
    prompt_builder: GroundedPromptBuilder


class DeterministicEmbeddingAdapter:
    """Stable hashed-token embedding adapter for offline eval and CI."""

    model_name = "deterministic-eval-embedding-v1"
    model_revision = None
    max_seq_length = 512
    embedding_dimension = 64
    requested_device = "cpu"
    resolved_device = "cpu"
    sentence_transformers_version = "deterministic"
    torch_version = "deterministic"
    transformers_version = "deterministic"

    def __init__(self, *, normalize_embeddings: bool = True) -> None:
        self.normalize_embeddings = normalize_embeddings

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.embedding_dimension
            for token in tokenize_bm25_text(text):
                index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.embedding_dimension
                vector[index] += 1.0
            if self.normalize_embeddings:
                norm = math.sqrt(sum(value * value for value in vector))
                if norm > 0:
                    vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    def build_subchunks(self, chunk: ChunkRecord) -> tuple[EmbeddingSubchunk, ...]:
        return (
            EmbeddingSubchunk(
                subchunk_id=f"{chunk.chunk_id}__emb_0000",
                parent_chunk_id=chunk.chunk_id,
                subchunk_index=0,
                subchunk_count=1,
                text=chunk.text,
                ticker=chunk.ticker,
                form_type=chunk.form_type,
                filing_date=chunk.filing_date,
                accession_number=chunk.accession_number,
                document_id=chunk.document_id,
                section_title=chunk.section_title,
                source_url=chunk.source_url,
            ),
        )


class TokenOverlapReranker:
    """Deterministic reranker using signal-token overlap."""

    def __init__(self, *, rerank_top_k: int) -> None:
        self.rerank_top_k = rerank_top_k

    def rerank(self, question: str, candidates: tuple[RetrievedChunk, ...]) -> tuple[RetrievedChunk, ...]:
        if not candidates:
            return ()

        query_tokens = _signal_tokens(question)
        rescored = []
        for candidate in candidates:
            score = _token_overlap_score(query_tokens, candidate.text, candidate.section_title)
            rescored.append(candidate.model_copy(update={"rerank_score": score}))

        ordered = sorted(
            rescored,
            key=lambda chunk: (-(chunk.rerank_score or 0.0), chunk.chunk_id),
        )
        return tuple(
            chunk.model_copy(update={"rerank_rank": index + 1})
            for index, chunk in enumerate(ordered[: self.rerank_top_k])
        )


def build_offline_eval_runtime(
    *,
    config: RetrievalConfig,
    prompt_catalog: PromptCatalog,
    corpus_path: str | Path,
    persist_directory: str | Path,
    collection_name: str,
) -> OfflineEvalRuntime:
    """Build a deterministic fixture-backed hybrid retrieval runtime."""

    store = ProcessedChunkStore.load(corpus_path)
    runtime_config = config.model_copy(
        update={
            "index": config.index.model_copy(
                update={
                    "persist_directory": str(persist_directory),
                    "collection_name": collection_name,
                    "default_mode": "rebuild",
                }
            )
        }
    )
    adapter = DeterministicEmbeddingAdapter(normalize_embeddings=runtime_config.embedding.normalize_embeddings)
    index_manager = ChromaIndexManager(runtime_config, adapter)
    index_manager.build(store, mode="rebuild")
    dense_retriever = DenseRetriever(runtime_config, adapter, store, index_manager.get_collection())
    bm25_retriever = BM25Retriever(store)
    reranker = TokenOverlapReranker(rerank_top_k=runtime_config.reranking.rerank_top_k)
    prompt_template = PromptManager(prompt_catalog).get_prompt(
        runtime_config.prompting.prompt_name,
        expected_version=runtime_config.prompting.prompt_version,
    )
    prompt_builder = GroundedPromptBuilder(runtime_config.retrieval, runtime_config.prompting, prompt_template)
    retriever = HybridRetriever(runtime_config, store, dense_retriever, bm25_retriever, reranker)
    return OfflineEvalRuntime(
        config=runtime_config,
        store=store,
        retriever=retriever,
        prompt_builder=prompt_builder,
    )


def _signal_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize_bm25_text(text)
        if token not in _SIGNAL_STOPWORDS and (len(token) > 2 or token.isdigit())
    }


def _token_overlap_score(query_tokens: set[str], text: str, section_title: str) -> float:
    if not query_tokens:
        return 0.1
    text_tokens = _signal_tokens(text)
    overlap = len(query_tokens & text_tokens)
    coverage = overlap / len(query_tokens)
    section_bonus = 0.05 if any(token in section_title.lower() for token in ("business", "discussion", "risk")) else 0.0
    return min(0.99, 0.18 + (coverage * 1.45) + section_bonus)


__all__ = [
    "DeterministicEmbeddingAdapter",
    "OfflineEvalRuntime",
    "TokenOverlapReranker",
    "build_offline_eval_runtime",
]

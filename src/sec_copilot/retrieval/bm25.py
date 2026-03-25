"""Deterministic BM25 retrieval over parent chunks."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
import math
import re
import unicodedata

from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.schemas.ingestion import ChunkRecord
from sec_copilot.schemas.retrieval import RetrievedChunk, RetrievalFilters


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
UNICODE_DASH_PATTERN = re.compile(r"[\u2010-\u2015\u2212]")


def tokenize_bm25_text(text: str) -> list[str]:
    """Normalize and tokenize SEC text deterministically for BM25 scoring."""

    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = UNICODE_DASH_PATTERN.sub("-", normalized)
    tokens: list[str] = []
    for token in TOKEN_PATTERN.findall(normalized):
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


@dataclass(frozen=True)
class BM25RetrievalResult:
    """BM25 parent-candidate results and candidate count."""

    results: tuple[RetrievedChunk, ...]
    candidate_count: int


class BM25Retriever:
    """Inspectable BM25 retriever over filtered parent chunks."""

    def __init__(self, store: ProcessedChunkStore, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.store = store
        self.k1 = k1
        self.b = b
        self._tokens_by_chunk_id = {
            chunk.chunk_id: tuple(tokenize_bm25_text(chunk.text))
            for chunk in store.values()
        }

    def retrieve(self, question: str, filters: RetrievalFilters, top_k: int) -> BM25RetrievalResult:
        filtered_chunks = self.store.filtered_values(filters)
        if not filtered_chunks:
            return BM25RetrievalResult(results=(), candidate_count=0)

        query_tokens = tokenize_bm25_text(question)
        if not query_tokens:
            return BM25RetrievalResult(results=(), candidate_count=0)

        tokenized_docs = {
            chunk.chunk_id: self._tokens_by_chunk_id.get(chunk.chunk_id, ())
            for chunk in filtered_chunks
        }
        lengths = {chunk_id: len(tokens) for chunk_id, tokens in tokenized_docs.items()}
        total_doc_len = sum(lengths.values())
        avgdl = total_doc_len / len(filtered_chunks) if filtered_chunks else 0.0

        doc_freqs: dict[str, int] = defaultdict(int)
        for token in dict.fromkeys(query_tokens):
            for tokens in tokenized_docs.values():
                if token in tokens:
                    doc_freqs[token] += 1

        scored: list[tuple[float, ChunkRecord]] = []
        for chunk in filtered_chunks:
            tokens = tokenized_docs[chunk.chunk_id]
            token_counts = Counter(tokens)
            score = self._score_tokens(
                query_tokens=query_tokens,
                token_counts=token_counts,
                doc_length=lengths[chunk.chunk_id],
                avgdl=avgdl,
                document_count=len(filtered_chunks),
                doc_freqs=doc_freqs,
            )
            if score <= 0.0:
                continue
            scored.append((score, chunk))

        ordered = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:top_k]
        results = tuple(
            _chunk_to_bm25_result(chunk, score, rank=index + 1)
            for index, (score, chunk) in enumerate(ordered)
        )
        return BM25RetrievalResult(results=results, candidate_count=len(scored))

    def _score_tokens(
        self,
        *,
        query_tokens: list[str],
        token_counts: Counter[str],
        doc_length: int,
        avgdl: float,
        document_count: int,
        doc_freqs: dict[str, int],
    ) -> float:
        score = 0.0
        for token in query_tokens:
            frequency = token_counts.get(token, 0)
            if frequency <= 0:
                continue
            doc_freq = doc_freqs.get(token, 0)
            idf = math.log(1.0 + ((document_count - doc_freq + 0.5) / (doc_freq + 0.5)))
            numerator = frequency * (self.k1 + 1.0)
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * (doc_length / avgdl if avgdl > 0 else 0.0)
            )
            score += idf * (numerator / denominator if denominator > 0 else 0.0)
        return score


def _chunk_to_bm25_result(chunk: ChunkRecord, score: float, *, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        ticker=chunk.ticker,
        company_name=chunk.company_name,
        form_type=chunk.form_type,
        filing_date=date.fromisoformat(chunk.filing_date),
        accession_number=chunk.accession_number,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        text=chunk.text,
        bm25_rank=rank,
        bm25_score=score,
    )


__all__ = ["BM25RetrievalResult", "BM25Retriever", "tokenize_bm25_text"]

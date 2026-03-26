"""Grounded answer pipeline over hybrid retrieval and reranked evidence."""

from __future__ import annotations

from dataclasses import dataclass
import re
from time import perf_counter

from sec_copilot.config.retrieval import RetrievalConfig
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptAssemblyResult
from sec_copilot.generation.providers import LLMProvider
from sec_copilot.retrieval.retriever import HybridRetriever, RetrievalTimingBreakdown
from sec_copilot.schemas.retrieval import Citation, QueryRequest, QueryResponse, RetrievedChunk
from sec_copilot.utils.logging import log_query_event


ABSTENTION_MESSAGES = {
    "filters_excluded_all_chunks": "No supporting evidence matched the requested filters in the indexed corpus.",
    "no_hits": "No supporting evidence found in the indexed corpus for this query.",
    "weak_support": "The retrieved evidence is too weak to support a grounded answer.",
    "insufficient_supporting_chunks": "Too few strong supporting chunks were retrieved to answer safely.",
    "model_abstained": "The model abstained because the provided evidence was insufficient.",
    "invalid_citations": "The generated answer did not reference valid cited evidence from the final context.",
    "reranker_unavailable": "The reranker was unavailable, so the system abstained instead of answering from lower-confidence evidence.",
}


@dataclass(frozen=True)
class AnswerTimingBreakdown:
    """Latency summary for grounded answer execution."""

    retrieval: RetrievalTimingBreakdown
    prompt_build_ms: float
    generation_ms: float
    citation_validation_ms: float
    total_ms: float


@dataclass(frozen=True)
class GroundedAnswerExecution:
    """Full grounded answer execution trace for API and eval use."""

    response: QueryResponse
    retrieval: object
    prompt: PromptAssemblyResult | None
    provider_name: str
    timings: AnswerTimingBreakdown


class GroundedAnswerPipeline:
    """Run hybrid retrieval, context packing, and structured grounded generation."""

    def __init__(
        self,
        config: RetrievalConfig,
        retriever: HybridRetriever,
        prompt_builder: GroundedPromptBuilder,
        provider: LLMProvider,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.provider = provider

    def answer(self, request: QueryRequest) -> QueryResponse:
        return self.execute(request).response

    def answer_with_trace(
        self,
        request: QueryRequest,
        retrieval=None,
    ) -> tuple[QueryResponse, object, PromptAssemblyResult | None]:
        execution = self.execute(request, retrieval=retrieval)
        return execution.response, execution.retrieval, execution.prompt

    def execute(
        self,
        request: QueryRequest,
        retrieval=None,
    ) -> GroundedAnswerExecution:
        started_at = perf_counter()
        retrieval = retrieval or self.retriever.retrieve(request)
        prompt: PromptAssemblyResult | None = None
        prompt_build_ms = 0.0
        generation_ms = 0.0
        citation_validation_ms = 0.0
        provider_name = getattr(self.provider, "name", "unknown")

        if retrieval.reason_code in {"filters_excluded_all_chunks", "no_hits"}:
            response = self._abstained_response(
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code=retrieval.reason_code,
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        if not retrieval.reranker_applied and self.config.reranking.required_for_generation:
            response = self._abstained_response(
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code="reranker_unavailable",
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        top_rerank_score = retrieval.retrieved_chunks[0].rerank_score if retrieval.retrieved_chunks else None
        if top_rerank_score is None or top_rerank_score < self.config.abstention.weak_top_rerank_score_threshold:
            response = self._abstained_response(
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code="weak_support",
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        strong_support_count = sum(
            1
            for chunk in retrieval.retrieved_chunks
            if (chunk.rerank_score or 0.0) >= self.config.abstention.supporting_chunk_rerank_score_threshold
        )
        if strong_support_count < self.config.abstention.min_supporting_chunks:
            response = self._abstained_response(
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code="insufficient_supporting_chunks",
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        prompt_started_at = perf_counter()
        prompt = self.prompt_builder.build(
            question=request.question,
            retrieved_chunks=list(retrieval.retrieved_chunks),
        )
        prompt_build_ms = (perf_counter() - prompt_started_at) * 1000.0
        generation_started_at = perf_counter()
        provider_answer = self.provider.generate(prompt)
        generation_ms = (perf_counter() - generation_started_at) * 1000.0
        if provider_answer.abstained:
            response = QueryResponse(
                answer=provider_answer.answer or ABSTENTION_MESSAGES["model_abstained"],
                citations=[],
                abstained=True,
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code="model_abstained",
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        citation_started_at = perf_counter()
        citations = self._validate_and_build_citations(
            citation_chunk_ids=provider_answer.citation_chunk_ids,
            retrieved_chunks=list(retrieval.retrieved_chunks),
            final_context_chunk_ids=prompt.context_chunk_ids,
        )
        citation_validation_ms = (perf_counter() - citation_started_at) * 1000.0
        if citations is None:
            response = self._abstained_response(
                retrieved_chunks=list(retrieval.retrieved_chunks),
                reason_code="invalid_citations",
            )
            self._log_query(request, retrieval, response, prompt)
            return GroundedAnswerExecution(
                response=response,
                retrieval=retrieval,
                prompt=prompt,
                provider_name=provider_name,
                timings=AnswerTimingBreakdown(
                    retrieval=retrieval.timings,
                    prompt_build_ms=prompt_build_ms,
                    generation_ms=generation_ms,
                    citation_validation_ms=citation_validation_ms,
                    total_ms=(perf_counter() - started_at) * 1000.0,
                ),
            )

        response = QueryResponse(
            answer=provider_answer.answer,
            citations=citations,
            abstained=False,
            retrieved_chunks=list(retrieval.retrieved_chunks),
            reason_code="ok",
        )
        self._log_query(request, retrieval, response, prompt)
        return GroundedAnswerExecution(
            response=response,
            retrieval=retrieval,
            prompt=prompt,
            provider_name=provider_name,
            timings=AnswerTimingBreakdown(
                retrieval=retrieval.timings,
                prompt_build_ms=prompt_build_ms,
                generation_ms=generation_ms,
                citation_validation_ms=citation_validation_ms,
                total_ms=(perf_counter() - started_at) * 1000.0,
            ),
        )

    def _validate_and_build_citations(
        self,
        *,
        citation_chunk_ids: list[str],
        retrieved_chunks: list[RetrievedChunk],
        final_context_chunk_ids: list[str],
    ) -> list[Citation] | None:
        if not citation_chunk_ids:
            return None

        retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        allowed_chunk_ids = set(retrieved_by_id)
        if self.config.grounding.citations_must_come_from_final_context:
            allowed_chunk_ids &= set(final_context_chunk_ids)

        if any(citation_id not in allowed_chunk_ids for citation_id in citation_chunk_ids):
            return None

        return [
            _citation_from_chunk(retrieved_by_id[citation_id])
            for citation_id in citation_chunk_ids
        ]

    def _abstained_response(self, *, retrieved_chunks: list[RetrievedChunk], reason_code: str) -> QueryResponse:
        return QueryResponse(
            answer=ABSTENTION_MESSAGES[reason_code],
            citations=[],
            abstained=True,
            retrieved_chunks=retrieved_chunks,
            reason_code=reason_code,
        )

    def _log_query(
        self,
        request: QueryRequest,
        retrieval,
        response: QueryResponse,
        prompt: PromptAssemblyResult | None,
    ) -> None:
        log_query_event(
            {
                "query_text": request.question,
                "filters": request.filters.model_dump(mode="json"),
                "filtered_parent_count": retrieval.stage_counts.filtered_parent_count,
                "dense_subchunk_hit_count": retrieval.stage_counts.dense_subchunk_hit_count,
                "dense_parent_candidate_count": retrieval.stage_counts.dense_parent_candidate_count,
                "bm25_candidate_count": retrieval.stage_counts.bm25_candidate_count,
                "fused_candidate_count": retrieval.stage_counts.fused_candidate_count,
                "reranked_candidate_count": retrieval.stage_counts.reranked_candidate_count,
                "final_context_chunk_ids": prompt.context_chunk_ids if prompt is not None else [],
                "prompt_name": prompt.prompt_name if prompt is not None else None,
                "prompt_version": prompt.prompt_version if prompt is not None else None,
                "abstained": response.abstained,
                "reason_code": response.reason_code,
            }
        )


def _citation_from_chunk(chunk: RetrievedChunk) -> Citation:
    snippet = re.sub(r"\s+", " ", chunk.text).strip()
    if len(snippet) > 280:
        snippet = snippet[:280].rstrip() + "..."
    return Citation(
        citation_id=chunk.chunk_id,
        ticker=chunk.ticker,
        form_type=chunk.form_type,
        filing_date=chunk.filing_date,
        accession_number=chunk.accession_number,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        snippet=snippet,
    )


__all__ = [
    "AnswerTimingBreakdown",
    "GroundedAnswerExecution",
    "GroundedAnswerPipeline",
]

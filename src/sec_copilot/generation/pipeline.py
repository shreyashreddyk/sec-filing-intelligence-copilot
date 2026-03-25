"""Grounded answer pipeline over dense retrieval results."""

from __future__ import annotations

import re

from sec_copilot.generation.prompts import GroundedPromptBuilder
from sec_copilot.generation.providers import LLMProvider
from sec_copilot.retrieval.retriever import DenseRetriever
from sec_copilot.schemas.retrieval import AnswerResponse, Citation, QueryRequest, RetrievedChunk


NO_EVIDENCE_MESSAGE = "No supporting evidence found in the indexed corpus for this query."
INVALID_CITATION_MESSAGE = "The generated answer did not reference valid retrieved evidence."


class GroundedAnswerPipeline:
    """Run dense retrieval, grounded prompt assembly, and structured generation."""

    def __init__(self, retriever: DenseRetriever, prompt_builder: GroundedPromptBuilder, provider: LLMProvider) -> None:
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.provider = provider

    def answer(self, request: QueryRequest) -> AnswerResponse:
        retrieval = self.retriever.retrieve(request)
        if retrieval.status in {"no_results", "weak_results"}:
            return AnswerResponse(
                status=retrieval.status,
                reason_code=retrieval.reason_code,
                answer_text=NO_EVIDENCE_MESSAGE,
                citations=[],
                retrieval_debug=retrieval.debug if request.debug else None,
                provider_name=None,
                prompt_version=self.prompt_builder.prompting.prompt_version,
            )

        retrieved_chunks = list(retrieval.results)
        prompt = self.prompt_builder.build(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
            prompt_top_n=request.prompt_top_n,
        )
        provider_answer = self.provider.generate(prompt)
        retrieved_by_id = {chunk.chunk_id: chunk for chunk in retrieved_chunks}
        has_invalid_citation = any(
            citation_id not in retrieved_by_id for citation_id in provider_answer.citation_chunk_ids
        )
        if has_invalid_citation or not provider_answer.citation_chunk_ids:
            return AnswerResponse(
                status="invalid_grounding",
                reason_code="invalid_citations",
                answer_text=INVALID_CITATION_MESSAGE,
                citations=[],
                retrieval_debug=retrieval.debug if request.debug else None,
                provider_name=self.provider.name,
                prompt_version=prompt.prompt_version,
            )

        citations = [_citation_from_chunk(retrieved_by_id[citation_id]) for citation_id in provider_answer.citation_chunk_ids]
        return AnswerResponse(
            status="ok",
            reason_code="none",
            answer_text=provider_answer.answer_text,
            citations=citations,
            retrieval_debug=retrieval.debug if request.debug else None,
            provider_name=self.provider.name,
            prompt_version=prompt.prompt_version,
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
    "GroundedAnswerPipeline",
    "INVALID_CITATION_MESSAGE",
    "NO_EVIDENCE_MESSAGE",
]

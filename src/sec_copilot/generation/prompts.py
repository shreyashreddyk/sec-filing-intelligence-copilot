"""Inspectable prompt assembly for grounded answer generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sec_copilot.config.retrieval import PromptTemplateConfig, PromptingConfig
from sec_copilot.schemas.retrieval import RetrievedChunk


class PromptMessage(BaseModel):
    """Serialized prompt message independent of any provider SDK."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class PromptAssemblyResult(BaseModel):
    """Inspectable prompt payload emitted before the provider call."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    messages: list[PromptMessage]
    context_chunk_ids: list[str]
    truncated_chunk_ids: list[str]
    used_context_chars: int


class GroundedPromptBuilder:
    """Build a deterministic grounded prompt from retrieved parent chunks."""

    def __init__(self, prompting: PromptingConfig, template: PromptTemplateConfig) -> None:
        self.prompting = prompting
        self.template = template

    def build(
        self,
        *,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
        prompt_top_n: int | None = None,
    ) -> PromptAssemblyResult:
        top_n = prompt_top_n or self.prompting.prompt_top_n
        remaining_budget = self.prompting.prompt_context_max_chars
        context_chunk_ids: list[str] = []
        truncated_chunk_ids: list[str] = []
        blocks: list[str] = []
        used_chars = 0

        for chunk in retrieved_chunks[:top_n]:
            if remaining_budget <= 0:
                break

            allowed_chars = min(self.prompting.prompt_chunk_max_chars, remaining_budget)
            chunk_text = chunk.text
            if len(chunk_text) > allowed_chars:
                if remaining_budget < 300:
                    continue
                chunk_text = chunk_text[:allowed_chars].rstrip()
                truncated_chunk_ids.append(chunk.chunk_id)

            block = (
                f"chunk_id: {chunk.chunk_id}\n"
                f"ticker: {chunk.ticker}\n"
                f"form_type: {chunk.form_type}\n"
                f"filing_date: {chunk.filing_date.isoformat()}\n"
                f"section_title: {chunk.section_title}\n"
                f"source_url: {chunk.source_url}\n"
                f"text:\n{chunk_text}"
            )
            blocks.append(block)
            context_chunk_ids.append(chunk.chunk_id)
            used_chars += len(chunk_text)
            remaining_budget -= len(chunk_text)

        context_blocks = "\n\n---\n\n".join(blocks)
        messages = [
            PromptMessage(
                role="system",
                content=self.template.system or "You are a grounded SEC filing analyst.",
            ),
            PromptMessage(
                role="user",
                content=(
                    self.template.user
                    or "Question:\n{question}\n\nRetrieved context:\n{context_blocks}\n\nReturn structured output only."
                ).format(
                    question=question,
                    context_blocks=context_blocks,
                ),
            ),
        ]
        return PromptAssemblyResult(
            prompt_version=self.prompting.prompt_version,
            messages=messages,
            context_chunk_ids=context_chunk_ids,
            truncated_chunk_ids=truncated_chunk_ids,
            used_context_chars=used_chars,
        )


__all__ = ["GroundedPromptBuilder", "PromptAssemblyResult", "PromptMessage"]

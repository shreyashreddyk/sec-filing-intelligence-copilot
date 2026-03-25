"""Prompt management and token-budgeted context packing for grounded answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from sec_copilot.config.retrieval import PromptCatalog, PromptTemplateConfig, PromptingConfig, RetrievalSettings
from sec_copilot.schemas.retrieval import RetrievedChunk


TOKENIZER_NAME = "cl100k_base"
FALLBACK_TOKEN_PATTERN = re.compile(r"\S+")


class PromptMessage(BaseModel):
    """Serialized prompt message independent of any provider SDK."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user"]
    content: str = Field(min_length=1)


class PromptAssemblyResult(BaseModel):
    """Inspectable prompt payload emitted before the provider call."""

    model_config = ConfigDict(extra="forbid")

    prompt_name: str
    prompt_version: str
    messages: list[PromptMessage]
    context_chunk_ids: list[str]
    truncated_chunk_ids: list[str]
    used_context_tokens: int


@dataclass(frozen=True)
class PromptTemplate:
    """Resolved prompt metadata used by the prompt builder."""

    name: str
    version: str
    system: str | None
    user: str | None


class PromptManager:
    """Resolve named prompts from the YAML catalog."""

    def __init__(self, catalog: PromptCatalog) -> None:
        self.catalog = catalog

    def get_prompt(self, name: str, *, expected_version: str | None = None) -> PromptTemplate:
        template = self.catalog.get(name)
        if expected_version is not None and template.version != expected_version:
            raise ValueError(
                f"Prompt {name!r} expected version {expected_version!r}, got {template.version!r}"
            )
        return PromptTemplate(
            name=name,
            version=template.version,
            system=template.system,
            user=template.user,
        )


class GroundedPromptBuilder:
    """Build a deterministic grounded prompt from final parent chunks."""

    def __init__(
        self,
        retrieval: RetrievalSettings,
        prompting: PromptingConfig,
        template: PromptTemplate,
    ) -> None:
        self.retrieval = retrieval
        self.prompting = prompting
        self.template = template
        self._token_utils = _TokenUtils()

    def build(
        self,
        *,
        question: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> PromptAssemblyResult:
        remaining_budget = self.prompting.max_context_tokens
        context_chunk_ids: list[str] = []
        truncated_chunk_ids: list[str] = []
        blocks: list[str] = []
        used_tokens = 0
        seen_chunk_ids: set[str] = set()

        for chunk in retrieved_chunks:
            if len(context_chunk_ids) >= self.retrieval.generation_context_top_k:
                break
            if chunk.chunk_id in seen_chunk_ids:
                continue
            if remaining_budget <= 0:
                break

            chunk_text = chunk.text
            chunk_token_count = self._token_utils.count_tokens(chunk_text)
            if chunk_token_count > remaining_budget:
                if remaining_budget < 128:
                    break
                chunk_text = self._token_utils.truncate_to_tokens(chunk_text, remaining_budget)
                truncated_chunk_ids.append(chunk.chunk_id)
                chunk_token_count = self._token_utils.count_tokens(chunk_text)

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
            seen_chunk_ids.add(chunk.chunk_id)
            used_tokens += chunk_token_count
            remaining_budget -= chunk_token_count

        context_blocks = "\n\n---\n\n".join(blocks)
        user_template = (
            self.template.user
            or "Question:\n{question}\n\nRetrieved context:\n{context_blocks}\n\nReturn structured output only."
        )
        messages = [
            PromptMessage(
                role="system",
                content=self.template.system or "You are a grounded SEC filing analyst.",
            ),
            PromptMessage(
                role="user",
                content=user_template.format(
                    question=question,
                    context_blocks=context_blocks,
                ),
            ),
        ]
        return PromptAssemblyResult(
            prompt_name=self.template.name,
            prompt_version=self.template.version,
            messages=messages,
            context_chunk_ids=context_chunk_ids,
            truncated_chunk_ids=truncated_chunk_ids,
            used_context_tokens=used_tokens,
        )


class _TokenUtils:
    """Token counting and truncation with tiktoken when available."""

    def __init__(self) -> None:
        self._encoding = self._load_encoding()

    def count_tokens(self, text: str) -> int:
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        return len(FALLBACK_TOKEN_PATTERN.findall(text))

    def truncate_to_tokens(self, text: str, token_limit: int) -> str:
        if self._encoding is not None:
            return self._encoding.decode(self._encoding.encode(text)[:token_limit]).strip()

        matches = list(FALLBACK_TOKEN_PATTERN.finditer(text))
        if len(matches) <= token_limit:
            return text.strip()
        cutoff = matches[token_limit - 1].end()
        return text[:cutoff].strip()

    def _load_encoding(self):
        try:
            import tiktoken

            return tiktoken.get_encoding(TOKENIZER_NAME)
        except Exception:
            return None


__all__ = [
    "GroundedPromptBuilder",
    "PromptAssemblyResult",
    "PromptManager",
    "PromptMessage",
    "PromptTemplate",
]

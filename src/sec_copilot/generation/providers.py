"""Provider abstraction for structured grounded-answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
import os

from openai import OpenAI

from sec_copilot.generation.prompts import PromptAssemblyResult
from sec_copilot.schemas.retrieval import ProviderAnswer


class LLMProvider(ABC):
    """Provider-agnostic interface for grounded-answer generation."""

    name: str

    @abstractmethod
    def generate(self, prompt: PromptAssemblyResult) -> ProviderAnswer:
        """Generate a structured answer from a prepared prompt."""


class MockLLMProvider(LLMProvider):
    """Deterministic provider for tests and offline development."""

    name = "mock"

    def generate(self, prompt: PromptAssemblyResult) -> ProviderAnswer:
        citation_chunk_ids = prompt.context_chunk_ids[:2]
        if not citation_chunk_ids:
            return ProviderAnswer(
                answer="Insufficient evidence in the supplied context.",
                citation_chunk_ids=[],
                abstained=True,
                notes="Deterministic mock abstention.",
            )
        suffix = ", ".join(citation_chunk_ids)
        return ProviderAnswer(
            answer=f"Mock grounded answer based on {suffix}.",
            citation_chunk_ids=citation_chunk_ids,
            abstained=False,
            notes="Deterministic mock provider output.",
        )


class OpenAILLMProvider(LLMProvider):
    """OpenAI provider using strict structured outputs."""

    name = "openai"

    def __init__(self, model_name: str = "gpt-5-nano", api_key: str | None = None, client: OpenAI | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if client is not None:
            self.client = client
        elif self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: PromptAssemblyResult) -> ProviderAnswer:
        if self.client is None:
            raise RuntimeError("OPENAI_API_KEY is required for live grounded answer generation.")

        input_messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in prompt.messages
        ]
        response = self.client.responses.parse(
            model=self.model_name,
            input=input_messages,
            text_format=ProviderAnswer,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise RuntimeError("OpenAI structured output response did not contain a parsed payload.")
        if isinstance(parsed, ProviderAnswer):
            return parsed
        return ProviderAnswer.model_validate(parsed)


__all__ = ["LLMProvider", "MockLLMProvider", "OpenAILLMProvider"]

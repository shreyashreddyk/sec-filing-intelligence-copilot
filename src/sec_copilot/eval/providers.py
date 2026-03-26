"""Eval-only provider helpers, including the deterministic reference provider."""

from __future__ import annotations

from sec_copilot.eval.schemas import EvalExample, EvalProviderName
from sec_copilot.generation.providers import MockLLMProvider, OpenAILLMProvider
from sec_copilot.generation.prompts import PromptAssemblyResult
from sec_copilot.schemas.retrieval import ProviderAnswer


class ReferenceEvalProvider:
    """Deterministic provider that returns the gold reference answer for one example."""

    name = "reference"

    def __init__(self, example: EvalExample) -> None:
        self.example = example

    def generate(self, prompt: PromptAssemblyResult) -> ProviderAnswer:
        if self.example.expected_abstention:
            return ProviderAnswer(
                answer="Insufficient evidence for a grounded answer.",
                citation_chunk_ids=[],
                abstained=True,
                notes="Eval reference provider abstention.",
            )
        if self.example.reference_answer is None:
            raise ValueError(f"Answerable example {self.example.example_id!r} is missing reference_answer.")
        return ProviderAnswer(
            answer=self.example.reference_answer,
            citation_chunk_ids=list(self.example.required_citation_chunk_ids),
            abstained=False,
            notes="Eval reference provider used the gold answer contract.",
        )


def build_eval_provider(
    provider_name: EvalProviderName,
    *,
    example: EvalExample,
    openai_model: str,
) -> object:
    """Build one provider instance for the requested eval mode."""

    if provider_name == "reference":
        return ReferenceEvalProvider(example)
    if provider_name == "mock":
        return MockLLMProvider()
    return OpenAILLMProvider(model_name=openai_model)


__all__ = ["ReferenceEvalProvider", "build_eval_provider"]

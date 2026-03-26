from __future__ import annotations

from pathlib import Path

import pytest

from sec_copilot.eval import load_eval_config, load_eval_dataset
from sec_copilot.eval.providers import ReferenceEvalProvider
from sec_copilot.eval.schemas import EvalExample
from sec_copilot.generation.prompts import PromptAssemblyResult, PromptMessage


def test_eval_dataset_contract_loads_expected_shape() -> None:
    dataset = load_eval_dataset("tests/fixtures/eval/sec_filing_qa_gold.yaml")

    assert dataset.schema_version == "sec_eval_dataset.v1"
    assert len(dataset.examples) == 12
    assert sum(1 for example in dataset.examples if example.category == "fact_lookup") == 3
    assert sum(1 for example in dataset.examples if example.category == "cross_period_comparison") == 3
    assert sum(1 for example in dataset.examples if example.category == "multi_document_synthesis") == 3
    assert sum(1 for example in dataset.examples if example.category == "unanswerable") == 3
    assert sum(1 for example in dataset.examples if "ci_smoke" in example.tags) == 8


def test_eval_config_loads_threshold_defaults() -> None:
    config = load_eval_config("configs/eval.yaml")

    assert config.schema_version == "sec_eval_config.v1"
    assert config.default_subset == "ci_smoke"
    assert config.default_mode == "full"
    assert config.default_provider == "reference"
    assert config.retrieval_ks == [1, 2, 4]
    assert len(config.thresholds.blocking) == 4
    assert len(config.thresholds.non_blocking) == 4


def test_unanswerable_eval_example_rejects_reference_answer() -> None:
    with pytest.raises(ValueError):
        EvalExample(
            example_id="bad_unanswerable",
            question="Unsupported question",
            category="unanswerable",
            expected_abstention=True,
            gold_chunk_ids=[],
            required_citation_chunk_ids=[],
            reference_answer="Should not be set",
            reference_key_points=[],
            tags=["full"],
        )


def test_reference_provider_returns_gold_answer_for_answerable_example() -> None:
    dataset = load_eval_dataset("tests/fixtures/eval/sec_filing_qa_gold.yaml")
    example = next(example for example in dataset.examples if example.example_id == "fact_nvda_h20_license")
    provider = ReferenceEvalProvider(example)
    prompt = PromptAssemblyResult(
        prompt_name="eval",
        prompt_version="v1",
        messages=[PromptMessage(role="user", content="question")],
        context_chunk_ids=list(example.required_citation_chunk_ids),
        truncated_chunk_ids=[],
        used_context_tokens=32,
    )

    answer = provider.generate(prompt)

    assert answer.abstained is False
    assert answer.answer == example.reference_answer
    assert answer.citation_chunk_ids == example.required_citation_chunk_ids


def test_reference_provider_abstains_for_unanswerable_example() -> None:
    dataset = load_eval_dataset("tests/fixtures/eval/sec_filing_qa_gold.yaml")
    example = next(example for example in dataset.examples if example.example_id == "unanswerable_nvda_dividend_increase")
    provider = ReferenceEvalProvider(example)
    prompt = PromptAssemblyResult(
        prompt_name="eval",
        prompt_version="v1",
        messages=[PromptMessage(role="user", content="question")],
        context_chunk_ids=[],
        truncated_chunk_ids=[],
        used_context_tokens=0,
    )

    answer = provider.generate(prompt)

    assert answer.abstained is True
    assert answer.citation_chunk_ids == []
    assert "Insufficient evidence" in answer.answer

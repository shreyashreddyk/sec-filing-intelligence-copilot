from __future__ import annotations

from pathlib import Path

from sec_copilot.processing.parser import parse_filing


def _fixture_text(name: str) -> str:
    return (Path("tests/fixtures/sec/filings") / name).read_text(encoding="utf-8")


def test_parse_10k_html_extracts_item_sections_and_cover_page() -> None:
    parsed = parse_filing(_fixture_text("nvda_10k_primary.html"), "10-K")

    assert parsed.parser_strategy == "item_headers"
    assert parsed.sections[0].section_key == "cover_page"
    assert [section.section_key for section in parsed.sections[1:5]] == [
        "business",
        "risk_factors",
        "md_and_a",
        "financial_statements",
    ]
    assert "toc_duplicate_detected" in parsed.warnings


def test_parse_10q_html_extracts_expected_sections() -> None:
    parsed = parse_filing(_fixture_text("amd_10q_primary.html"), "10-Q")

    assert parsed.parser_strategy == "item_headers"
    assert [section.section_key for section in parsed.sections] == [
        "financial_statements",
        "risk_factors",
        "legal_proceedings",
        "controls_and_procedures",
    ] or [section.section_key for section in parsed.sections] == [
        "financial_statements",
        "md_and_a",
        "risk_factors",
        "legal_proceedings",
        "controls_and_procedures",
    ]


def test_parse_fallback_emits_full_text_when_structure_is_weak() -> None:
    parsed = parse_filing(_fixture_text("full_submission_fallback.txt"), "10-K")

    assert parsed.parser_strategy == "full_text_fallback"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].section_key == "full_text"

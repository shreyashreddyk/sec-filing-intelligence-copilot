from __future__ import annotations

from sec_copilot.processing.chunker import chunk_filing
from sec_copilot.schemas import DownloadedFiling, FilingRecord, ParsedDocument, ParsedSection


def test_chunk_filing_is_deterministic_and_preserves_overlap() -> None:
    repeated_sentence = "Risk factors include supply constraints, customer concentration, and competition. "
    section_text = repeated_sentence * 120
    parsed = ParsedDocument(
        text=section_text,
        parser_strategy="full_text_fallback",
        warnings=(),
        sections=(
            ParsedSection(
                section_key="full_text",
                section_title="Full Text",
                section_order=1,
                item_number=None,
                text=section_text,
                char_start=0,
                char_end=len(section_text),
                parser_strategy="full_text_fallback",
                warnings=(),
            ),
        ),
    )
    filing = FilingRecord(
        company_name="NVIDIA",
        ticker="NVDA",
        cik="0001045810",
        form_type="10-K",
        filing_date="2025-02-20",
        report_date="2025-01-26",
        acceptance_datetime="2025-02-20T16:10:00.000Z",
        accession_number="0001045810-25-000050",
        accession_no_dash="000104581025000050",
        primary_document="nvda-20250126x10k.htm",
        primary_doc_description="Form 10-K",
        filing_index_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/",
        filing_metadata_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581025000050/index.json",
    )
    downloaded = DownloadedFiling(
        filing=filing,
        source_url="https://www.sec.gov/example.htm",
        source_kind="primary_document",
        raw_path="data/raw/example.htm",
        raw_text=section_text,
        warnings=(),
    )

    first_run, _, _ = chunk_filing(downloaded, parsed)
    second_run, _, _ = chunk_filing(downloaded, parsed)

    assert len(first_run) > 1
    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert first_run[1].token_start < first_run[0].token_end
    assert all(chunk.content_hash for chunk in first_run)

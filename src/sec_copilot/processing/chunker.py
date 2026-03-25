"""Deterministic section-aware chunking for SEC filings."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from sec_copilot.ingest.constants import (
    CHUNKER_VERSION,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SCHEMA_VERSION,
    CHUNK_TARGET_TOKENS,
    TOKENIZER_NAME,
)
from sec_copilot.schemas.ingestion import ChunkRecord, DownloadedFiling, ParsedDocument, SectionSummary


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    token_count: int


class _RegexFallbackTokenizer:
    """Offline-safe tokenizer fallback when tiktoken assets are unavailable."""

    name = "regex_fallback"

    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def chunk_filing(
    downloaded_filing: DownloadedFiling,
    parsed_document: ParsedDocument,
) -> tuple[tuple[ChunkRecord, ...], tuple[SectionSummary, ...], dict[str, str | int]]:
    """Chunk one parsed filing into deterministic JSONL-ready records."""

    encoding, tokenizer_name = _get_tokenizer()
    document_id = f"sec_{downloaded_filing.filing.cik}_{downloaded_filing.filing.accession_no_dash}"
    chunk_records: list[ChunkRecord] = []
    section_summaries: list[SectionSummary] = []

    for section in parsed_document.sections:
        section_chunks = _chunk_section(
            downloaded_filing=downloaded_filing,
            document_id=document_id,
            parsed_document=parsed_document,
            section=section,
            encoding=encoding,
        )
        chunk_records.extend(section_chunks)
        section_summaries.append(
            SectionSummary(
                section_key=section.section_key,
                section_title=section.section_title,
                section_order=section.section_order,
                item_number=section.item_number,
                char_start=section.char_start,
                char_end=section.char_end,
                warnings=section.warnings,
                chunk_count=len(section_chunks),
            )
        )

    chunker_config = {
        "tokenizer_name": tokenizer_name,
        "target_tokens": CHUNK_TARGET_TOKENS,
        "max_tokens": CHUNK_MAX_TOKENS,
        "overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "chunker_version": CHUNKER_VERSION,
    }
    return tuple(chunk_records), tuple(section_summaries), chunker_config


def _chunk_section(
    downloaded_filing: DownloadedFiling,
    document_id: str,
    parsed_document: ParsedDocument,
    section,
    encoding,
) -> list[ChunkRecord]:
    spans = _token_spans(section.text, encoding)
    if not spans:
        return []

    prefix_tokens = [0]
    for span in spans:
        prefix_tokens.append(prefix_tokens[-1] + span.token_count)

    document_token_offset = len(encoding.encode(parsed_document.text[: section.char_start]))
    windows = _chunk_windows(spans, prefix_tokens)
    chunk_records: list[ChunkRecord] = []

    for chunk_index, (start_index, end_index) in enumerate(windows):
        span_start = spans[start_index].start
        span_end = spans[end_index - 1].end
        raw_chunk_text = section.text[span_start:span_end]
        chunk_text = raw_chunk_text.strip()
        if not chunk_text:
            continue

        leading = len(raw_chunk_text) - len(raw_chunk_text.lstrip())
        trailing = len(raw_chunk_text) - len(raw_chunk_text.rstrip())
        char_start = section.char_start + span_start + leading
        char_end = section.char_start + span_end - trailing
        token_start = document_token_offset + prefix_tokens[start_index]
        token_end = document_token_offset + prefix_tokens[end_index]
        token_count = token_end - token_start
        content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()

        chunk_records.append(
            ChunkRecord(
                schema_version=CHUNK_SCHEMA_VERSION,
                chunk_id=f"{document_id}_s{section.section_order:02d}_c{chunk_index:04d}",
                document_id=document_id,
                company_name=downloaded_filing.filing.company_name,
                ticker=downloaded_filing.filing.ticker,
                cik=downloaded_filing.filing.cik,
                form_type=downloaded_filing.filing.form_type,
                filing_date=downloaded_filing.filing.filing_date,
                report_date=downloaded_filing.filing.report_date,
                accession_number=downloaded_filing.filing.accession_number,
                source_url=downloaded_filing.source_url,
                filing_index_url=downloaded_filing.filing.filing_index_url,
                source_kind=downloaded_filing.source_kind,
                raw_path=downloaded_filing.raw_path,
                section_key=section.section_key,
                section_title=section.section_title,
                section_order=section.section_order,
                item_number=section.item_number,
                parser_strategy=section.parser_strategy,
                chunk_index=chunk_index,
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                token_count=token_count,
                content_hash=content_hash,
                text=chunk_text,
            )
        )

    return chunk_records


def _token_spans(text: str, encoding: tiktoken.Encoding) -> list[_Span]:
    matches = list(re.finditer(r"\S+\s*", text))
    spans = [
        _Span(
            start=match.start(),
            end=match.end(),
            token_count=len(encoding.encode(match.group())),
        )
        for match in matches
    ]
    if spans:
        return spans
    if text:
        return [_Span(start=0, end=len(text), token_count=len(encoding.encode(text)))]
    return []


def _get_tokenizer():
    try:
        encoding = tiktoken.get_encoding(TOKENIZER_NAME)
        # Force initialization so offline environments fall back immediately.
        encoding.encode("tokenizer probe")
        return encoding, TOKENIZER_NAME
    except Exception:  # pragma: no cover - depends on local tokenizer cache state
        fallback = _RegexFallbackTokenizer()
        return fallback, fallback.name


def _chunk_windows(spans: list[_Span], prefix_tokens: list[int]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start_index = 0
    total_spans = len(spans)

    while start_index < total_spans:
        end_index = start_index
        token_count = 0

        while end_index < total_spans and token_count < CHUNK_TARGET_TOKENS:
            token_count += spans[end_index].token_count
            end_index += 1

        while end_index < total_spans:
            next_token_count = token_count + spans[end_index].token_count
            if next_token_count > CHUNK_MAX_TOKENS:
                break
            token_count = next_token_count
            end_index += 1

        windows.append((start_index, end_index))
        if end_index >= total_spans:
            break

        next_start = end_index
        while next_start > start_index:
            overlap_tokens = prefix_tokens[end_index] - prefix_tokens[next_start - 1]
            if overlap_tokens >= CHUNK_OVERLAP_TOKENS:
                next_start -= 1
                break
            next_start -= 1

        if next_start <= start_index:
            next_start = min(end_index, start_index + 1)
        start_index = next_start

    return windows

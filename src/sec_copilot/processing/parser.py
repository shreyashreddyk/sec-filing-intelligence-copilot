"""Section-aware filing parsing with deterministic fallbacks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from bs4 import BeautifulSoup

from sec_copilot.ingest.constants import (
    ITEM_SECTION_MAPPINGS,
    MIN_USEFUL_TEXT_CHARS,
    PARSER_STRATEGY_FULL_TEXT,
    PARSER_STRATEGY_HEADING_BLOCKS,
    PARSER_STRATEGY_ITEM_HEADERS,
    WARNING_FULL_TEXT_FALLBACK,
    WARNING_HEADING_FALLBACK,
    WARNING_MISSING_HEADERS,
    WARNING_TOC_DUPLICATE,
)
from sec_copilot.schemas.ingestion import ParsedDocument, ParsedSection


@dataclass(frozen=True)
class _HeaderCandidate:
    item_number: str
    section_key: str
    default_title: str
    start: int
    end: int
    title: str


def parse_filing(raw_text: str, form_type: str) -> ParsedDocument:
    """Parse a filing artifact into section-aware text when possible."""

    normalized_text = normalize_document_text(raw_text)
    if not normalized_text:
        return ParsedDocument(
            text="",
            parser_strategy=PARSER_STRATEGY_FULL_TEXT,
            warnings=(WARNING_FULL_TEXT_FALLBACK,),
            sections=(),
        )

    item_header_document = _parse_by_item_headers(normalized_text, form_type)
    if item_header_document is not None:
        return item_header_document

    heading_document = _parse_by_headings(normalized_text)
    if heading_document is not None:
        return heading_document

    full_section = ParsedSection(
        section_key="full_text",
        section_title="Full Text",
        section_order=1,
        item_number=None,
        text=normalized_text,
        char_start=0,
        char_end=len(normalized_text),
        parser_strategy=PARSER_STRATEGY_FULL_TEXT,
        warnings=(WARNING_FULL_TEXT_FALLBACK,),
    )
    return ParsedDocument(
        text=normalized_text,
        parser_strategy=PARSER_STRATEGY_FULL_TEXT,
        warnings=(WARNING_FULL_TEXT_FALLBACK,),
        sections=(full_section,),
    )


def normalize_document_text(raw_text: str) -> str:
    """Normalize filing HTML or text into deterministic plain text."""

    if _looks_like_html(raw_text):
        raw_text = _html_to_text(raw_text)

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = unicodedata.normalize("NFKC", text)

    normalized_lines: list[str] = []
    previous_blank = False
    for line in text.split("\n"):
        clean_line = re.sub(r"[ \t]+", " ", line).strip()
        if clean_line:
            normalized_lines.append(clean_line)
            previous_blank = False
        elif not previous_blank:
            normalized_lines.append("")
            previous_blank = True

    return "\n".join(normalized_lines).strip()


def _parse_by_item_headers(text: str, form_type: str) -> ParsedDocument | None:
    rules = ITEM_SECTION_MAPPINGS.get(form_type, ())
    if not rules:
        return None

    candidates_by_rule: list[list[_HeaderCandidate]] = []
    duplicate_detected = False
    for item_number, section_key, default_title in rules:
        matches = _find_header_candidates(text, item_number, section_key, default_title)
        if len(matches) > 1:
            duplicate_detected = True
            matches = _prune_toc_like_duplicates(text, matches)
        candidates_by_rule.append(matches)

    chosen_headers = _choose_best_header_sequence(text, candidates_by_rule)
    if len(chosen_headers) < 2:
        return None

    sections = _sections_from_headers(text, chosen_headers)
    if len(sections) < 2:
        return None

    warnings: list[str] = []
    if duplicate_detected:
        warnings.append(WARNING_TOC_DUPLICATE)
    if len(chosen_headers) < len(rules):
        warnings.append(WARNING_MISSING_HEADERS)

    return ParsedDocument(
        text=text,
        parser_strategy=PARSER_STRATEGY_ITEM_HEADERS,
        warnings=tuple(dict.fromkeys(warnings)),
        sections=tuple(sections),
    )


def _parse_by_headings(text: str) -> ParsedDocument | None:
    matches = list(_find_heading_candidates(text))
    if len(matches) < 2:
        return None

    sections: list[ParsedSection] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = _build_section(
            text=text,
            section_key="other",
            section_title=match.group().strip().rstrip(":"),
            section_order=index + 1,
            item_number=None,
            char_start=start,
            char_end=end,
            parser_strategy=PARSER_STRATEGY_HEADING_BLOCKS,
            warnings=(WARNING_HEADING_FALLBACK,),
        )
        if section is not None:
            sections.append(section)

    if len(sections) < 2:
        return None

    return ParsedDocument(
        text=text,
        parser_strategy=PARSER_STRATEGY_HEADING_BLOCKS,
        warnings=(WARNING_HEADING_FALLBACK,),
        sections=tuple(sections),
    )


def _find_header_candidates(
    text: str,
    item_number: str,
    section_key: str,
    default_title: str,
) -> list[_HeaderCandidate]:
    item_pattern = _item_number_pattern(item_number)
    pattern = re.compile(
        rf"(?im)^\s*item\s+{item_pattern}(?![A-Za-z0-9])(?:\s*[\.\-:\)]\s*|\s+)(?P<title>[^\n]{{0,200}})$"
    )
    candidates: list[_HeaderCandidate] = []
    for match in pattern.finditer(text):
        title = (match.group("title") or "").strip() or default_title
        candidates.append(
            _HeaderCandidate(
                item_number=item_number,
                section_key=section_key,
                default_title=default_title,
                start=match.start(),
                end=match.end(),
                title=title,
            )
        )
    return candidates


def _choose_best_header_sequence(
    text: str,
    candidates_by_rule: list[list[_HeaderCandidate]],
) -> list[_HeaderCandidate]:
    best_score = -1.0
    best_sequence: list[_HeaderCandidate] = []

    def recurse(rule_index: int, previous_start: int, current: list[_HeaderCandidate]) -> None:
        nonlocal best_score, best_sequence
        if rule_index >= len(candidates_by_rule):
            score = _score_header_sequence(text, current)
            if score > best_score:
                best_score = score
                best_sequence = current.copy()
            return

        recurse(rule_index + 1, previous_start, current)
        for candidate in candidates_by_rule[rule_index]:
            if candidate.start <= previous_start:
                continue
            current.append(candidate)
            recurse(rule_index + 1, candidate.start, current)
            current.pop()

    recurse(0, -1, [])
    return best_sequence


def _score_header_sequence(text: str, sequence: list[_HeaderCandidate]) -> float:
    if not sequence:
        return 0.0
    score = 0.0
    for index, candidate in enumerate(sequence):
        next_start = sequence[index + 1].start if index + 1 < len(sequence) else len(text)
        span = max(0, next_start - candidate.start)
        score += 1000.0
        score += min(span, 20_000) / 100.0
        score += candidate.start / 1000.0
        if span < MIN_USEFUL_TEXT_CHARS:
            score -= 500.0
    return score


def _prune_toc_like_duplicates(text: str, candidates: list[_HeaderCandidate]) -> list[_HeaderCandidate]:
    threshold = max(120, int(len(text) * 0.15))
    later_candidates = [candidate for candidate in candidates if candidate.start >= threshold]
    if later_candidates:
        return later_candidates
    return candidates


def _sections_from_headers(text: str, headers: list[_HeaderCandidate]) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    next_order = 1

    first_start = headers[0].start
    if first_start >= MIN_USEFUL_TEXT_CHARS:
        cover_section = _build_section(
            text=text,
            section_key="cover_page",
            section_title="Cover Page",
            section_order=next_order,
            item_number=None,
            char_start=0,
            char_end=first_start,
            parser_strategy=PARSER_STRATEGY_ITEM_HEADERS,
        )
        if cover_section is not None:
            sections.append(cover_section)
            next_order += 1

    for index, header in enumerate(headers):
        end = headers[index + 1].start if index + 1 < len(headers) else len(text)
        section = _build_section(
            text=text,
            section_key=header.section_key,
            section_title=header.title,
            section_order=next_order,
            item_number=header.item_number,
            char_start=header.start,
            char_end=end,
            parser_strategy=PARSER_STRATEGY_ITEM_HEADERS,
        )
        if section is not None:
            sections.append(section)
            next_order += 1

    return sections


def _build_section(
    text: str,
    section_key: str,
    section_title: str,
    section_order: int,
    item_number: str | None,
    char_start: int,
    char_end: int,
    parser_strategy: str,
    warnings: tuple[str, ...] = (),
) -> ParsedSection | None:
    raw_slice = text[char_start:char_end]
    stripped_slice = raw_slice.strip()
    if not stripped_slice:
        return None
    leading_whitespace = len(raw_slice) - len(raw_slice.lstrip())
    trailing_whitespace = len(raw_slice) - len(raw_slice.rstrip())
    return ParsedSection(
        section_key=section_key,
        section_title=section_title.strip(),
        section_order=section_order,
        item_number=item_number,
        text=stripped_slice,
        char_start=char_start + leading_whitespace,
        char_end=char_end - trailing_whitespace,
        parser_strategy=parser_strategy,
        warnings=warnings,
    )


def _find_heading_candidates(text: str) -> list[re.Match[str]]:
    pattern = re.compile(
        r"(?m)^(?P<title>(?:[A-Z][A-Za-z/&,\-'\s]{2,78}|[A-Z0-9][A-Z0-9/&,\-'\s]{2,78})):?$"
    )
    return [
        match
        for match in pattern.finditer(text)
        if _looks_like_heading(match.group("title"))
    ]


def _looks_like_heading(value: str) -> bool:
    words = value.split()
    if len(words) < 2 or len(words) > 12:
        return False
    letters_only = re.sub(r"[^A-Za-z]", "", value)
    if not letters_only:
        return False
    return value.isupper() or value.istitle()


def _item_number_pattern(item_number: str) -> str:
    characters: list[str] = []
    for character in item_number.upper():
        if character.isdigit():
            characters.append(character)
        else:
            characters.append(r"\s*" + re.escape(character))
    return "".join(characters)


def _looks_like_html(raw_text: str) -> bool:
    return bool(re.search(r"<(html|body|div|p|table|span)\b", raw_text, flags=re.IGNORECASE))


def _html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for tag in soup.find_all(lambda current: current.name and current.name.lower() == "ix:header"):
        tag.decompose()
    return soup.get_text("\n")

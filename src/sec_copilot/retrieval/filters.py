"""Centralized retrieval-filter normalization and application."""

from __future__ import annotations

from datetime import date
from typing import Any

from sec_copilot.schemas.ingestion import ChunkRecord
from sec_copilot.schemas.retrieval import RetrievalFilters
from sec_copilot.utils.normalization import filing_date_to_ordinal, normalize_form_type, normalize_ticker


def chunk_matches_filters(chunk: ChunkRecord, filters: RetrievalFilters) -> bool:
    """Apply normalized metadata filters to a parent chunk."""

    if filters.tickers and normalize_ticker(chunk.ticker) not in filters.tickers:
        return False
    if filters.form_types and normalize_form_type(chunk.form_type) not in filters.form_types:
        return False

    filing_date = date.fromisoformat(chunk.filing_date)
    if filters.filing_date_from and filing_date < filters.filing_date_from:
        return False
    if filters.filing_date_to and filing_date > filters.filing_date_to:
        return False
    return True


def build_chroma_where(filters: RetrievalFilters) -> dict[str, Any] | None:
    """Translate normalized retrieval filters into Chroma where syntax."""

    clauses: list[dict[str, Any]] = []
    if filters.tickers:
        clauses.append(
            {"ticker": {"$eq": filters.tickers[0]}}
            if len(filters.tickers) == 1
            else {"ticker": {"$in": filters.tickers}}
        )
    if filters.form_types:
        clauses.append(
            {"form_type": {"$eq": filters.form_types[0]}}
            if len(filters.form_types) == 1
            else {"form_type": {"$in": filters.form_types}}
        )
    if filters.filing_date_from:
        clauses.append({"filing_date_ordinal": {"$gte": filing_date_to_ordinal(filters.filing_date_from)}})
    if filters.filing_date_to:
        clauses.append({"filing_date_ordinal": {"$lte": filing_date_to_ordinal(filters.filing_date_to)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


__all__ = ["build_chroma_where", "chunk_matches_filters"]

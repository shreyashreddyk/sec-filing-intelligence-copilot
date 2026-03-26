"""Starter queries for the live-first Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StarterQuery:
    """One live-friendly starter query for the UI."""

    query_id: str
    label: str
    question: str
    tickers: tuple[str, ...]
    form_types: tuple[str, ...]
    description: str


STARTER_QUERIES: tuple[StarterQuery, ...] = (
    StarterQuery(
        query_id="nvda_export_controls",
        label="NVIDIA export control risk",
        question="What export control risks does NVIDIA describe?",
        tickers=("NVDA",),
        form_types=("10-K",),
        description="Good first check after bootstrap. It usually returns risk-factor evidence with clear citations.",
    ),
    StarterQuery(
        query_id="nvda_ai_infrastructure",
        label="NVIDIA AI infrastructure",
        question="What does NVIDIA say about AI infrastructure and accelerated computing?",
        tickers=("NVDA",),
        form_types=("10-K",),
        description="Useful for testing business-section retrieval and answer grounding over narrative sections.",
    ),
    StarterQuery(
        query_id="amd_supply_chain_risk",
        label="AMD supply chain risk",
        question="What does AMD say about supply chain risk?",
        tickers=("AMD",),
        form_types=("10-K", "10-Q"),
        description="A cross-form example that helps verify the live five-company corpus is queryable beyond NVIDIA.",
    ),
)


def starter_queries_by_label() -> dict[str, StarterQuery]:
    """Return starter queries keyed by the UI label."""

    return {query.label: query for query in STARTER_QUERIES}


__all__ = ["STARTER_QUERIES", "StarterQuery", "starter_queries_by_label"]

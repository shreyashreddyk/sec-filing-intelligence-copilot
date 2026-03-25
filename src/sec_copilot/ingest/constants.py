"""Constants shared across the V1 ingestion pipeline."""

from __future__ import annotations


ANNUAL_FORM = "10-K"
QUARTERLY_FORM = "10-Q"
TARGET_FORMS = (ANNUAL_FORM, QUARTERLY_FORM)
FORM_LIMIT_KEYS = {
    ANNUAL_FORM: "annual_limit",
    QUARTERLY_FORM: "quarterly_limit",
}
PRIMARY_DOCUMENT_EXTENSIONS = {".htm", ".html", ".txt"}
PLACEHOLDER_USER_AGENT = "Your Name your.email@example.com"

PARSER_VERSION = "parser.v1"
CHUNKER_VERSION = "chunker.v1"
CHUNK_SCHEMA_VERSION = "chunk_record.v1"
MANIFEST_SCHEMA_VERSION = "filing_manifest.v1"
RUN_SUMMARY_SCHEMA_VERSION = "run_summary.v1"

CHUNK_TARGET_TOKENS = 650
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 100
TOKENIZER_NAME = "cl100k_base"
MIN_USEFUL_TEXT_CHARS = 200

PARSER_STRATEGY_ITEM_HEADERS = "item_headers"
PARSER_STRATEGY_HEADING_BLOCKS = "heading_blocks"
PARSER_STRATEGY_FULL_TEXT = "full_text_fallback"

WARNING_TOC_DUPLICATE = "toc_duplicate_detected"
WARNING_MISSING_HEADERS = "missing_expected_item_headers"
WARNING_HEADING_FALLBACK = "fell_back_to_heading_blocks"
WARNING_FULL_TEXT_FALLBACK = "fell_back_to_full_text"
WARNING_PRIMARY_DOC_UNUSABLE = "primary_doc_unusable"
WARNING_UNSUPPORTED_PRIMARY_DOC_EXTENSION = "unsupported_primary_doc_extension"
WARNING_FEWER_FILINGS = "fewer_filings_than_requested"
WARNING_MISSING_FILING_DATE = "missing_filing_date"

ERROR_CONFIG = "config_error"
ERROR_NETWORK = "network_error"
ERROR_DOWNLOAD = "download_error"
ERROR_PRIMARY_DOC = "primary_doc_unusable"
ERROR_PARSE = "parse_error"
ERROR_WRITE = "write_error"

SECTION_KEY_VOCAB = (
    "cover_page",
    "business",
    "risk_factors",
    "unresolved_staff_comments",
    "properties",
    "legal_proceedings",
    "mine_safety_disclosures",
    "market_for_equity",
    "selected_financial_data",
    "md_and_a",
    "quantitative_market_risk",
    "financial_statements",
    "controls_and_procedures",
    "other_information",
    "other",
    "full_text",
)

ITEM_SECTION_MAPPINGS = {
    ANNUAL_FORM: (
        ("1", "business", "Business"),
        ("1A", "risk_factors", "Risk Factors"),
        ("1B", "unresolved_staff_comments", "Unresolved Staff Comments"),
        ("2", "properties", "Properties"),
        ("3", "legal_proceedings", "Legal Proceedings"),
        ("4", "mine_safety_disclosures", "Mine Safety Disclosures"),
        ("5", "market_for_equity", "Market for Registrant's Common Equity"),
        ("6", "selected_financial_data", "Selected Financial Data"),
        ("7", "md_and_a", "Management's Discussion and Analysis"),
        ("7A", "quantitative_market_risk", "Quantitative and Qualitative Disclosures About Market Risk"),
        ("8", "financial_statements", "Financial Statements and Supplementary Data"),
        ("9A", "controls_and_procedures", "Controls and Procedures"),
    ),
    QUARTERLY_FORM: (
        ("1", "financial_statements", "Financial Statements"),
        ("2", "md_and_a", "Management's Discussion and Analysis"),
        ("1A", "risk_factors", "Risk Factors"),
        ("3", "legal_proceedings", "Legal Proceedings"),
        ("4", "controls_and_procedures", "Controls and Procedures"),
        ("5", "other_information", "Other Information"),
    ),
}

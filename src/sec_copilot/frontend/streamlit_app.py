"""Thin evidence-first Streamlit UI for the live application."""

from __future__ import annotations

from datetime import date

import streamlit as st
from dotenv import load_dotenv

from sec_copilot.api.models import (
    BuildInfoResponse,
    CoverageFailureResponse,
    IngestRunResponse,
    QuerySuccessResponse,
    RetrievalDebugResponse,
    ServiceNotReadyResponse,
)
from sec_copilot.frontend.client import ApiBackendError, ApiClient, ApiMalformedResponse, ApiNetworkError
from sec_copilot.frontend.presenters import (
    build_ingest_request,
    build_query_request,
    configured_company_tickers,
    resolve_scope_options,
    safe_json,
)
from sec_copilot.config.runtime import load_runtime_paths_from_env
from sec_copilot.frontend.runtime import (
    load_frontend_backend_url_from_env,
    load_frontend_enable_bootstrap_from_env,
    load_frontend_timeouts_from_env,
)
from sec_copilot.frontend.starter_queries import STARTER_QUERIES, starter_queries_by_label

load_dotenv()

DEFAULT_BACKEND_URL = load_frontend_backend_url_from_env()
DEFAULT_COMPANIES_CONFIG_PATH = load_runtime_paths_from_env().companies_config_path
DEFAULT_FORM_TYPES = ("10-K", "10-Q")


def main() -> None:
    """Render the Streamlit application."""

    st.set_page_config(
        page_title="SEC Filing Intelligence Copilot",
        page_icon=":material/description:",
        layout="wide",
    )
    try:
        timeouts = load_frontend_timeouts_from_env()
        bootstrap_enabled = load_frontend_enable_bootstrap_from_env()
    except ValueError as exc:
        st.title("SEC Filing Intelligence Copilot")
        st.error(str(exc))
        return
    _init_session_state()

    st.title("SEC Filing Intelligence Copilot")
    if bootstrap_enabled:
        st.caption(
            "Evidence-first local app over the live FastAPI backend. "
            "Bootstrap the corpus, then inspect answers, citations, and retrieved evidence."
        )
    else:
        st.caption(
            "Evidence-first hosted app over the live FastAPI backend. "
            "Refresh the corpus separately, then inspect answers, citations, and retrieved evidence."
        )

    backend_url = _render_sidebar(bootstrap_enabled)
    client = ApiClient(
        backend_url,
        status_timeout_seconds=timeouts.status_seconds,
        query_timeout_seconds=timeouts.query_seconds,
        retrieve_debug_timeout_seconds=timeouts.retrieve_debug_seconds,
        ingest_timeout_seconds=timeouts.ingest_seconds,
    )
    health_result = client.health()
    build_info_result = client.build_info()

    scope_options = resolve_scope_options(
        build_info_result,
        companies_config_path=DEFAULT_COMPANIES_CONFIG_PATH,
        fallback_form_types=DEFAULT_FORM_TYPES,
    )
    configured_companies = configured_company_tickers(DEFAULT_COMPANIES_CONFIG_PATH)
    _render_status_panels(health_result, build_info_result)
    if bootstrap_enabled:
        _render_bootstrap_panel(client, configured_companies)
    else:
        _render_refresh_panel()
    _render_query_form(client, scope_options, build_info_result)
    _render_results(build_info_result)


def _init_session_state() -> None:
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL
    if "bootstrap_companies" not in st.session_state:
        st.session_state.bootstrap_companies = list(configured_company_tickers(DEFAULT_COMPANIES_CONFIG_PATH))
    if "bootstrap_form_types" not in st.session_state:
        st.session_state.bootstrap_form_types = list(DEFAULT_FORM_TYPES)
    if "bootstrap_annual_limit" not in st.session_state:
        st.session_state.bootstrap_annual_limit = 1
    if "bootstrap_quarterly_limit" not in st.session_state:
        st.session_state.bootstrap_quarterly_limit = 1
    if "bootstrap_force_refresh" not in st.session_state:
        st.session_state.bootstrap_force_refresh = False
    if "bootstrap_index_mode" not in st.session_state:
        st.session_state.bootstrap_index_mode = "rebuild"
    if "bootstrap_user_agent" not in st.session_state:
        st.session_state.bootstrap_user_agent = ""
    if "query_question" not in st.session_state:
        query = STARTER_QUERIES[0]
        st.session_state.query_question = query.question
        st.session_state.query_tickers = list(query.tickers)
        st.session_state.query_form_types = list(query.form_types)
        st.session_state.query_use_date_filter = False
        st.session_state.query_date_from = date.today()
        st.session_state.query_date_to = date.today()
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    if "last_ingest_result" not in st.session_state:
        st.session_state.last_ingest_result = None
    if "last_query_result" not in st.session_state:
        st.session_state.last_query_result = None
    if "last_debug_result" not in st.session_state:
        st.session_state.last_debug_result = None
    if "last_query_request" not in st.session_state:
        st.session_state.last_query_request = None


def _render_sidebar(bootstrap_enabled: bool) -> str:
    st.sidebar.header("App Controls")
    st.sidebar.text_input("Backend URL", key="backend_url")
    if bootstrap_enabled:
        st.sidebar.caption("Start the backend with `make serve-api`, then bootstrap the corpus from this UI.")
    else:
        st.sidebar.caption("Hosted mode hides bootstrap controls. Refresh the corpus through the Kubernetes Job or CronJob.")

    example_by_label = starter_queries_by_label()
    selected_label = st.sidebar.selectbox(
        "Starter Query",
        options=list(example_by_label),
        index=0,
    )
    if st.sidebar.button("Load Query", use_container_width=True):
        query = example_by_label[selected_label]
        st.session_state.query_question = query.question
        st.session_state.query_tickers = list(query.tickers)
        st.session_state.query_form_types = list(query.form_types)
        st.session_state.query_use_date_filter = False

    st.sidebar.markdown("**Why this query is useful**")
    st.sidebar.write(example_by_label[selected_label].description)

    return st.session_state.backend_url


def _render_status_panels(health_result, build_info_result) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Backend Status")
        if isinstance(health_result, ApiNetworkError):
            st.error(f"Network error while calling `{health_result.endpoint}`: {health_result.message}")
        elif isinstance(health_result, ApiBackendError):
            st.error(f"{health_result.message}")
        elif isinstance(health_result, ApiMalformedResponse):
            st.error(f"Malformed response from `{health_result.endpoint}`.")
            with st.expander("Raw response"):
                st.code(safe_json(health_result.raw_body), language="json")
        else:
            readiness = "Ready" if health_result.query_ready else "Not ready"
            st.metric("Query readiness", readiness)
            st.write(f"`retrieve_ready`: `{health_result.retrieve_ready}`")
            st.write(f"`index_status`: `{health_result.index_status}`")
            if health_result.last_index_refresh_at:
                st.write(f"`last_index_refresh_at`: `{health_result.last_index_refresh_at}`")
            for warning in health_result.warnings:
                st.warning(warning)

    with right:
        st.subheader("Indexed Scope")
        if isinstance(build_info_result, ApiNetworkError):
            st.error(f"Network error while calling `{build_info_result.endpoint}`: {build_info_result.message}")
        elif isinstance(build_info_result, ApiBackendError):
            st.error(build_info_result.message)
        elif isinstance(build_info_result, ApiMalformedResponse):
            st.error(f"Malformed response from `{build_info_result.endpoint}`.")
            with st.expander("Raw response"):
                st.code(safe_json(build_info_result.raw_body), language="json")
        elif isinstance(build_info_result, BuildInfoResponse):
            st.write(f"`coverage_status`: `{build_info_result.coverage_status}`")
            st.write(f"`index_status`: `{build_info_result.index_status}`")
            st.write(f"`effective_provider`: `{build_info_result.effective_provider}`")
            st.write(f"`collection_name`: `{build_info_result.collection_name}`")
            st.write(f"`persist_directory`: `{build_info_result.persist_directory}`")
            if build_info_result.provider_fallback_active and build_info_result.provider_fallback_reason:
                st.warning(
                    "OpenAI generation is unavailable. Answers will use the mock fallback "
                    f"(`{build_info_result.provider_fallback_reason}`)."
                )
            st.write("Target companies:", ", ".join(build_info_result.target_scope.companies) or "None")
            st.write("Indexed companies:", ", ".join(build_info_result.indexed_scope.companies) or "None")
            st.write("Indexed form types:", ", ".join(build_info_result.indexed_scope.form_types) or "None")
        else:
            st.info("Build info is unavailable.")


def _render_bootstrap_panel(client: ApiClient, configured_companies: tuple[str, ...]) -> None:
    st.subheader("Bootstrap Corpus")
    st.write(
        "Fetch SEC filings for the configured company universe, process chunks, and rebuild the local retrieval index."
    )
    with st.form("bootstrap_form", clear_on_submit=False):
        st.multiselect(
            "Companies to ingest",
            options=list(configured_companies),
            default=st.session_state.bootstrap_companies,
            key="bootstrap_companies",
        )
        st.multiselect(
            "Form types to ingest",
            options=list(DEFAULT_FORM_TYPES),
            default=st.session_state.bootstrap_form_types,
            key="bootstrap_form_types",
        )
        limits_left, limits_right = st.columns(2)
        with limits_left:
            st.number_input(
                "Annual limit per company",
                min_value=0,
                step=1,
                key="bootstrap_annual_limit",
            )
        with limits_right:
            st.number_input(
                "Quarterly limit per company",
                min_value=0,
                step=1,
                key="bootstrap_quarterly_limit",
            )
        st.selectbox(
            "Index mode",
            options=["rebuild", "upsert"],
            key="bootstrap_index_mode",
        )
        st.checkbox("Force refresh SEC downloads", key="bootstrap_force_refresh")
        st.text_input(
            "SEC User Agent",
            key="bootstrap_user_agent",
            help="Leave blank to use SEC_USER_AGENT from the backend environment.",
        )
        submitted = st.form_submit_button("Run ingest and rebuild index", use_container_width=True)

    if submitted:
        request = build_ingest_request(
            companies=st.session_state.bootstrap_companies,
            form_types=st.session_state.bootstrap_form_types,
            annual_limit=int(st.session_state.bootstrap_annual_limit),
            quarterly_limit=int(st.session_state.bootstrap_quarterly_limit),
            force_refresh=st.session_state.bootstrap_force_refresh,
            user_agent=st.session_state.bootstrap_user_agent,
            index_mode=st.session_state.bootstrap_index_mode,
        )
        with st.spinner("Running SEC ingest, processing filings, and refreshing the index..."):
            st.session_state.last_ingest_result = client.ingest_run(request)
        st.rerun()

    _render_ingest_result(st.session_state.last_ingest_result)


def _render_refresh_panel() -> None:
    st.subheader("Corpus Refresh")
    st.info(
        "This hosted deployment hides the UI bootstrap form because the public API disables admin routes. "
        "Run the separate Kubernetes corpus-refresh Job or CronJob to update the mounted corpus and Chroma state."
    )


def _render_query_form(client: ApiClient, scope_options, build_info_result) -> None:
    st.subheader("Ask A Filing Question")
    st.write(
        "Select companies, constrain form and date scope, ask a question, then inspect citations and retrieved evidence."
    )
    if isinstance(build_info_result, BuildInfoResponse) and not build_info_result.query_ready:
        st.info("Grounded query execution is not ready yet. Bootstrap the corpus above, then run a query.")

    with st.form("query_form", clear_on_submit=False):
        selected_tickers = st.multiselect(
            "Companies",
            options=list(scope_options.companies),
            default=[ticker for ticker in st.session_state.query_tickers if ticker in scope_options.companies],
            key="query_tickers",
        )
        selected_form_types = st.multiselect(
            "Form types",
            options=list(scope_options.form_types),
            default=[form_type for form_type in st.session_state.query_form_types if form_type in scope_options.form_types],
            key="query_form_types",
        )
        st.checkbox("Apply filing date filter", key="query_use_date_filter")
        date_left, date_right = st.columns(2)
        with date_left:
            st.date_input("Filing date from", key="query_date_from")
        with date_right:
            st.date_input("Filing date to", key="query_date_to")
        st.text_area("Question", key="query_question", height=120)
        st.checkbox("Show retrieval debug details", key="show_debug")
        submitted = st.form_submit_button(
            "Run grounded query",
            use_container_width=True,
            disabled=not st.session_state.query_question.strip(),
        )

    if submitted:
        request = build_query_request(
            question=st.session_state.query_question,
            tickers=selected_tickers,
            form_types=selected_form_types,
            use_date_filter=st.session_state.query_use_date_filter,
            filing_date_from=st.session_state.query_date_from,
            filing_date_to=st.session_state.query_date_to,
        )
        st.session_state.last_query_request = request
        with st.spinner("Running grounded query over the indexed corpus..."):
            query_result = client.query(request)
            debug_result = None
            if isinstance(query_result, QuerySuccessResponse) and st.session_state.show_debug:
                debug_result = client.retrieve_debug(request)
        st.session_state.last_query_result = query_result
        st.session_state.last_debug_result = debug_result


def _render_ingest_result(result) -> None:
    if result is None:
        return

    if isinstance(result, IngestRunResponse):
        st.success("Ingest and index refresh completed.")
        st.write(f"`run_status`: `{result.run_summary.status}`")
        st.write(
            f"`successful_filings`: `{result.run_summary.successful_filings}` | "
            f"`failed_filings`: `{result.run_summary.failed_filings}`"
        )
        st.write(
            f"`coverage_status`: `{result.coverage_state.coverage_status}` | "
            f"`index_status`: `{result.coverage_state.index_status}`"
        )
        st.write("Indexed companies:", ", ".join(result.coverage_state.indexed_scope.companies) or "None")
        st.write("Indexed form types:", ", ".join(result.coverage_state.indexed_scope.form_types) or "None")
        st.caption(f"Bootstrap completed in {result.timings.total_ms:.2f} ms.")
        with st.expander("Company run summary"):
            st.json(
                [
                    company.model_dump(mode="json")
                    for company in result.run_summary.company_results
                ]
            )
        return

    if isinstance(result, ApiNetworkError):
        st.error(f"Network error while calling `{result.endpoint}`: {result.message}")
        return

    if isinstance(result, ApiBackendError):
        st.error(result.message)
        with st.expander("Raw backend response"):
            st.code(safe_json(result.raw_body), language="json")
        return

    if isinstance(result, ApiMalformedResponse):
        st.error(f"Malformed response from `{result.endpoint}`.")
        with st.expander("Raw response"):
            st.code(safe_json(result.raw_body), language="json")


def _render_results(build_info_result) -> None:
    result = st.session_state.last_query_result
    if result is None:
        st.info(
            "No query has been run yet. Bootstrap the corpus above, then load one of the starter queries "
            "from the sidebar and inspect the answer, citations, and evidence chunks here."
        )
        st.markdown("### Starter Queries")
        for query in STARTER_QUERIES:
            st.write(f"**{query.label}**: {query.question}")
        return

    if isinstance(result, QuerySuccessResponse):
        if result.abstained:
            st.warning(f"Abstained with `reason_code={result.reason_code}`")
        else:
            st.success(f"Answered with `reason_code={result.reason_code}`")
        st.markdown("### Answer")
        st.write(result.answer)
        if result.abstained:
            st.caption("Abstention is shown as a normal query result, not a backend failure.")
        elif (
            isinstance(build_info_result, BuildInfoResponse)
            and build_info_result.provider_fallback_active
            and build_info_result.effective_provider == "mock"
        ):
            st.caption("This answer used the mock fallback because live OpenAI generation is unavailable.")

        st.markdown("### Citations")
        if result.citations:
            for citation in result.citations:
                with st.container(border=True):
                    st.write(
                        f"**{citation.ticker} {citation.form_type}** | "
                        f"{citation.filing_date} | {citation.section_title}"
                    )
                    st.write(f"`{citation.citation_id}`")
                    st.write(citation.snippet)
                    st.link_button("Open SEC source", citation.source_url)
        else:
            st.info("No citations were returned.")

        st.markdown("### Evidence Chunks")
        if result.retrieved_chunks:
            for chunk in result.retrieved_chunks:
                label = (
                    f"{chunk.ticker} {chunk.form_type} | {chunk.filing_date} | "
                    f"{chunk.section_title} | rerank={chunk.rerank_score}"
                )
                with st.expander(label):
                    st.write(f"`{chunk.chunk_id}`")
                    st.write(f"Source: {chunk.source_url}")
                    st.write(chunk.text)
                    st.caption(
                        f"dense_rank={chunk.dense_rank} bm25_rank={chunk.bm25_rank} "
                        f"rrf_score={chunk.rrf_score} rerank_rank={chunk.rerank_rank}"
                    )
        else:
            st.info("No evidence chunks were returned.")

        _render_debug_panel(st.session_state.last_debug_result)
        return

    if isinstance(result, ServiceNotReadyResponse):
        st.error("Backend is not ready for grounded query execution.")
        st.write(f"`index_status`: `{result.index_status}`")
        st.write(f"`coverage_status`: `{result.coverage_status}`")
        if result.last_index_refresh_at:
            st.write(f"`last_index_refresh_at`: `{result.last_index_refresh_at}`")
        if result.last_ingest_completed_at:
            st.write(f"`last_ingest_completed_at`: `{result.last_ingest_completed_at}`")
        for warning in result.warnings:
            st.warning(warning)
        return

    if isinstance(result, CoverageFailureResponse):
        st.error("Requested scope is not fully covered by the indexed corpus.")
        st.write(f"`coverage_status`: `{result.coverage_status}`")
        st.write("Missing tickers:", ", ".join(result.missing_scope.tickers) or "None")
        st.write("Missing form types:", ", ".join(result.missing_scope.form_types) or "None")
        if result.missing_scope.pairs:
            st.write(
                "Missing pairs:",
                ", ".join(f"{pair.ticker}|{pair.form_type}" for pair in result.missing_scope.pairs),
            )
        return

    if isinstance(result, ApiNetworkError):
        st.error(f"Network error while calling `{result.endpoint}`: {result.message}")
        return

    if isinstance(result, ApiBackendError):
        st.error(result.message)
        with st.expander("Raw backend response"):
            st.code(safe_json(result.raw_body), language="json")
        return

    if isinstance(result, ApiMalformedResponse):
        st.error(f"Malformed response from `{result.endpoint}`.")
        with st.expander("Raw response"):
            st.code(safe_json(result.raw_body), language="json")


def _render_debug_panel(result) -> None:
    if result is None:
        return

    st.markdown("### Retrieval Debug")
    if isinstance(result, RetrievalDebugResponse):
        st.write(f"`reason_code`: `{result.reason_code}`")
        st.write(f"`reranker_applied`: `{result.reranker_applied}`")
        if result.reranker_skipped_reason:
            st.write(f"`reranker_skipped_reason`: `{result.reranker_skipped_reason}`")
        st.json(result.stage_counts.model_dump(mode="json"))
        st.json(result.timings.model_dump(mode="json"))
        return

    if isinstance(result, ApiMalformedResponse):
        st.error(f"Malformed response from `{result.endpoint}`.")
        with st.expander("Raw response"):
            st.code(safe_json(result.raw_body), language="json")
        return

    if isinstance(result, (ApiNetworkError, ApiBackendError, ServiceNotReadyResponse, CoverageFailureResponse)):
        st.info("Retrieval debug details were unavailable for this query.")


if __name__ == "__main__":
    main()

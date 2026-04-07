"""Service-layer orchestration for the V5 FastAPI backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import chromadb
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sec_copilot import __version__
from sec_copilot.api.coverage import (
    CoverageState,
    IndexedScope,
    TargetScope,
    assess_request_coverage,
    build_coverage_state,
    build_indexed_scope,
    coverage_state_path,
    latest_ingest_snapshot,
    load_coverage_state,
    write_coverage_state,
)
from sec_copilot.api.models import (
    BuildInfoResponse,
    CoverageFailureResponse,
    EvalRunRequest,
    EvalRunResponse,
    EvalRunTimings,
    HealthResponse,
    IngestRunRequest,
    IngestRunResponse,
    IngestRunTimings,
    ProviderMode,
    QuerySuccessResponse,
    QueryTimings,
    ReadinessResponse,
    RetrievalDebugResponse,
    RetrievalTimings,
    RunSummaryModel,
    ServiceNotReadyResponse,
)
from sec_copilot.config import (
    PromptCatalog,
    RetrievalConfig,
    default_project_root,
    load_api_runtime_settings_from_env,
    load_company_universe,
    load_prompt_catalog,
    load_retrieval_config,
    load_runtime_paths_from_env,
    resolve_runtime_path,
)
from sec_copilot.eval.artifacts import resolve_output_dir, write_eval_artifacts
from sec_copilot.eval.config import load_eval_config
from sec_copilot.eval.dataset import load_eval_dataset
from sec_copilot.eval.runner import run_eval
from sec_copilot.generation.pipeline import GroundedAnswerPipeline
from sec_copilot.generation.prompts import GroundedPromptBuilder, PromptManager
from sec_copilot.generation.providers import LLMProvider, MockLLMProvider, OpenAILLMProvider
from sec_copilot.ingest.pipeline import IngestionConfig, run_ingestion
from sec_copilot.rerank.cross_encoder import CrossEncoderReranker
from sec_copilot.retrieval.bm25 import BM25Retriever
from sec_copilot.retrieval.corpus import ProcessedChunkStore
from sec_copilot.retrieval.embedding import EmbeddingAdapter, SentenceTransformerEmbeddingAdapter
from sec_copilot.retrieval.indexer import ChromaIndexManager, IndexBuildMetadata, IndexBuildResult, compute_corpus_fingerprint
from sec_copilot.retrieval.retriever import DenseRetriever, HybridRetriever
from sec_copilot.schemas.retrieval import QueryRequest
from sec_copilot.utils.io import to_jsonable
from sec_copilot.utils.logging import log_api_event
from sec_copilot.utils.normalization import normalize_form_type


EmbeddingAdapterFactory = Callable[[object], EmbeddingAdapter]
RerankerFactory = Callable[[object], object]
ProviderFactory = Callable[[RetrievalConfig, bool], tuple[LLMProvider | None, ProviderMode, bool, str | None]]


class ApiSettings(BaseModel):
    """Configuration for the V5 API runtime."""

    model_config = ConfigDict(extra="forbid")

    service_name: str = "sec-filing-intelligence-copilot"
    project_root: Path = Field(default_factory=default_project_root)
    data_dir: Path = Field(default_factory=lambda: load_runtime_paths_from_env().data_dir)
    companies_config_path: Path = Field(default_factory=lambda: load_runtime_paths_from_env().companies_config_path)
    retrieval_config_path: Path = Field(default_factory=lambda: load_runtime_paths_from_env().retrieval_config_path)
    prompts_config_path: Path = Field(default_factory=lambda: load_runtime_paths_from_env().prompts_config_path)
    eval_config_path: Path = Field(default_factory=lambda: load_runtime_paths_from_env().eval_config_path)
    log_level: str = Field(default_factory=lambda: load_api_runtime_settings_from_env().log_level)
    strict_coverage: bool = True
    mock_fallback_when_openai_missing: bool = True
    default_annual_limit: int = Field(default=2, ge=0)
    default_quarterly_limit: int = Field(default=4, ge=0)
    default_form_types: list[str] = Field(default_factory=lambda: ["10-K", "10-Q"])

    @field_validator("default_form_types", mode="before")
    @classmethod
    def _normalize_default_form_types(cls, value: object) -> list[str]:
        if value is None:
            return ["10-K", "10-Q"]
        if isinstance(value, str):
            value = [value]
        normalized = [normalize_form_type(str(item)) for item in value]
        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("default_form_types must contain at least one form type")
        return deduped


@dataclass(frozen=True)
class ApiRuntime:
    """Loaded runtime components used by live retrieval and query endpoints."""

    store: ProcessedChunkStore
    retriever: HybridRetriever
    pipeline: GroundedAnswerPipeline
    provider_name: ProviderMode
    reranker: object | None


@dataclass(frozen=True)
class ApiState:
    """Current service state exposed through health and build-info."""

    config: RetrievalConfig
    prompt_catalog: PromptCatalog
    coverage_state: CoverageState
    index_metadata: IndexBuildMetadata | None
    retrieve_ready: bool
    query_ready: bool
    configured_provider: ProviderMode
    effective_provider: ProviderMode
    provider_fallback_enabled: bool
    provider_fallback_active: bool
    provider_fallback_reason: str | None
    warnings: list[str]
    runtime: ApiRuntime | None


class CopilotApiService:
    """Thin service layer that coordinates API requests with existing domain modules."""

    def __init__(
        self,
        settings: ApiSettings | None = None,
        *,
        embedding_adapter_factory: EmbeddingAdapterFactory | None = None,
        reranker_factory: RerankerFactory | None = None,
        provider_factory: ProviderFactory | None = None,
    ) -> None:
        self.settings = settings or ApiSettings()
        self.embedding_adapter_factory = embedding_adapter_factory or SentenceTransformerEmbeddingAdapter
        self.reranker_factory = reranker_factory or CrossEncoderReranker
        self.provider_factory = provider_factory or self._default_provider_factory
        self._state: ApiState | None = None

    @property
    def state(self) -> ApiState:
        if self._state is None:
            self.refresh_state(load_query_runtime=False)
        return self._state  # type: ignore[return-value]

    def initialize(self) -> None:
        """Load the service state and preflight the runtime when possible."""

        self.refresh_state(load_query_runtime=True)

    def refresh_state(self, *, load_query_runtime: bool) -> ApiState:
        """Refresh the persisted state and optionally preflight the query runtime."""

        config = self._load_runtime_retrieval_config()
        prompt_catalog = load_prompt_catalog(self.settings.prompts_config_path)
        default_target_scope = self._default_target_scope()
        store = ProcessedChunkStore.load(self.settings.data_dir)
        processed_corpus_fingerprint = (
            compute_corpus_fingerprint(store.values())
            if len(store) > 0
            else None
        )

        index_metadata = self._load_index_metadata(config)
        collection_exists = self._collection_exists(config.index.persist_directory, config.index.collection_name)
        if len(store) == 0 or index_metadata is None or not collection_exists:
            index_status = "missing"
        elif processed_corpus_fingerprint != index_metadata.corpus_fingerprint:
            index_status = "stale"
        else:
            index_status = "fresh"

        coverage_path = coverage_state_path(config.index.persist_directory, config.index.collection_name)
        persisted_coverage = load_coverage_state(coverage_path)
        last_ingest_run_id, last_ingest_completed_at = latest_ingest_snapshot(self.settings.data_dir)

        if persisted_coverage is not None:
            coverage_state = persisted_coverage.model_copy(
                update={
                    "last_ingest_run_id": persisted_coverage.last_ingest_run_id or last_ingest_run_id,
                    "last_ingest_completed_at": persisted_coverage.last_ingest_completed_at or last_ingest_completed_at,
                    "processed_corpus_fingerprint": processed_corpus_fingerprint,
                    "indexed_corpus_fingerprint": (
                        index_metadata.corpus_fingerprint if index_metadata is not None else persisted_coverage.indexed_corpus_fingerprint
                    ),
                    "last_index_refresh_at": (
                        datetime.fromisoformat(index_metadata.built_at)
                        if index_metadata is not None
                        else persisted_coverage.last_index_refresh_at
                    ),
                    "index_status": index_status,
                }
            )
        else:
            indexed_scope = build_indexed_scope(store) if index_status == "fresh" else IndexedScope()
            coverage_state = build_coverage_state(
                target_scope=default_target_scope,
                indexed_scope=indexed_scope,
                last_ingest_run_id=last_ingest_run_id,
                last_ingest_completed_at=last_ingest_completed_at,
                last_index_refresh_at=datetime.fromisoformat(index_metadata.built_at) if index_metadata is not None else None,
                processed_corpus_fingerprint=processed_corpus_fingerprint,
                indexed_corpus_fingerprint=index_metadata.corpus_fingerprint if index_metadata is not None else None,
                index_status=index_status,
            )

        warnings = self._build_warnings(
            store=store,
            coverage_state=coverage_state,
            index_metadata=index_metadata,
            index_status=index_status,
        )
        configured_provider: ProviderMode = config.provider.default_name
        provider_fallback_active = False
        provider_fallback_reason: str | None = None
        effective_provider: ProviderMode = configured_provider
        retrieve_ready = index_status == "fresh" and len(store) > 0
        query_ready = False
        runtime: ApiRuntime | None = None

        if load_query_runtime and retrieve_ready:
            try:
                provider, effective_provider, provider_fallback_active, provider_fallback_reason = self.provider_factory(
                    config,
                    self.settings.mock_fallback_when_openai_missing,
                )
                runtime_provider = provider or MockLLMProvider()
                runtime = self._build_runtime(config, prompt_catalog, store, runtime_provider, effective_provider)
                query_ready = provider is not None
                if provider is None:
                    warnings.append("Query provider is not available for grounded answering.")
                if query_ready and config.reranking.enabled and config.reranking.required_for_generation and runtime.reranker is not None:
                    query_ready = self._preflight_reranker(runtime.reranker, warnings)
                if provider_fallback_active and provider_fallback_reason:
                    warnings.append(
                        f"OpenAI provider was unavailable; using mock fallback ({provider_fallback_reason})."
                    )
            except Exception as exc:
                warnings.append(f"Failed to build live retrieval runtime: {exc}")
                retrieve_ready = False
                query_ready = False
                runtime = None
        else:
            _, effective_provider, provider_fallback_active, provider_fallback_reason = self.provider_factory(
                config,
                self.settings.mock_fallback_when_openai_missing,
            )
            if provider_fallback_active and provider_fallback_reason:
                warnings.append(
                    f"OpenAI provider was unavailable; mock fallback is configured ({provider_fallback_reason})."
                )

        deduped_warnings = list(dict.fromkeys(warnings))
        self._state = ApiState(
            config=config,
            prompt_catalog=prompt_catalog,
            coverage_state=coverage_state,
            index_metadata=index_metadata,
            retrieve_ready=retrieve_ready,
            query_ready=query_ready,
            configured_provider=configured_provider,
            effective_provider=effective_provider,
            provider_fallback_enabled=self.settings.mock_fallback_when_openai_missing,
            provider_fallback_active=provider_fallback_active,
            provider_fallback_reason=provider_fallback_reason,
            warnings=deduped_warnings,
            runtime=runtime,
        )
        return self._state

    def health(self) -> HealthResponse:
        """Return the fast liveness and readiness summary."""

        state = self.state
        return HealthResponse(
            service=self.settings.service_name,
            version=__version__,
            retrieve_ready=state.retrieve_ready,
            query_ready=state.query_ready,
            index_status=state.coverage_state.index_status,
            last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            last_ingest_completed_at=state.coverage_state.last_ingest_completed_at,
            warnings=state.warnings,
        )

    def readiness(self) -> ReadinessResponse:
        """Return the compact readiness payload used by Kubernetes probes."""

        state = self.state
        return ReadinessResponse(
            status="ready" if state.query_ready else "not_ready",
            service=self.settings.service_name,
            version=__version__,
            retrieve_ready=state.retrieve_ready,
            query_ready=state.query_ready,
            index_status=state.coverage_state.index_status,
            last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            last_ingest_completed_at=state.coverage_state.last_ingest_completed_at,
            warnings=state.warnings,
        )

    def build_info(self) -> BuildInfoResponse:
        """Return the detailed build and runtime state."""

        state = self.state
        return BuildInfoResponse(
            service=self.settings.service_name,
            version=__version__,
            retrieve_ready=state.retrieve_ready,
            query_ready=state.query_ready,
            configured_provider=state.configured_provider,
            effective_provider=state.effective_provider,
            provider_fallback_enabled=state.provider_fallback_enabled,
            provider_fallback_active=state.provider_fallback_active,
            provider_fallback_reason=state.provider_fallback_reason,
            prompt_name=state.config.prompting.prompt_name,
            prompt_version=state.config.prompting.prompt_version,
            collection_name=state.config.index.collection_name,
            persist_directory=state.config.index.persist_directory,
            coverage_status=state.coverage_state.coverage_status,
            target_scope=state.coverage_state.target_scope,
            indexed_scope=state.coverage_state.indexed_scope,
            index_status=state.coverage_state.index_status,
            processed_corpus_fingerprint=state.coverage_state.processed_corpus_fingerprint,
            indexed_corpus_fingerprint=state.coverage_state.indexed_corpus_fingerprint,
            index_build_metadata=state.index_metadata,
            last_ingest_completed_at=state.coverage_state.last_ingest_completed_at,
            last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            warnings=state.warnings,
        )

    def query(
        self,
        request: QueryRequest,
    ) -> QuerySuccessResponse | ServiceNotReadyResponse | CoverageFailureResponse:
        """Run the grounded query flow under strict readiness and coverage policy."""

        state = self.state
        not_ready = self._not_ready_response(require_query=True)
        if not_ready is not None:
            return not_ready
        assert state.runtime is not None

        assessment = assess_request_coverage(state.runtime.store, request.filters)
        if self.settings.strict_coverage and assessment.coverage_status != "covered":
            return CoverageFailureResponse(
                message="The requested scope is not fully covered by the indexed corpus.",
                coverage_status=assessment.coverage_status,
                indexed_scope=assessment.indexed_scope,
                missing_scope=assessment.missing_scope,
                last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            )

        execution = state.runtime.pipeline.execute(request)
        return QuerySuccessResponse.from_query_response(
            execution.response,
            coverage_status=assessment.coverage_status,
            indexed_scope=assessment.indexed_scope,
            missing_scope=assessment.missing_scope,
            last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            timings=QueryTimings(
                retrieval=RetrievalTimings.model_validate(to_jsonable(execution.timings.retrieval)),
                prompt_build_ms=execution.timings.prompt_build_ms,
                generation_ms=execution.timings.generation_ms,
                citation_validation_ms=execution.timings.citation_validation_ms,
                total_ms=execution.timings.total_ms,
            ),
        )

    def retrieve_debug(
        self,
        request: QueryRequest,
    ) -> RetrievalDebugResponse | ServiceNotReadyResponse | CoverageFailureResponse:
        """Run retrieval-only debug under strict readiness and coverage policy."""

        not_ready = self._not_ready_response(require_query=False)
        if not_ready is not None:
            return not_ready
        assert self.state.runtime is not None

        assessment = assess_request_coverage(self.state.runtime.store, request.filters)
        if self.settings.strict_coverage and assessment.coverage_status != "covered":
            return CoverageFailureResponse(
                message="The requested scope is not fully covered by the indexed corpus.",
                coverage_status=assessment.coverage_status,
                indexed_scope=assessment.indexed_scope,
                missing_scope=assessment.missing_scope,
                last_index_refresh_at=self.state.coverage_state.last_index_refresh_at,
            )

        outcome = self.state.runtime.retriever.retrieve(request)
        return RetrievalDebugResponse(
            reason_code=outcome.reason_code,
            retrieved_chunks=list(outcome.retrieved_chunks),
            stage_counts=outcome.stage_counts,
            reranker_applied=outcome.reranker_applied,
            reranker_skipped_reason=outcome.reranker_skipped_reason,
            coverage_status=assessment.coverage_status,
            indexed_scope=assessment.indexed_scope,
            missing_scope=assessment.missing_scope,
            last_index_refresh_at=self.state.coverage_state.last_index_refresh_at,
            timings=RetrievalTimings.model_validate(to_jsonable(outcome.timings)),
        )

    def run_ingest(self, request: IngestRunRequest) -> IngestRunResponse:
        """Run ingestion plus index refresh and persist a coverage snapshot."""

        state_started_at = perf_counter()
        target_scope = self._default_target_scope(
            companies=request.companies,
            form_types=request.form_types,
            annual_limit=request.annual_limit,
            quarterly_limit=request.quarterly_limit,
        )

        ingest_started_at = perf_counter()
        summary = run_ingestion(
            IngestionConfig(
                companies_config=self.settings.companies_config_path,
                data_dir=self.settings.data_dir,
                user_agent=request.user_agent or os.getenv("SEC_USER_AGENT", ""),
                annual_limit=request.annual_limit,
                quarterly_limit=request.quarterly_limit,
                companies=tuple(request.companies),
                form_types=tuple(request.form_types),
                force_refresh=request.force_refresh,
            )
        )
        ingest_ms = (perf_counter() - ingest_started_at) * 1000.0

        index_build: IndexBuildResult | None = None
        index_ms = 0.0
        config = self._load_runtime_retrieval_config()
        store = ProcessedChunkStore.load(self.settings.data_dir)
        if len(store) > 0:
            index_started_at = perf_counter()
            adapter = self.embedding_adapter_factory(config.embedding)
            index_manager = ChromaIndexManager(config, adapter)
            index_build = index_manager.build(store=store, mode=request.index_mode)
            index_ms = (perf_counter() - index_started_at) * 1000.0

        processed_corpus_fingerprint = (
            compute_corpus_fingerprint(store.values())
            if len(store) > 0
            else None
        )
        index_metadata = self._load_index_metadata(config)
        index_status = "fresh" if index_build is not None and index_metadata is not None else "missing"
        coverage_state = build_coverage_state(
            target_scope=target_scope,
            indexed_scope=build_indexed_scope(store) if len(store) > 0 else IndexedScope(),
            last_ingest_run_id=summary.run_id,
            last_ingest_completed_at=datetime.fromisoformat(summary.completed_at) if summary.completed_at else None,
            last_index_refresh_at=datetime.fromisoformat(index_metadata.built_at) if index_metadata is not None else None,
            processed_corpus_fingerprint=processed_corpus_fingerprint,
            indexed_corpus_fingerprint=index_build.corpus_fingerprint if index_build is not None else None,
            index_status=index_status,
        )
        write_coverage_state(
            coverage_state_path(config.index.persist_directory, config.index.collection_name),
            coverage_state,
        )

        refresh_started_at = perf_counter()
        self.refresh_state(load_query_runtime=True)
        refresh_state_ms = (perf_counter() - refresh_started_at) * 1000.0
        total_ms = (perf_counter() - state_started_at) * 1000.0

        log_api_event(
            {
                "event": "ingest_run_completed",
                "run_id": summary.run_id,
                "status": summary.status,
                "successful_filings": summary.successful_filings,
                "failed_filings": summary.failed_filings,
                "index_built": index_build is not None,
                "coverage_status": self.state.coverage_state.coverage_status,
                "index_status": self.state.coverage_state.index_status,
                "timings": {
                    "ingest_ms": ingest_ms,
                    "index_ms": index_ms,
                    "refresh_state_ms": refresh_state_ms,
                    "total_ms": total_ms,
                },
            }
        )

        return IngestRunResponse(
            run_summary=RunSummaryModel.model_validate(to_jsonable(summary)),
            index_build=index_build,
            coverage_state=self.state.coverage_state,
            timings=IngestRunTimings(
                ingest_ms=ingest_ms,
                index_ms=index_ms,
                refresh_state_ms=refresh_state_ms,
                total_ms=total_ms,
            ),
        )

    def run_eval(self, request: EvalRunRequest) -> EvalRunResponse:
        """Run the offline eval harness and return the typed result."""

        started_at = perf_counter()
        eval_config = self._load_runtime_eval_config()
        retrieval_config = self._load_runtime_retrieval_config()
        prompt_catalog = load_prompt_catalog(self.settings.prompts_config_path)
        dataset = load_eval_dataset(eval_config.dataset_path)

        resolved = self._resolve_eval_request(request, eval_config)
        self._validate_eval_request(resolved, request)
        requested_output_dir = resolved["output_dir"]
        output_dir = resolve_output_dir(
            eval_config.output_root,
            (
                resolve_runtime_path(str(requested_output_dir), project_root=self.settings.project_root)
                if requested_output_dir is not None
                else None
            ),
        )
        result = run_eval(
            eval_config=eval_config,
            retrieval_config=retrieval_config,
            prompt_catalog=prompt_catalog,
            dataset=dataset,
            dataset_path=eval_config.dataset_path,
            corpus_path=eval_config.corpus_path,
            ragas_config=eval_config.ragas,
            subset=resolved["subset"],
            mode=resolved["mode"],
            provider=resolved["provider"],
            score_backend=resolved["score_backend"],
            output_dir=output_dir,
        )
        paths = write_eval_artifacts(result, output_dir)
        updated_result = result.model_copy(update={"paths": paths})
        total_ms = (perf_counter() - started_at) * 1000.0

        log_api_event(
            {
                "event": "eval_run_completed",
                "run_id": updated_result.run_id,
                "status": updated_result.status,
                "subset": updated_result.subset,
                "mode": updated_result.mode,
                "provider": updated_result.provider,
                "score_backend": updated_result.score_backend,
                "total_ms": total_ms,
            }
        )
        return EvalRunResponse(
            result=updated_result,
            timings=EvalRunTimings(total_ms=total_ms),
        )

    def _build_runtime(
        self,
        config: RetrievalConfig,
        prompt_catalog: PromptCatalog,
        store: ProcessedChunkStore,
        provider: LLMProvider,
        provider_name: ProviderMode,
    ) -> ApiRuntime:
        adapter = self.embedding_adapter_factory(config.embedding)
        index_manager = ChromaIndexManager(config, adapter)
        collection = index_manager.get_collection()
        dense_retriever = DenseRetriever(config, adapter, store, collection)
        bm25_retriever = BM25Retriever(store)
        reranker = self.reranker_factory(config.reranking) if config.reranking.enabled else None
        prompt_template = PromptManager(prompt_catalog).get_prompt(
            config.prompting.prompt_name,
            expected_version=config.prompting.prompt_version,
        )
        prompt_builder = GroundedPromptBuilder(config.retrieval, config.prompting, prompt_template)
        retriever = HybridRetriever(config, store, dense_retriever, bm25_retriever, reranker)
        pipeline = GroundedAnswerPipeline(config, retriever, prompt_builder, provider)
        return ApiRuntime(
            store=store,
            retriever=retriever,
            pipeline=pipeline,
            provider_name=provider_name,
            reranker=reranker,
        )

    def _load_runtime_retrieval_config(self) -> RetrievalConfig:
        """Load retrieval config with runtime path and provider overrides applied."""

        config = load_retrieval_config(self.settings.retrieval_config_path)
        runtime_settings = load_api_runtime_settings_from_env(project_root=self.settings.project_root)
        persist_directory = runtime_settings.chroma_dir_override or resolve_runtime_path(
            config.index.persist_directory,
            project_root=self.settings.project_root,
        )
        provider_update: dict[str, object] = {}
        if runtime_settings.openai_model_override is not None:
            provider_update["openai_model"] = runtime_settings.openai_model_override

        return config.model_copy(
            update={
                "index": config.index.model_copy(update={"persist_directory": str(persist_directory)}),
                "provider": config.provider.model_copy(update=provider_update) if provider_update else config.provider,
            }
        )

    def _load_runtime_eval_config(self):
        """Load eval config with runtime paths resolved from the repo root."""

        eval_config = load_eval_config(self.settings.eval_config_path)
        return eval_config.model_copy(
            update={
                "dataset_path": str(resolve_runtime_path(eval_config.dataset_path, project_root=self.settings.project_root)),
                "corpus_path": str(resolve_runtime_path(eval_config.corpus_path, project_root=self.settings.project_root)),
                "output_root": str(resolve_runtime_path(eval_config.output_root, project_root=self.settings.project_root)),
            }
        )

    def _default_provider_factory(
        self,
        config: RetrievalConfig,
        allow_mock_fallback: bool,
    ) -> tuple[LLMProvider | None, ProviderMode, bool, str | None]:
        configured_provider: ProviderMode = config.provider.default_name
        if configured_provider == "mock":
            return MockLLMProvider(), "mock", False, None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAILLMProvider(model_name=config.provider.openai_model, api_key=api_key), "openai", False, None
        if allow_mock_fallback:
            return MockLLMProvider(), "mock", True, "missing_openai_api_key"
        return None, "openai", False, "missing_openai_api_key"

    def _default_target_scope(
        self,
        *,
        companies: list[str] | None = None,
        form_types: list[str] | None = None,
        annual_limit: int | None = None,
        quarterly_limit: int | None = None,
    ) -> TargetScope:
        universe = load_company_universe(self.settings.companies_config_path)
        default_companies = [company.ticker for company in universe.enabled_companies()]
        return TargetScope(
            companies=list(companies or default_companies),
            form_types=list(form_types or self.settings.default_form_types),
            annual_limit=self.settings.default_annual_limit if annual_limit is None else annual_limit,
            quarterly_limit=self.settings.default_quarterly_limit if quarterly_limit is None else quarterly_limit,
        )

    def _build_warnings(
        self,
        *,
        store: ProcessedChunkStore,
        coverage_state: CoverageState,
        index_metadata: IndexBuildMetadata | None,
        index_status: str,
    ) -> list[str]:
        warnings: list[str] = []
        if len(store) == 0:
            warnings.append("No processed chunks are available under the configured data directory.")
        elif index_status == "missing":
            warnings.append("Processed chunks are present but the Chroma index is missing.")
        elif index_status == "stale":
            warnings.append("The Chroma index is stale relative to the current processed corpus.")
        if index_metadata is not None and index_metadata.stale_id_count:
            warnings.append("The current Chroma collection reports stale vector IDs from a prior upsert run.")
        if coverage_state.coverage_status == "partial":
            warnings.append("The indexed corpus only partially covers the target scope.")
        if coverage_state.coverage_status == "uninitialized":
            warnings.append("No completed V5 coverage snapshot exists yet.")
        return warnings

    def _preflight_reranker(self, reranker: object, warnings: list[str]) -> bool:
        ensure_loaded = getattr(reranker, "ensure_loaded", None)
        if ensure_loaded is None:
            return True
        try:
            ensure_loaded()
        except Exception as exc:
            warnings.append(f"Cross-encoder reranker preflight failed: {exc}")
            return False
        return True

    def _not_ready_response(self, *, require_query: bool) -> ServiceNotReadyResponse | None:
        state = self.state
        endpoint_ready = (
            (state.query_ready and state.runtime is not None)
            if require_query
            else (state.retrieve_ready and state.runtime is not None)
        )
        if endpoint_ready:
            return None
        return ServiceNotReadyResponse(
            message=(
                "Grounded query execution is not ready."
                if require_query
                else "Retrieval debug execution is not ready."
            ),
            retrieve_ready=state.retrieve_ready,
            query_ready=state.query_ready,
            index_status=state.coverage_state.index_status,
            coverage_status=state.coverage_state.coverage_status,
            indexed_scope=state.coverage_state.indexed_scope,
            last_ingest_completed_at=state.coverage_state.last_ingest_completed_at,
            last_index_refresh_at=state.coverage_state.last_index_refresh_at,
            warnings=state.warnings,
        )

    def _load_index_metadata(self, config: RetrievalConfig) -> IndexBuildMetadata | None:
        sidecar_path = Path(config.index.persist_directory) / f"{config.index.collection_name}.build.json"
        if not sidecar_path.exists():
            return None
        return IndexBuildMetadata.model_validate_json(sidecar_path.read_text(encoding="utf-8"))

    def _collection_exists(self, persist_directory: str | Path, collection_name: str) -> bool:
        if not Path(persist_directory).exists():
            return False
        try:
            client = chromadb.PersistentClient(path=str(persist_directory))
            client.get_collection(name=collection_name)
            return True
        except Exception:
            return False

    def _resolve_eval_request(self, request: EvalRunRequest, eval_config) -> dict[str, object]:
        mode = request.mode or eval_config.default_mode
        subset = request.subset or eval_config.default_subset
        provider = None if mode == "retrieval" else (request.provider or eval_config.default_provider)
        score_backend = None if mode == "retrieval" else (request.score_backend or eval_config.default_score_backend)
        return {
            "subset": subset,
            "mode": mode,
            "provider": provider,
            "score_backend": score_backend,
            "output_dir": request.output_dir,
        }

    def _validate_eval_request(self, resolved: dict[str, object], request: EvalRunRequest) -> None:
        mode = resolved["mode"]
        provider = resolved["provider"]
        score_backend = resolved["score_backend"]

        if mode == "retrieval" and request.provider is not None:
            raise ValueError("mode=retrieval does not accept an explicit provider.")
        if mode == "retrieval" and request.score_backend in {"ragas", "both"}:
            raise ValueError("mode=retrieval does not support ragas scoring.")
        if provider in {"reference", "mock"} and score_backend in {"ragas", "both"}:
            raise ValueError("provider=reference or provider=mock does not support ragas scoring.")
        if provider == "openai" and mode in {"answer", "full"} and not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")


__all__ = ["ApiSettings", "CopilotApiService"]

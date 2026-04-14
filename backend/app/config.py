from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _get_env(name: str, default=None, cast=str, required: bool = False):
    value = os.getenv(name, default)
    if value is None or value == "":
        if required and default is None:
            raise RuntimeError(f"Missing required env var: {name}")
        return default
    try:
        return cast(value)
    except Exception as exc:
        raise RuntimeError(f"Invalid value for env var {name}: {value}") from exc


def _bool_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalize_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _normalize_app_env(value: str) -> str:
    value = str(value or "development").strip().lower()
    return value or "development"


def _default_require_auth_for_env(app_env: str) -> bool:
    return app_env not in {"development", "dev", "local", "test"}


@dataclass(frozen=True)
class Settings:
    app_env: str = _get_env("APP_ENV", "development")
    kb_app_token: str | None = _get_env("KB_APP_TOKEN", default=None)
    kb_admin_token: str | None = _get_env("KB_ADMIN_TOKEN", default=None)
    kb_require_auth: bool = False
    kb_require_auth_explicit: bool = False

    # OpenAI
    # Keep optional so the app can boot and answer `/api/health` even when secrets
    # are not yet configured in a deployment environment. Use `/api/health/deps`
    # to verify whether required dependency config is present.
    openai_api_key: str | None = _get_env("OPENAI_API_KEY", default=None)
    openai_base_url: str | None = _get_env("OPENAI_BASE_URL", default=None)
    openai_timeout_seconds: float = _get_env("OPENAI_TIMEOUT_SECONDS", 90, float)
    openai_max_retries: int = _get_env("OPENAI_MAX_RETRIES", 1, int)

    # Models
    openai_embedding_model: str = _get_env(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    openai_embedding_batch_size: int = _get_env("OPENAI_EMBEDDING_BATCH_SIZE", 128, int)
    openai_embedding_dimensions: int | None = _get_env(
        "OPENAI_EMBEDDING_DIMENSIONS",
        default=None,
        cast=lambda v: int(v) if v is not None and v != "" else None,
    )
    openai_generation_model: str = _get_env(
        "OPENAI_GENERATION_MODEL", "gpt-5.4-nano-2026-03-17"
    )
    openai_tree_model: str = _get_env(
        "OPENAI_TREE_MODEL", "gpt-5.4-nano-2026-03-17"
    )
    openai_tree_reasoning_effort: str = _get_env(
        "OPENAI_TREE_REASONING_EFFORT", "high"
    )
    openai_tree_search_model: str = _get_env(
        "OPENAI_TREE_SEARCH_MODEL", "gpt-5.4-nano-2026-03-17"
    )
    openai_rewrite_model: str = _get_env("OPENAI_REWRITE_MODEL", "gpt-4o-mini")

    # Pinecone
    pinecone_api_key: str | None = _get_env("PINECONE_API_KEY", default=None)
    pinecone_index: str | None = _get_env("PINECONE_INDEX", default=None)
    pinecone_host: str | None = _get_env("PINECONE_HOST", default=None)
    pinecone_timeout_seconds: float = _get_env("PINECONE_TIMEOUT_SECONDS", 30, float)
    pinecone_upsert_batch_size: int = _get_env("PINECONE_UPSERT_BATCH_SIZE", 100, int)

    # Storage
    data_dir: str = _get_env("DATA_DIR", "./data")
    docs_dir: str | None = _get_env("DOCS_DIR", default=None)
    tree_dir: str | None = _get_env("TREE_DIR", default=None)
    local_storage_dir: str | None = _get_env("LOCAL_STORAGE_DIR", default=None)
    cache_dir: str = _get_env("CACHE_DIR", "./data/cache")

    # Routing
    page_tree_threshold: int = _get_env("PAGE_TREE_THRESHOLD", 50, int)

    # Standard chunking (token-based)
    chunk_target_tokens: int = _get_env("CHUNK_TARGET_TOKENS", 400, int)
    chunk_min_tokens: int = _get_env("CHUNK_MIN_TOKENS", 350, int)
    chunk_max_tokens: int = _get_env("CHUNK_MAX_TOKENS", 450, int)
    chunk_overlap_tokens: int = _get_env("CHUNK_OVERLAP_TOKENS", 40, int)
    chunking_logic_version: str = _get_env("CHUNKING_LOGIC_VERSION", "v3_heading_aware")

    # OCR config mirrored here for index-versioning and operational visibility.
    mistral_base_url: str = _get_env("MISTRAL_BASE_URL", "https://api.mistral.ai")
    mistral_ocr_model: str = _get_env("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
    mistral_ocr_timeout_seconds: float = _get_env("MISTRAL_OCR_TIMEOUT_SECONDS", 120.0, float)
    mistral_ocr_delete_uploaded_file: bool = _get_env(
        "MISTRAL_OCR_DELETE_UPLOADED_FILE", "true", cast=_bool_env
    )

    # Doc selection via multi-vector doc summaries
    doc_summary_namespace: str = _get_env("DOC_SUMMARY_NAMESPACE", "__doc_summaries")
    doc_summary_top_k: int = _get_env("DOC_SUMMARY_TOP_K", 3, int)
    doc_summary_match_top_k: int = _get_env("DOC_SUMMARY_MATCH_TOP_K", 12, int)
    doc_summary_strategy_version: str = _get_env(
        "DOC_SUMMARY_STRATEGY_VERSION", "multi_vector_v1"
    )
    doc_strong_min_score: float = _get_env("DOC_STRONG_MIN_SCORE", 0.35, float)
    doc_strong_min_ratio: float = _get_env("DOC_STRONG_MIN_RATIO", 1.15, float)
    doc_rewrite_max_attempts: int = _get_env("DOC_REWRITE_MAX_ATTEMPTS", 1, int)

    # Retrieval / rerank / context
    retrieve_top_k: int = _get_env("RETRIEVE_TOP_K", 60, int)
    rerank_candidate_k: int = _get_env("RERANK_CANDIDATE_K", 50, int)
    rerank_top_k: int = _get_env("RERANK_TOP_K", 10, int)
    max_context_tokens: int = _get_env("MAX_CONTEXT_TOKENS", 4500, int)
    # Cap any single context item to avoid one long chunk starving the entire context budget.
    context_max_item_tokens: int = _get_env("CONTEXT_MAX_ITEM_TOKENS", 1200, int)
    low_confidence_threshold: float = _get_env("LOW_CONFIDENCE_THRESHOLD", 0.35, float)
    context_use_mmr: bool = _get_env("CONTEXT_USE_MMR", "true", cast=_bool_env)
    context_mmr_lambda: float = _get_env("CONTEXT_MMR_LAMBDA", 0.7, float)
    # Feature flag: when false, skip final answer generation and return chunks only.
    rag_generate_answers_enabled: bool = _get_env(
        "RAG_GENERATE_ANSWERS_ENABLED", "true", cast=_bool_env
    )

    # Tree indexing + retrieval (PageIndex-style caps)
    toc_check_page_num: int = _get_env("TOC_CHECK_PAGE_NUM", 20, int)
    max_pages_per_node: int = _get_env("MAX_PAGES_PER_NODE", 10, int)
    max_tokens_per_node: int = _get_env("MAX_TOKENS_PER_NODE", 20000, int)
    overlap_pages: int = _get_env("OVERLAP_PAGES", 1, int)
    tree_section_top_k: int = _get_env("TREE_SECTION_TOP_K", 12, int)
    tree_pageindex_toc_enabled: bool = _get_env("TREE_PAGEINDEX_TOC_ENABLED", "true", cast=_bool_env)
    tree_verify_toc_enabled: bool = _get_env("TREE_VERIFY_TOC_ENABLED", "true", cast=_bool_env)
    tree_verify_toc_max_items: int = _get_env("TREE_VERIFY_TOC_MAX_ITEMS", 25, int)
    tree_fix_toc_enabled: bool = _get_env("TREE_FIX_TOC_ENABLED", "true", cast=_bool_env)
    tree_fix_toc_max_attempts: int = _get_env("TREE_FIX_TOC_MAX_ATTEMPTS", 2, int)
    tree_fix_toc_window_pages: int = _get_env("TREE_FIX_TOC_WINDOW_PAGES", 8, int)
    tree_split_large_nodes_enabled: bool = _get_env("TREE_SPLIT_LARGE_NODES_ENABLED", "true", cast=_bool_env)
    tree_split_large_nodes_max_depth: int = _get_env("TREE_SPLIT_LARGE_NODES_MAX_DEPTH", 2, int)
    tree_node_summaries_enabled: bool = _get_env("TREE_NODE_SUMMARIES_ENABLED", "true", cast=_bool_env)
    tree_node_summary_max_tokens: int = _get_env("TREE_NODE_SUMMARY_MAX_TOKENS", 1200, int)
    tree_node_summary_token_threshold: int = _get_env("TREE_NODE_SUMMARY_TOKEN_THRESHOLD", 200, int)
    tree_node_summary_max_workers: int = _get_env("TREE_NODE_SUMMARY_MAX_WORKERS", 4, int)
    tree_node_summary_max_nodes: int = _get_env("TREE_NODE_SUMMARY_MAX_NODES", 200, int)
    tree_node_selection_mode: str = _get_env("TREE_NODE_SELECTION_MODE", "vector_then_llm")
    tree_node_selection_candidate_k: int = _get_env("TREE_NODE_SELECTION_CANDIDATE_K", 24, int)
    tree_node_selection_top_n: int = _get_env("TREE_NODE_SELECTION_TOP_N", 6, int)
    tree_node_selection_max_tree_nodes: int = _get_env("TREE_NODE_SELECTION_MAX_TREE_NODES", 80, int)

    # Debugging / SSE
    rag_debug: bool = _get_env("RAG_DEBUG", "false", cast=_bool_env)
    log_payloads: bool = _get_env("RAG_LOG_PAYLOADS", "false", cast=_bool_env)
    # Log only the final chat answer/citations (no prompts, no vectors).
    log_chat_answer: bool = _get_env("RAG_LOG_CHAT_ANSWER", "false", cast=_bool_env)
    # Max characters per logged string field (prompts/responses). Set <=0 to disable truncation.
    log_payload_max_chars: int = _get_env("RAG_LOG_PAYLOAD_MAX_CHARS", 8000, int)
    sse_heartbeat_seconds: float = _get_env("SSE_HEARTBEAT_SECONDS", 15.0, float)

    # Versioning
    index_schema_version: str = _get_env("INDEX_SCHEMA_VERSION", "v4")

    def __post_init__(self) -> None:
        app_env = _normalize_app_env(self.app_env)
        raw_require_auth = os.getenv("KB_REQUIRE_AUTH")
        explicit_require_auth = raw_require_auth is not None and raw_require_auth != ""
        require_auth = (
            _bool_env(raw_require_auth)
            if explicit_require_auth
            else _default_require_auth_for_env(app_env)
        )

        object.__setattr__(self, "app_env", app_env)
        object.__setattr__(self, "kb_app_token", _normalize_optional_str(self.kb_app_token))
        object.__setattr__(self, "kb_admin_token", _normalize_optional_str(self.kb_admin_token))
        object.__setattr__(self, "kb_require_auth", require_auth)
        object.__setattr__(self, "kb_require_auth_explicit", explicit_require_auth)


settings = Settings()


def validate_settings(s: Settings) -> None:
    if not s.kb_require_auth:
        return

    missing: list[str] = []
    if not s.kb_app_token:
        missing.append("KB_APP_TOKEN")
    if not s.kb_admin_token:
        missing.append("KB_ADMIN_TOKEN")
    if not missing:
        return

    if s.kb_require_auth_explicit:
        reason = "KB_REQUIRE_AUTH=true"
    else:
        reason = f"APP_ENV={s.app_env} defaults auth to required outside development"

    missing_list = ", ".join(missing)
    raise RuntimeError(
        f"Authentication is required ({reason}) but missing required env var(s): {missing_list}"
    )


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def resolve_data_dir(s: Settings) -> Path:
    return resolve_path(s.data_dir)


def resolve_docs_dir(s: Settings) -> Path:
    if s.docs_dir:
        return resolve_path(s.docs_dir)
    return resolve_data_dir(s) / "docs"


def resolve_tree_dir(s: Settings) -> Path:
    if s.tree_dir:
        return resolve_path(s.tree_dir)
    return resolve_data_dir(s) / "tree_index"


def resolve_storage_dir(s: Settings) -> Path:
    if s.local_storage_dir:
        return resolve_path(s.local_storage_dir)
    return resolve_data_dir(s) / "uploads"


def resolve_cache_dir(s: Settings) -> Path:
    return resolve_path(s.cache_dir)

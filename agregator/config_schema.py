"""Canonical schema registry for all application settings.

This module is the single source of truth for:
- runtime field names and types,
- UI visibility and mutability,
- restart requirements and constraints,
- env aliases and legacy json aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FieldDef:
    """Descriptor for a single configuration parameter."""

    name: str
    type_hint: str
    default: Any = None
    doc: str = ""
    env_key: str | None = None
    runtime_mutable: bool = True
    json_key: str | None = None
    group: str = "general"
    visibility: str = "ui_expert"  # ui_safe | ui_expert | env_only
    restart_required: bool = False
    constraints: Dict[str, Any] = field(default_factory=dict)
    ui_component: str = "input"  # input | select | switch | json
    depends_on: Optional[Dict[str, Any]] = None
    risk_level: str = "medium"  # low | medium | high
    aliases: tuple[str, ...] = tuple()


# ---------------------------------------------------------------------------
# Field definitions grouped by domain
# ---------------------------------------------------------------------------

_LLM_FIELDS: list[FieldDef] = [
    FieldDef("lmstudio_api_base", "str", "", "OpenAI-compatible API base URL",
             env_key="LMSTUDIO_API_BASE", group="llm", visibility="ui_safe"),
    FieldDef("lmstudio_model", "str", "", "Model name for default LLM route",
             env_key="LMSTUDIO_MODEL", group="llm", visibility="ui_safe"),
    FieldDef("lmstudio_api_key", "str", "", "API key for default LLM route",
             env_key="LMSTUDIO_API_KEY", group="llm", visibility="ui_expert", risk_level="high"),
    FieldDef(
        "lm_default_provider",
        "str",
        "openai",
        "Default LLM provider for fallback route",
        env_key="LM_DEFAULT_PROVIDER",
        group="llm",
        visibility="ui_safe",
        constraints={"enum": ["openai", "ollama"]},
        ui_component="select",
        aliases=("LM_PROVIDER",),
    ),
    FieldDef(
        "lm_max_input_chars",
        "int",
        4000,
        "Maximum input length passed to LLM",
        env_key="LM_MAX_INPUT_CHARS",
        group="llm",
        visibility="ui_expert",
        constraints={"min": 500, "max": 200000},
    ),
    FieldDef(
        "lm_max_output_tokens",
        "int",
        256,
        "Maximum output tokens requested from LLM",
        env_key="LM_MAX_OUTPUT_TOKENS",
        group="llm",
        visibility="ui_expert",
        constraints={"min": 16, "max": 16000},
    ),
    FieldDef("ai_rerank_llm", "bool", False, "Use LLM for re-ranking",
             env_key="AI_RERANK_LLM", group="llm", visibility="ui_expert", ui_component="switch"),
    FieldDef(
        "azure_openai_api_version",
        "str",
        "2024-02-15-preview",
        "Azure OpenAI API version for compatible endpoints",
        env_key="AZURE_OPENAI_API_VERSION",
        group="llm",
        visibility="ui_expert",
    ),
]

_LLM_POOL_FIELDS: list[FieldDef] = [
    FieldDef("llm_pool_global_concurrency", "int", 4, "Global LLM worker concurrency",
             env_key="LLM_POOL_GLOBAL_CONCURRENCY", group="llm_pool", visibility="ui_expert",
             constraints={"min": 1, "max": 128}, restart_required=False),
    FieldDef("llm_pool_per_user_concurrency", "int", 1, "Per-user LLM worker concurrency",
             env_key="LLM_POOL_PER_USER_CONCURRENCY", group="llm_pool", visibility="ui_expert",
             constraints={"min": 1, "max": 32}, restart_required=False),
    FieldDef("llm_queue_max_size", "int", 100, "Max LLM queue size",
             env_key="LLM_QUEUE_MAX_SIZE", group="llm_pool", visibility="ui_expert",
             constraints={"min": 1, "max": 50000}),
    FieldDef("llm_queue_per_user_max", "int", 10, "Max queued LLM requests per user",
             env_key="LLM_QUEUE_PER_USER_MAX", group="llm_pool", visibility="ui_expert",
             constraints={"min": 1, "max": 5000}),
    FieldDef("llm_request_timeout_sec", "int", 90, "LLM request timeout in seconds",
             env_key="LLM_REQUEST_TIMEOUT_SEC", group="llm_pool", visibility="ui_expert",
             constraints={"min": 1, "max": 1800}),
    FieldDef("llm_retry_count", "int", 1, "Retry count for transient LLM failures",
             env_key="LLM_RETRY_COUNT", group="llm_pool", visibility="ui_expert",
             constraints={"min": 0, "max": 10}),
    FieldDef("llm_retry_backoff_ms", "int", 500, "Retry backoff in milliseconds",
             env_key="LLM_RETRY_BACKOFF_MS", group="llm_pool", visibility="ui_expert",
             constraints={"min": 0, "max": 30000}),
]

_LMSTUDIO_MGMT_FIELDS: list[FieldDef] = [
    FieldDef(
        "lmstudio_idle_unload_minutes",
        "int",
        0,
        "Unload LM Studio models after N minutes of global inactivity (0 disables)",
        env_key="LMSTUDIO_IDLE_UNLOAD_MINUTES",
        group="llm_pool",
        visibility="ui_expert",
        constraints={"min": 0, "max": 1440},
    ),
]

_RAG_FIELDS: list[FieldDef] = [
    FieldDef("rag_embedding_backend", "str", "auto", "Embedding backend (auto/lm-studio/sentence-transformers/hash)",
             env_key="RAG_EMBEDDING_BACKEND", group="rag", visibility="ui_safe",
             constraints={"enum": ["auto", "lm-studio", "sentence-transformers", "hash"]}, ui_component="select"),
    FieldDef("rag_embedding_model", "str", "intfloat/multilingual-e5-large", "Embedding model name",
             env_key="RAG_EMBEDDING_MODEL", group="rag", visibility="ui_safe"),
    FieldDef("rag_embedding_dim", "int", 384, "Embedding vector dimension",
             env_key="RAG_EMBEDDING_DIM", group="rag", visibility="ui_expert", constraints={"min": 1, "max": 8192}),
    FieldDef("rag_embedding_batch_size", "int", 32, "Batch size for embedding requests",
             env_key="RAG_EMBEDDING_BATCH_SIZE", json_key="rag_embedding_batch", group="rag",
             visibility="ui_expert", constraints={"min": 1, "max": 2048}, aliases=("RAG_EMBEDDING_BATCH",)),
    FieldDef("rag_embedding_device", "Optional[str]", None, "Device for local embedding",
             env_key="RAG_EMBEDDING_DEVICE", group="rag", visibility="ui_expert"),
    FieldDef("rag_embedding_endpoint", "str", "", "Custom embedding API endpoint",
             env_key="RAG_EMBEDDING_ENDPOINT", group="rag", visibility="ui_expert"),
    FieldDef("rag_embedding_api_key", "str", "", "API key for embedding endpoint",
             env_key="RAG_EMBEDDING_API_KEY", group="rag", visibility="ui_expert", risk_level="high"),
    FieldDef("rag_rerank_backend", "str", "none", "Re-ranking backend (none/cross-encoder)",
             env_key="RAG_RERANK_BACKEND", group="rag", visibility="ui_expert"),
    FieldDef("rag_rerank_model", "str", "", "Re-ranking model name",
             env_key="RAG_RERANK_MODEL", group="rag", visibility="ui_expert"),
    FieldDef("rag_rerank_device", "Optional[str]", None, "Device for re-ranker",
             env_key="RAG_RERANK_DEVICE", group="rag", visibility="ui_expert"),
    FieldDef("rag_rerank_batch_size", "int", 16, "Batch size for re-ranker",
             env_key="RAG_RERANK_BATCH_SIZE", group="rag", visibility="ui_expert"),
    FieldDef("rag_rerank_max_length", "int", 512, "Max input length for re-ranker",
             env_key="RAG_RERANK_MAX_LENGTH", group="rag", visibility="ui_expert"),
    FieldDef("rag_rerank_max_chars", "int", 1200, "Max chars per chunk for re-ranker",
             env_key="RAG_RERANK_MAX_CHARS", group="rag", visibility="ui_expert"),
    FieldDef("ai_query_variants_max", "int", 0, "Max query variants for AI search",
             env_key="AI_QUERY_VARIANTS_MAX", group="rag", visibility="ui_expert", constraints={"min": 0, "max": 6}),
    FieldDef("ai_rag_retry_enabled", "bool", True, "Enable RAG retry on low confidence",
             env_key="AI_RAG_RETRY_ENABLED", group="rag", visibility="ui_expert", ui_component="switch"),
    FieldDef("ai_rag_retry_threshold", "float", 0.6, "RAG retry confidence threshold",
             env_key="AI_RAG_RETRY_THRESHOLD", group="rag", visibility="ui_expert", constraints={"min": 0.0, "max": 1.0}),
]

_DOC_CHAT_FIELDS: list[FieldDef] = [
    FieldDef("doc_chat_chunk_max_tokens", "int", 700, "Max tokens per doc-chat chunk",
             env_key="DOC_CHAT_CHUNK_MAX_TOKENS", group="doc_chat", visibility="ui_safe", constraints={"min": 16}),
    FieldDef("doc_chat_chunk_overlap", "int", 120, "Token overlap between chunks",
             env_key="DOC_CHAT_CHUNK_OVERLAP", group="doc_chat", visibility="ui_safe", constraints={"min": 0}),
    FieldDef("doc_chat_chunk_min_tokens", "int", 80, "Minimum chunk token count",
             env_key="DOC_CHAT_CHUNK_MIN_TOKENS", group="doc_chat", visibility="ui_safe", constraints={"min": 1}),
    FieldDef("doc_chat_max_chunks", "int", 0, "Max chunks to index (0 = unlimited)",
             env_key="DOC_CHAT_MAX_CHUNKS", group="doc_chat", visibility="ui_expert", constraints={"min": 0}),
    FieldDef("doc_chat_fallback_chunks", "int", 0, "Fallback chunks when score is low",
             env_key="DOC_CHAT_FALLBACK_CHUNKS", group="doc_chat", visibility="ui_expert", constraints={"min": 0}),
    FieldDef("doc_chat_image_min_width", "int", 32, "Minimum image width for doc-chat",
             env_key="DOC_CHAT_IMAGE_MIN_WIDTH", group="doc_chat", visibility="ui_expert", constraints={"min": 0}),
    FieldDef("doc_chat_image_min_height", "int", 32, "Minimum image height for doc-chat",
             env_key="DOC_CHAT_IMAGE_MIN_HEIGHT", group="doc_chat", visibility="ui_expert", constraints={"min": 0}),
]

_SCAN_FIELDS: list[FieldDef] = [
    FieldDef("scan_root", "Path", ".", "Root directory for file scanning",
             env_key="SCAN_ROOT", group="scan", visibility="ui_safe", runtime_mutable=False, restart_required=True),
    FieldDef("extract_text", "bool", True, "Extract text from documents during scan",
             env_key="EXTRACT_TEXT", group="scan", visibility="ui_safe", ui_component="switch"),
    FieldDef("import_subdir", "str", "import", "Subdirectory for imports",
             env_key="IMPORT_SUBDIR", group="scan", visibility="ui_safe"),
    FieldDef("collections_in_separate_dirs", "bool", True, "Separate directories per collection",
             env_key="COLLECTIONS_IN_SEPARATE_DIRS", group="scan", visibility="ui_safe", ui_component="switch"),
    FieldDef("collection_type_subdirs", "bool", False, "Organize by material type in subdirs",
             env_key="COLLECTION_TYPE_SUBDIRS", group="scan", visibility="ui_expert",
             ui_component="switch", depends_on={"collections_in_separate_dirs": True}),
    FieldDef("move_on_rename", "bool", True, "Move file when renaming",
             env_key="MOVE_ON_RENAME", group="scan", visibility="ui_safe", ui_component="switch"),
    FieldDef("default_use_llm", "bool", True, "Use LLM by default during scan",
             env_key="DEFAULT_USE_LLM", group="scan", visibility="ui_safe", ui_component="switch"),
    FieldDef("default_prune", "bool", True, "Prune orphaned files during scan",
             env_key="DEFAULT_PRUNE", group="scan", visibility="ui_safe", ui_component="switch"),
]

_OCR_FIELDS: list[FieldDef] = [
    FieldDef("ocr_langs", "str", "rus+eng", "OCR languages",
             env_key="OCR_LANGS", json_key="ocr_langs_cfg", group="ocr", visibility="ui_safe",
             aliases=("OCR_LANGS_CFG",)),
    FieldDef("pdf_ocr_pages", "int", 3, "Number of PDF pages to OCR",
             env_key="PDF_OCR_PAGES", json_key="pdf_ocr_pages_cfg", group="ocr", visibility="ui_safe",
             constraints={"min": 1, "max": 50}, aliases=("PDF_OCR_PAGES_CFG",)),
    FieldDef("always_ocr_first_page_dissertation", "bool", True,
             "Always OCR first page of dissertations", env_key="OCR_DISS_FIRST_PAGE",
             group="ocr", visibility="ui_expert", ui_component="switch"),
]

_TRANSCRIPTION_FIELDS: list[FieldDef] = [
    FieldDef("transcribe_enabled", "bool", True, "Enable audio transcription",
             env_key="TRANSCRIBE_ENABLED", group="transcription", visibility="ui_safe", ui_component="switch"),
    FieldDef("transcribe_backend", "str", "faster-whisper", "Transcription backend",
             env_key="TRANSCRIBE_BACKEND", group="transcription", visibility="ui_safe"),
    FieldDef("transcribe_model_path", "str", "", "Path to transcription model",
             env_key="TRANSCRIBE_MODEL_PATH", group="transcription", visibility="ui_safe",
             aliases=("FASTER_WHISPER_DEFAULT_MODEL",)),
    FieldDef("transcribe_language", "str", "ru", "Default transcription language",
             env_key="TRANSCRIBE_LANGUAGE", group="transcription", visibility="ui_safe"),
    FieldDef("summarize_audio", "bool", True, "Summarize transcribed audio",
             env_key="SUMMARIZE_AUDIO", group="transcription", visibility="ui_safe", ui_component="switch"),
    FieldDef("audio_keywords_llm", "bool", False, "Extract keywords from audio via LLM",
             env_key="AUDIO_KEYWORDS_LLM", group="transcription", visibility="ui_expert", ui_component="switch"),
]

_VISION_FIELDS: list[FieldDef] = [
    FieldDef("images_vision_enabled", "bool", True, "Enable image analysis via LLM vision",
             env_key="IMAGES_VISION_ENABLED", group="vision", visibility="ui_safe", ui_component="switch"),
    FieldDef("keywords_to_tags_enabled", "bool", True, "Convert keywords to tags",
             env_key="KEYWORDS_TO_TAGS_ENABLED", group="vision", visibility="ui_safe", ui_component="switch"),
]

_CACHE_FIELDS: list[FieldDef] = [
    FieldDef("llm_cache_enabled", "bool", True, "Enable LLM response caching",
             env_key="LLM_CACHE_ENABLED", group="cache", visibility="ui_expert", ui_component="switch"),
    FieldDef("llm_cache_ttl_seconds", "int", 600, "LLM cache TTL in seconds",
             env_key="LLM_CACHE_TTL_SECONDS", group="cache", visibility="ui_expert", constraints={"min": 1}),
    FieldDef("llm_cache_max_items", "int", 256, "Max items in LLM cache",
             env_key="LLM_CACHE_MAX_ITEMS", group="cache", visibility="ui_expert", constraints={"min": 1}),
    FieldDef("llm_cache_only_mode", "bool", False, "LLM cache-only mode (no API calls)",
             env_key="LLM_CACHE_ONLY_MODE", group="cache", visibility="ui_expert", ui_component="switch", risk_level="high"),
    FieldDef("search_cache_enabled", "bool", True, "Enable search result caching",
             env_key="SEARCH_CACHE_ENABLED", group="cache", visibility="ui_expert", ui_component="switch"),
    FieldDef("search_cache_ttl_seconds", "int", 90, "Search cache TTL in seconds",
             env_key="SEARCH_CACHE_TTL_SECONDS", group="cache", visibility="ui_expert", constraints={"min": 1}),
    FieldDef("search_cache_max_items", "int", 64, "Max items in search cache",
             env_key="SEARCH_CACHE_MAX_ITEMS", group="cache", visibility="ui_expert", constraints={"min": 1}),
]

_FACET_FIELDS: list[FieldDef] = [
    FieldDef("search_facet_tag_keys", "Optional[List[str]]", None,
             "Allowed tag keys for search facets", env_key="SEARCH_FACET_TAG_KEYS",
             group="facets", visibility="ui_expert", ui_component="json"),
    FieldDef("graph_facet_tag_keys", "Optional[List[str]]", None,
             "Allowed tag keys for graph facets", env_key="GRAPH_FACET_TAG_KEYS",
             group="facets", visibility="ui_expert", ui_component="json"),
    FieldDef("search_facet_include_types", "bool", True,
             "Include material types in search facets", env_key="SEARCH_FACET_INCLUDE_TYPES",
             group="facets", visibility="ui_expert", ui_component="switch"),
]

_TYPE_DETECTION_FIELDS: list[FieldDef] = [
    FieldDef("type_detect_flow", "str", "rules", "Type detection flow (rules/llm/rules+llm)",
             env_key="TYPE_DETECT_FLOW", group="type_detection", visibility="ui_safe"),
    FieldDef("type_llm_override", "bool", False, "Allow LLM to override rule-based type",
             env_key="TYPE_LLM_OVERRIDE", group="type_detection", visibility="ui_expert", ui_component="switch"),
    FieldDef("type_dirs", "Dict[str, str]", None, "Type-to-directory mapping",
             runtime_mutable=True, group="type_detection", visibility="ui_expert", ui_component="json"),
    FieldDef("material_types", "List[Dict]", None, "Material type profile definitions",
             runtime_mutable=True, group="type_detection", visibility="ui_expert", ui_component="json"),
]

_FEEDBACK_FIELDS: list[FieldDef] = [
    FieldDef("feedback_train_interval_hours", "float", 0.0,
             "Feedback model retraining interval (hours)", env_key="AI_FEEDBACK_TRAIN_INTERVAL_HOURS",
             group="feedback", visibility="ui_expert", constraints={"min": 0.0}),
    FieldDef("feedback_train_cutoff_days", "int", 90,
             "Feedback data cutoff in days", env_key="AI_FEEDBACK_TRAIN_CUTOFF_DAYS",
             group="feedback", visibility="ui_expert", constraints={"min": 1}),
]

_PROMPT_FIELDS: list[FieldDef] = [
    FieldDef(
        "prompts",
        "Dict[str, str]",
        None,
        "Prompt overrides for LLM workflows",
        group="prompts",
        visibility="ui_expert",
        ui_component="json",
        risk_level="high",
    ),
]

_SYSTEM_FIELDS: list[FieldDef] = [
    FieldDef("database_url", "str", "", "Database DSN",
             env_key="DATABASE_URL", runtime_mutable=False, visibility="env_only", restart_required=True,
             group="system", risk_level="high"),
    FieldDef("sqlalchemy_database_uri", "str", "", "SQLAlchemy DB URI override",
             env_key="SQLALCHEMY_DATABASE_URI", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system", risk_level="high"),
    FieldDef("flask_secret_key", "str", "", "Flask session secret",
             env_key="FLASK_SECRET_KEY", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="security", risk_level="high"),
    FieldDef("pipeline_api_key", "str", "", "Pipeline API key",
             env_key="PIPELINE_API_KEY", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="security", risk_level="high"),
    FieldDef("max_content_length", "int", 52428800, "Max upload body size in bytes",
             env_key="MAX_CONTENT_LENGTH", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("http_timeout", "float", 120.0, "HTTP total timeout",
             env_key="AG_HTTP_TIMEOUT", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("http_connect_timeout", "float", 10.0, "HTTP connect timeout",
             env_key="AG_HTTP_CONNECT_TIMEOUT", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("http_retries", "int", 3, "HTTP retries",
             env_key="AG_HTTP_RETRIES", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("http_backoff_factor", "float", 0.5, "HTTP retry backoff factor",
             env_key="AG_HTTP_BACKOFF", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("log_level", "str", "INFO", "Logging level",
             env_key="LOG_LEVEL", runtime_mutable=False, visibility="env_only",
             restart_required=False, group="system"),
    FieldDef("sentry_dsn", "str", "", "Sentry DSN",
             env_key="SENTRY_DSN", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("sentry_environment", "str", "local", "Sentry environment",
             env_key="SENTRY_ENVIRONMENT", runtime_mutable=False, visibility="env_only",
             restart_required=True, group="system"),
    FieldDef("settings_hub_v2_enabled", "bool", True, "Feature flag for new settings hub UI",
             env_key="SETTINGS_HUB_V2_ENABLED", runtime_mutable=False, visibility="env_only",
             restart_required=False, group="system"),
]

# ---------------------------------------------------------------------------
# Aggregate registry
# ---------------------------------------------------------------------------

CONFIG_FIELDS: list[FieldDef] = (
    _LLM_FIELDS
    + _LLM_POOL_FIELDS
    + _LMSTUDIO_MGMT_FIELDS
    + _RAG_FIELDS
    + _DOC_CHAT_FIELDS
    + _SCAN_FIELDS
    + _OCR_FIELDS
    + _TRANSCRIPTION_FIELDS
    + _VISION_FIELDS
    + _CACHE_FIELDS
    + _FACET_FIELDS
    + _TYPE_DETECTION_FIELDS
    + _FEEDBACK_FIELDS
    + _PROMPT_FIELDS
    + _SYSTEM_FIELDS
)

# Index for fast lookups
FIELDS_BY_NAME: Dict[str, FieldDef] = {f.name: f for f in CONFIG_FIELDS}
FIELDS_BY_ENV: Dict[str, FieldDef] = {}
for _field in CONFIG_FIELDS:
    if _field.env_key:
        FIELDS_BY_ENV[_field.env_key] = _field
    for _alias in _field.aliases:
        FIELDS_BY_ENV[_alias] = _field
FIELDS_BY_GROUP: Dict[str, list[FieldDef]] = {}
for _f in CONFIG_FIELDS:
    FIELDS_BY_GROUP.setdefault(_f.group, []).append(_f)


def runtime_mutable_field_names() -> list[str]:
    """Return names of fields that can be changed at runtime."""
    return [f.name for f in CONFIG_FIELDS if f.runtime_mutable]


def field_snapshot_key(field: FieldDef) -> str:
    """Return RuntimeSettings snapshot key for a schema field."""
    if field.name == "rag_embedding_batch_size":
        return "RAG_EMBEDDING_BATCH"
    if field.name == "ocr_langs":
        return "OCR_LANGS_CFG"
    if field.name == "pdf_ocr_pages":
        return "PDF_OCR_PAGES_CFG"
    return (field.json_key or field.name).upper()


def field_api_key(field: FieldDef) -> str:
    """Return UI API key for field in /api/settings payload."""
    return field.json_key or field.name

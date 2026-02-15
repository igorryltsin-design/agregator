"""Вспомогательный модуль для управления runtime-настройками Agregator.

Инкапсулирует параметры, которые могут меняться во время работы приложения и
сохраняться на диск. Предоставляет единый интерфейс для обновления, загрузки и
применения настроек к Flask-приложению и окружению.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from .config_schema import CONFIG_FIELDS, FIELDS_BY_ENV, FIELDS_BY_NAME, field_snapshot_key

if False:  # pragma: no cover - только для подсказок типов без циклического импорта
    from flask import Flask  # noqa: F401
    from .config import AppConfig  # noqa: F401


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _sanitize_list(values: Optional[Iterable[Any]]) -> Optional[List[str]]:
    if values is None:
        return None
    items = [str(v or "").strip() for v in values]
    items = [item for item in items if item]
    return items


def _sanitize_subdir(name: str) -> str:
    return str(name or "").strip().strip("/\\")


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


_FIELD_BY_SNAPSHOT_KEY: Dict[str, Any] = {field_snapshot_key(f): f for f in CONFIG_FIELDS}
_FIELD_BY_JSON_KEY: Dict[str, Any] = {str((f.json_key or f.name)).lower(): f for f in CONFIG_FIELDS}
_FIELD_TO_ATTR: Dict[str, str] = {
    "ocr_langs": "ocr_langs_cfg",
    "pdf_ocr_pages": "pdf_ocr_pages_cfg",
}


def _resolve_field_by_input_key(key: str):
    token = str(key or "").strip()
    if not token:
        return None
    lower = token.lower()
    if lower in FIELDS_BY_NAME:
        return FIELDS_BY_NAME[lower]
    if lower in _FIELD_BY_JSON_KEY:
        return _FIELD_BY_JSON_KEY[lower]
    upper = token.upper()
    if upper in _FIELD_BY_SNAPSHOT_KEY:
        return _FIELD_BY_SNAPSHOT_KEY[upper]
    if upper in FIELDS_BY_ENV:
        return FIELDS_BY_ENV[upper]
    return None


def _coerce_schema_value(field_def, raw: Any) -> Any:
    type_hint = (field_def.type_hint or "").lower()
    if raw is None:
        return None
    if "bool" in type_hint:
        return _coerce_bool(raw, default=bool(field_def.default))
    if "int" in type_hint:
        return int(raw)
    if "float" in type_hint:
        return float(raw)
    if "path" in type_hint:
        return Path(str(raw)).expanduser().resolve()
    if "list" in type_hint:
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return [item.strip() for item in raw.split(",") if item.strip()]
        if isinstance(raw, (list, tuple)):
            return list(raw)
        return []
    if "dict" in type_hint:
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        if isinstance(raw, Mapping):
            return dict(raw)
        return {}
    return str(raw) if isinstance(raw, Path) else raw


def _apply_constraints(field_def, value: Any) -> Any:
    constraints = field_def.constraints or {}
    if value is None:
        return None
    if "enum" in constraints:
        allowed = set(constraints["enum"])
        if value not in allowed:
            raise ValueError(f"Value '{value}' not in enum for {field_def.name}")
    if isinstance(value, (int, float)):
        if "min" in constraints and value < constraints["min"]:
            raise ValueError(f"Value below min for {field_def.name}")
        if "max" in constraints and value > constraints["max"]:
            raise ValueError(f"Value above max for {field_def.name}")
    return value


def _normalize_material_types(
    raw: Optional[Iterable[Any]], *, fallback: Optional[List[Mapping[str, Any]]] = None
) -> List[Dict[str, Any]]:
    use_fallback = raw is None
    if isinstance(raw, Iterable) and not isinstance(raw, (str, bytes, dict)):
        source = list(raw)
    else:
        source = []
    if use_fallback and fallback:
        source = list(fallback)
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in source:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        entry: Dict[str, Any] = {"key": key}
        entry["label"] = str(item.get("label") or "")
        entry["description"] = str(item.get("description") or "")
        entry["enabled"] = _coerce_bool(item.get("enabled"), default=True)
        entry["extensions"] = _sanitize_list(item.get("extensions")) or []
        entry["text_keywords"] = _sanitize_list(item.get("text_keywords")) or []
        entry["filename_keywords"] = _sanitize_list(item.get("filename_keywords")) or []
        entry["exclude_keywords"] = _sanitize_list(item.get("exclude_keywords")) or []
        entry["aliases"] = _sanitize_list(item.get("aliases")) or []
        entry["llm_hint"] = str(item.get("llm_hint") or "")
        entry["priority"] = _safe_int(item.get("priority"), default=0)
        entry["threshold"] = _safe_float(item.get("threshold"), default=1.0)
        entry["extension_weight"] = _safe_float(item.get("extension_weight"), default=2.0)
        entry["filename_weight"] = _safe_float(item.get("filename_weight"), default=1.5)
        entry["text_weight"] = _safe_float(item.get("text_weight"), default=1.0)
        entry["require_extension"] = _coerce_bool(item.get("require_extension"), default=False)
        entry["require_filename"] = _coerce_bool(item.get("require_filename"), default=False)
        entry["require_text"] = _coerce_bool(item.get("require_text"), default=False)
        flow_raw = item.get("flow")
        if isinstance(flow_raw, (list, tuple)):
            entry["flow"] = [str(v or "").strip().lower() for v in flow_raw if str(v or "").strip()]
        special_raw = item.get("special")
        special: Dict[str, Any] = {}
        if isinstance(special_raw, Mapping):
            if "journal_toc_required" in special_raw:
                special["journal_toc_required"] = _coerce_bool(
                    special_raw.get("journal_toc_required"), default=True
                )
            if "min_toc_entries" in special_raw:
                special["min_toc_entries"] = max(1, _safe_int(special_raw.get("min_toc_entries"), default=5))
            if "min_pages" in special_raw:
                special["min_pages"] = max(1, _safe_int(special_raw.get("min_pages"), default=20))
            if "weight" in special_raw:
                special["weight"] = _safe_float(special_raw.get("weight"), default=2.0)
        if special:
            entry["special"] = special
        result.append(entry)
    return result


@dataclass
class RuntimeSettings:
    scan_root: Path
    extract_text: bool
    lmstudio_api_base: str
    lmstudio_model: str
    lmstudio_api_key: str
    lm_default_provider: str
    transcribe_enabled: bool
    transcribe_backend: str
    transcribe_model_path: str
    transcribe_language: str
    summarize_audio: bool
    audio_keywords_llm: bool
    images_vision_enabled: bool
    keywords_to_tags_enabled: bool
    type_detect_flow: str
    type_llm_override: bool
    import_subdir: str
    move_on_rename: bool
    collections_in_separate_dirs: bool
    collection_type_subdirs: bool
    type_dirs: Dict[str, str]
    default_use_llm: bool
    default_prune: bool
    ocr_langs_cfg: str
    pdf_ocr_pages_cfg: int
    always_ocr_first_page_dissertation: bool
    doc_chat_chunk_max_tokens: int = 700
    doc_chat_chunk_overlap: int = 120
    doc_chat_chunk_min_tokens: int = 80
    doc_chat_max_chunks: int = 0
    doc_chat_fallback_chunks: int = 0
    doc_chat_image_min_width: int = 32
    doc_chat_image_min_height: int = 32
    rag_embedding_backend: str = "auto"
    rag_embedding_model: str = "intfloat/multilingual-e5-large"
    rag_embedding_dim: int = 384
    rag_embedding_batch_size: int = 32
    rag_embedding_device: Optional[str] = None
    rag_embedding_endpoint: str = ""
    rag_embedding_api_key: str = ""
    ai_query_variants_max: int = 0
    ai_rag_retry_enabled: bool = True
    ai_rag_retry_threshold: float = 0.6
    feedback_train_interval_hours: float = 0.0
    feedback_train_cutoff_days: int = 90
    rag_rerank_backend: str = "none"
    rag_rerank_model: str = ""
    rag_rerank_device: Optional[str] = None
    rag_rerank_batch_size: int = 16
    rag_rerank_max_length: int = 512
    rag_rerank_max_chars: int = 1200
    prompts: Dict[str, str] = field(default_factory=dict)
    ai_rerank_llm: bool = False
    llm_pool_global_concurrency: int = 4
    llm_pool_per_user_concurrency: int = 1
    llm_queue_max_size: int = 100
    llm_queue_per_user_max: int = 10
    llm_request_timeout_sec: int = 90
    llm_retry_count: int = 1
    llm_retry_backoff_ms: int = 500
    lmstudio_idle_unload_minutes: int = 0
    llm_cache_enabled: bool = True
    llm_cache_ttl_seconds: int = 600
    llm_cache_max_items: int = 256
    llm_cache_only_mode: bool = False
    search_cache_enabled: bool = True
    search_cache_ttl_seconds: int = 90
    search_cache_max_items: int = 64
    lm_max_input_chars: int = 4000
    lm_max_output_tokens: int = 256
    azure_openai_api_version: str = "2024-02-15-preview"
    search_facet_tag_keys: Optional[List[str]] = None
    graph_facet_tag_keys: Optional[List[str]] = None
    search_facet_include_types: bool = True
    material_types: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: "AppConfig") -> "RuntimeSettings":
        return cls(
            scan_root=config.scan_root,
            extract_text=config.extract_text,
            lmstudio_api_base=config.lmstudio_api_base,
            lmstudio_model=config.lmstudio_model,
            lmstudio_api_key=config.lmstudio_api_key,
            lm_default_provider=config.lm_default_provider,
            rag_embedding_backend=config.rag_embedding_backend,
            rag_embedding_model=config.rag_embedding_model,
            rag_embedding_dim=config.rag_embedding_dim,
            rag_embedding_batch_size=config.rag_embedding_batch_size,
            rag_embedding_device=config.rag_embedding_device,
            rag_embedding_endpoint=config.rag_embedding_endpoint,
            rag_embedding_api_key=config.rag_embedding_api_key,
            rag_rerank_backend=config.rag_rerank_backend,
            rag_rerank_model=config.rag_rerank_model,
            rag_rerank_device=config.rag_rerank_device,
            rag_rerank_batch_size=config.rag_rerank_batch_size,
            rag_rerank_max_length=config.rag_rerank_max_length,
            rag_rerank_max_chars=config.rag_rerank_max_chars,
            ai_query_variants_max=config.ai_query_variants_max,
            ai_rag_retry_enabled=config.ai_rag_retry_enabled,
            ai_rag_retry_threshold=config.ai_rag_retry_threshold,
            feedback_train_interval_hours=float(os.getenv("AI_FEEDBACK_TRAIN_INTERVAL_HOURS", "0") or 0.0),
            feedback_train_cutoff_days=int(os.getenv("AI_FEEDBACK_TRAIN_CUTOFF_DAYS", "90") or 90),
            transcribe_enabled=config.transcribe_enabled,
            transcribe_backend=config.transcribe_backend,
            transcribe_model_path=config.transcribe_model_path,
            transcribe_language=config.transcribe_language,
            summarize_audio=config.summarize_audio,
            audio_keywords_llm=config.audio_keywords_llm,
            images_vision_enabled=config.images_vision_enabled,
            keywords_to_tags_enabled=config.keywords_to_tags_enabled,
            doc_chat_chunk_max_tokens=config.doc_chat_chunk_max_tokens,
            doc_chat_chunk_overlap=config.doc_chat_chunk_overlap,
            doc_chat_chunk_min_tokens=config.doc_chat_chunk_min_tokens,
            doc_chat_max_chunks=config.doc_chat_max_chunks,
            doc_chat_fallback_chunks=config.doc_chat_fallback_chunks,
            doc_chat_image_min_width=config.doc_chat_image_min_width,
            doc_chat_image_min_height=config.doc_chat_image_min_height,
            type_detect_flow=config.type_detect_flow,
            type_llm_override=config.type_llm_override,
            import_subdir=_sanitize_subdir(config.import_subdir),
            move_on_rename=config.move_on_rename,
            collections_in_separate_dirs=config.collections_in_separate_dirs,
            collection_type_subdirs=config.collection_type_subdirs,
            type_dirs=dict(config.type_dirs),
            default_use_llm=config.default_use_llm,
            default_prune=config.default_prune,
            ocr_langs_cfg=config.ocr_langs,
            pdf_ocr_pages_cfg=int(config.pdf_ocr_pages),
            always_ocr_first_page_dissertation=config.always_ocr_first_page_dissertation,
            prompts=dict(config.prompts),
            ai_rerank_llm=config.ai_rerank_llm,
            llm_pool_global_concurrency=config.llm_pool_global_concurrency,
            llm_pool_per_user_concurrency=config.llm_pool_per_user_concurrency,
            llm_queue_max_size=config.llm_queue_max_size,
            llm_queue_per_user_max=config.llm_queue_per_user_max,
            llm_request_timeout_sec=config.llm_request_timeout_sec,
            llm_retry_count=config.llm_retry_count,
            llm_retry_backoff_ms=config.llm_retry_backoff_ms,
            lmstudio_idle_unload_minutes=getattr(config, "lmstudio_idle_unload_minutes", 0),
            llm_cache_enabled=config.llm_cache_enabled,
            llm_cache_ttl_seconds=config.llm_cache_ttl_seconds,
            llm_cache_max_items=config.llm_cache_max_items,
            llm_cache_only_mode=config.llm_cache_read_only,
            search_cache_enabled=config.search_cache_enabled,
            search_cache_ttl_seconds=config.search_cache_ttl_seconds,
            search_cache_max_items=config.search_cache_max_items,
            lm_max_input_chars=config.lm_max_input_chars,
            lm_max_output_tokens=config.lm_max_output_tokens,
            azure_openai_api_version=config.azure_openai_api_version,
            search_facet_tag_keys=_sanitize_list(config.search_facet_tag_keys),
            graph_facet_tag_keys=_sanitize_list(config.graph_facet_tag_keys),
            search_facet_include_types=bool(config.search_facet_include_types),
            material_types=_normalize_material_types(config.material_type_profiles),
        )

    # -- сериализация -----------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "SCAN_ROOT": str(self.scan_root),
            "EXTRACT_TEXT": bool(self.extract_text),
            "LMSTUDIO_API_BASE": self.lmstudio_api_base,
            "LMSTUDIO_MODEL": self.lmstudio_model,
            "LMSTUDIO_API_KEY": self.lmstudio_api_key,
            "LM_DEFAULT_PROVIDER": self.lm_default_provider,
            "RAG_EMBEDDING_BACKEND": self.rag_embedding_backend,
            "RAG_EMBEDDING_MODEL": self.rag_embedding_model,
            "RAG_EMBEDDING_DIM": int(self.rag_embedding_dim),
            "RAG_EMBEDDING_BATCH": int(self.rag_embedding_batch_size),
            "RAG_EMBEDDING_DEVICE": self.rag_embedding_device,
            "RAG_EMBEDDING_ENDPOINT": self.rag_embedding_endpoint,
            "RAG_EMBEDDING_API_KEY": self.rag_embedding_api_key,
            "RAG_RERANK_BACKEND": self.rag_rerank_backend,
            "RAG_RERANK_MODEL": self.rag_rerank_model,
            "RAG_RERANK_DEVICE": self.rag_rerank_device,
            "RAG_RERANK_BATCH_SIZE": int(self.rag_rerank_batch_size),
            "RAG_RERANK_MAX_LENGTH": int(self.rag_rerank_max_length),
            "RAG_RERANK_MAX_CHARS": int(self.rag_rerank_max_chars),
            "AI_QUERY_VARIANTS_MAX": int(self.ai_query_variants_max),
            "TRANSCRIBE_ENABLED": bool(self.transcribe_enabled),
            "TRANSCRIBE_BACKEND": self.transcribe_backend,
            "TRANSCRIBE_MODEL_PATH": self.transcribe_model_path,
            "TRANSCRIBE_LANGUAGE": self.transcribe_language,
            "SUMMARIZE_AUDIO": bool(self.summarize_audio),
            "AUDIO_KEYWORDS_LLM": bool(self.audio_keywords_llm),
            "IMAGES_VISION_ENABLED": bool(self.images_vision_enabled),
            "KEYWORDS_TO_TAGS_ENABLED": bool(self.keywords_to_tags_enabled),
            "DOC_CHAT_CHUNK_MAX_TOKENS": int(self.doc_chat_chunk_max_tokens),
            "DOC_CHAT_CHUNK_OVERLAP": int(self.doc_chat_chunk_overlap),
            "DOC_CHAT_CHUNK_MIN_TOKENS": int(self.doc_chat_chunk_min_tokens),
            "DOC_CHAT_MAX_CHUNKS": int(self.doc_chat_max_chunks),
            "DOC_CHAT_FALLBACK_CHUNKS": int(self.doc_chat_fallback_chunks),
            "DOC_CHAT_IMAGE_MIN_WIDTH": int(self.doc_chat_image_min_width),
            "DOC_CHAT_IMAGE_MIN_HEIGHT": int(self.doc_chat_image_min_height),
            "TYPE_DETECT_FLOW": self.type_detect_flow,
            "TYPE_LLM_OVERRIDE": bool(self.type_llm_override),
            "IMPORT_SUBDIR": self.import_subdir,
            "MOVE_ON_RENAME": bool(self.move_on_rename),
            "COLLECTIONS_IN_SEPARATE_DIRS": bool(self.collections_in_separate_dirs),
            "COLLECTION_TYPE_SUBDIRS": bool(self.collection_type_subdirs),
            "TYPE_DIRS": dict(self.type_dirs),
            "DEFAULT_USE_LLM": bool(self.default_use_llm),
            "DEFAULT_PRUNE": bool(self.default_prune),
            "OCR_LANGS_CFG": self.ocr_langs_cfg,
            "PDF_OCR_PAGES_CFG": int(self.pdf_ocr_pages_cfg),
            "ALWAYS_OCR_FIRST_PAGE_DISSERTATION": bool(self.always_ocr_first_page_dissertation),
            "PROMPTS": dict(self.prompts),
            "AI_RERANK_LLM": bool(self.ai_rerank_llm),
            "LLM_POOL_GLOBAL_CONCURRENCY": int(self.llm_pool_global_concurrency),
            "LLM_POOL_PER_USER_CONCURRENCY": int(self.llm_pool_per_user_concurrency),
            "LLM_QUEUE_MAX_SIZE": int(self.llm_queue_max_size),
            "LLM_QUEUE_PER_USER_MAX": int(self.llm_queue_per_user_max),
            "LLM_REQUEST_TIMEOUT_SEC": int(self.llm_request_timeout_sec),
            "LLM_RETRY_COUNT": int(self.llm_retry_count),
            "LLM_RETRY_BACKOFF_MS": int(self.llm_retry_backoff_ms),
            "LMSTUDIO_IDLE_UNLOAD_MINUTES": int(self.lmstudio_idle_unload_minutes),
            "AI_RAG_RETRY_ENABLED": bool(self.ai_rag_retry_enabled),
            "AI_RAG_RETRY_THRESHOLD": float(self.ai_rag_retry_threshold),
            "AI_FEEDBACK_TRAIN_INTERVAL_HOURS": float(self.feedback_train_interval_hours),
            "AI_FEEDBACK_TRAIN_CUTOFF_DAYS": int(self.feedback_train_cutoff_days),
            "LLM_CACHE_ENABLED": bool(self.llm_cache_enabled),
            "LLM_CACHE_TTL_SECONDS": int(self.llm_cache_ttl_seconds),
            "LLM_CACHE_MAX_ITEMS": int(self.llm_cache_max_items),
            "LLM_CACHE_ONLY_MODE": bool(self.llm_cache_only_mode),
            "SEARCH_CACHE_ENABLED": bool(self.search_cache_enabled),
            "SEARCH_CACHE_TTL_SECONDS": int(self.search_cache_ttl_seconds),
            "SEARCH_CACHE_MAX_ITEMS": int(self.search_cache_max_items),
            "LM_MAX_INPUT_CHARS": int(self.lm_max_input_chars),
            "LM_MAX_OUTPUT_TOKENS": int(self.lm_max_output_tokens),
            "AZURE_OPENAI_API_VERSION": self.azure_openai_api_version,
            "SEARCH_FACET_TAG_KEYS": (
                list(self.search_facet_tag_keys) if self.search_facet_tag_keys is not None else None
            ),
            "GRAPH_FACET_TAG_KEYS": (
                list(self.graph_facet_tag_keys) if self.graph_facet_tag_keys is not None else None
            ),
            "SEARCH_FACET_INCLUDE_TYPES": bool(self.search_facet_include_types),
            "MATERIAL_TYPES": copy.deepcopy(self.material_types),
        }

    # -- применение -------------------------------------------------------
    def apply_env(self) -> None:
        os.environ["OCR_LANGS"] = self.ocr_langs_cfg
        os.environ["PDF_OCR_PAGES"] = str(self.pdf_ocr_pages_cfg)
        os.environ["OCR_DISS_FIRST_PAGE"] = "1" if self.always_ocr_first_page_dissertation else "0"
        os.environ["LM_MAX_INPUT_CHARS"] = str(self.lm_max_input_chars)
        os.environ["LM_MAX_OUTPUT_TOKENS"] = str(self.lm_max_output_tokens)
        os.environ["AZURE_OPENAI_API_VERSION"] = self.azure_openai_api_version
        os.environ["AI_FEEDBACK_TRAIN_INTERVAL_HOURS"] = str(self.feedback_train_interval_hours)
        os.environ["AI_FEEDBACK_TRAIN_CUTOFF_DAYS"] = str(self.feedback_train_cutoff_days)

    def apply_to_flask_config(self, app: "Flask") -> None:
        app.config["UPLOAD_FOLDER"] = str(self.scan_root)
        app.config["IMPORT_SUBDIR"] = self.import_subdir
        app.config["MOVE_ON_RENAME"] = self.move_on_rename
        app.config["COLLECTIONS_IN_SEPARATE_DIRS"] = self.collections_in_separate_dirs
        app.config["COLLECTION_TYPE_SUBDIRS"] = self.collection_type_subdirs
        app.config["TYPE_DIRS"] = dict(self.type_dirs)
        app.config["SEARCH_FACET_TAG_KEYS"] = (
            list(self.search_facet_tag_keys) if self.search_facet_tag_keys is not None else None
        )
        app.config["GRAPH_FACET_TAG_KEYS"] = (
            list(self.graph_facet_tag_keys) if self.graph_facet_tag_keys is not None else None
        )
        app.config["SEARCH_FACET_INCLUDE_TYPES"] = self.search_facet_include_types
        app.config["LLM_CACHE_ENABLED"] = self.llm_cache_enabled
        app.config["LLM_CACHE_TTL_SECONDS"] = self.llm_cache_ttl_seconds
        app.config["LLM_CACHE_MAX_ITEMS"] = self.llm_cache_max_items
        app.config["LLM_CACHE_ONLY_MODE"] = self.llm_cache_only_mode
        app.config["SEARCH_CACHE_ENABLED"] = self.search_cache_enabled
        app.config["SEARCH_CACHE_TTL_SECONDS"] = self.search_cache_ttl_seconds
        app.config["SEARCH_CACHE_MAX_ITEMS"] = self.search_cache_max_items
        app.config["LM_MAX_INPUT_CHARS"] = self.lm_max_input_chars
        app.config["LM_MAX_OUTPUT_TOKENS"] = self.lm_max_output_tokens
        app.config["AZURE_OPENAI_API_VERSION"] = self.azure_openai_api_version
        app.config["MATERIAL_TYPES"] = copy.deepcopy(self.material_types)
        app.config["AI_FEEDBACK_TRAIN_INTERVAL_HOURS"] = self.feedback_train_interval_hours
        app.config["AI_FEEDBACK_TRAIN_CUTOFF_DAYS"] = self.feedback_train_cutoff_days
        app.config["LLM_POOL_GLOBAL_CONCURRENCY"] = self.llm_pool_global_concurrency
        app.config["LLM_POOL_PER_USER_CONCURRENCY"] = self.llm_pool_per_user_concurrency
        app.config["LLM_QUEUE_MAX_SIZE"] = self.llm_queue_max_size
        app.config["LLM_QUEUE_PER_USER_MAX"] = self.llm_queue_per_user_max
        app.config["LLM_REQUEST_TIMEOUT_SEC"] = self.llm_request_timeout_sec
        app.config["LLM_RETRY_COUNT"] = self.llm_retry_count
        app.config["LLM_RETRY_BACKOFF_MS"] = self.llm_retry_backoff_ms
        app.config["LMSTUDIO_IDLE_UNLOAD_MINUTES"] = self.lmstudio_idle_unload_minutes

    # -- обновление -------------------------------------------------------
    def update_from_mapping(self, payload: Mapping[str, Any]) -> None:
        for key, raw in payload.items():
            if str(key).upper() == "PROMPTS" and isinstance(raw, Mapping):
                for p_key, p_val in raw.items():
                    if isinstance(p_val, str):
                        self.prompts[str(p_key)] = p_val
                continue

            field_def = _resolve_field_by_input_key(str(key))
            if not field_def or not field_def.runtime_mutable:
                continue
            attr_name = _FIELD_TO_ATTR.get(field_def.name, field_def.name)
            if not hasattr(self, attr_name):
                continue
            try:
                value = _coerce_schema_value(field_def, raw)
                value = _apply_constraints(field_def, value)
            except Exception:
                continue

            if field_def.name == "import_subdir" and value is not None:
                value = _sanitize_subdir(str(value))
            elif field_def.name in {"rag_embedding_device", "rag_rerank_device"}:
                value = str(value).strip() or None
            elif field_def.name in {"search_facet_tag_keys", "graph_facet_tag_keys"}:
                value = _sanitize_list(value) if value is not None else None
            elif field_def.name == "type_dirs" and isinstance(value, Mapping):
                value = {str(k): str(v) for k, v in value.items()}
            elif field_def.name == "material_types":
                fallback = self.material_types or None
                value = _normalize_material_types(value, fallback=fallback)
            elif field_def.name == "lm_default_provider" and value:
                value = str(value).strip().lower()
            setattr(self, attr_name, value)

        if not self.collections_in_separate_dirs:
            self.collection_type_subdirs = False
        self.apply_env()


class RuntimeSettingsStore:
    """Контейнер для runtime-настроек с ленивой инициализацией."""

    def __init__(self) -> None:
        self._settings: Optional[RuntimeSettings] = None

    @property
    def current(self) -> RuntimeSettings:
        if self._settings is None:
            raise RuntimeError("Runtime settings are not initialized")
        return self._settings

    def initialize(self, config: "AppConfig") -> RuntimeSettings:
        self._settings = RuntimeSettings.from_config(config)
        self._settings.apply_env()
        return self._settings

    def snapshot(self) -> Dict[str, Any]:
        return self.current.snapshot()

    def apply_updates(self, payload: Mapping[str, Any]) -> RuntimeSettings:
        settings = self.current
        settings.update_from_mapping(payload)
        return settings


runtime_settings_store = RuntimeSettingsStore()


__all__ = ["RuntimeSettings", "RuntimeSettingsStore", "runtime_settings_store"]

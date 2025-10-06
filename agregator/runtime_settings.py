"""Вспомогательный модуль для управления runtime-настройками Agregator.

Инкапсулирует параметры, которые могут меняться во время работы приложения и
сохраняться на диск. Предоставляет единый интерфейс для обновления, загрузки и
применения настроек к Flask-приложению и окружению.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
    prompts: Dict[str, str] = field(default_factory=dict)
    ai_rerank_llm: bool = False
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

    @classmethod
    def from_config(cls, config: "AppConfig") -> "RuntimeSettings":
        return cls(
            scan_root=config.scan_root,
            extract_text=config.extract_text,
            lmstudio_api_base=config.lmstudio_api_base,
            lmstudio_model=config.lmstudio_model,
            lmstudio_api_key=config.lmstudio_api_key,
            lm_default_provider=config.lm_default_provider,
            transcribe_enabled=config.transcribe_enabled,
            transcribe_backend=config.transcribe_backend,
            transcribe_model_path=config.transcribe_model_path,
            transcribe_language=config.transcribe_language,
            summarize_audio=config.summarize_audio,
            audio_keywords_llm=config.audio_keywords_llm,
            images_vision_enabled=config.images_vision_enabled,
            keywords_to_tags_enabled=config.keywords_to_tags_enabled,
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
            "TRANSCRIBE_ENABLED": bool(self.transcribe_enabled),
            "TRANSCRIBE_BACKEND": self.transcribe_backend,
            "TRANSCRIBE_MODEL_PATH": self.transcribe_model_path,
            "TRANSCRIBE_LANGUAGE": self.transcribe_language,
            "SUMMARIZE_AUDIO": bool(self.summarize_audio),
            "AUDIO_KEYWORDS_LLM": bool(self.audio_keywords_llm),
            "IMAGES_VISION_ENABLED": bool(self.images_vision_enabled),
            "KEYWORDS_TO_TAGS_ENABLED": bool(self.keywords_to_tags_enabled),
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
        }

    # -- применение -------------------------------------------------------
    def apply_env(self) -> None:
        os.environ["OCR_LANGS"] = self.ocr_langs_cfg
        os.environ["PDF_OCR_PAGES"] = str(self.pdf_ocr_pages_cfg)
        os.environ["OCR_DISS_FIRST_PAGE"] = "1" if self.always_ocr_first_page_dissertation else "0"
        os.environ["LM_MAX_INPUT_CHARS"] = str(self.lm_max_input_chars)
        os.environ["LM_MAX_OUTPUT_TOKENS"] = str(self.lm_max_output_tokens)
        os.environ["AZURE_OPENAI_API_VERSION"] = self.azure_openai_api_version

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

    # -- обновление -------------------------------------------------------
    def update_from_mapping(self, payload: Mapping[str, Any]) -> None:
        for key, raw in payload.items():
            if key == "SCAN_ROOT" and raw:
                self.scan_root = Path(str(raw)).expanduser().resolve()
            elif key == "EXTRACT_TEXT":
                self.extract_text = _coerce_bool(raw, default=self.extract_text)
            elif key == "LMSTUDIO_API_BASE" and raw is not None:
                self.lmstudio_api_base = str(raw)
            elif key == "LMSTUDIO_MODEL" and raw is not None:
                self.lmstudio_model = str(raw)
            elif key == "LMSTUDIO_API_KEY":
                self.lmstudio_api_key = str(raw or "")
            elif key == "LM_DEFAULT_PROVIDER" and raw:
                self.lm_default_provider = str(raw).strip().lower() or self.lm_default_provider
            elif key == "TRANSCRIBE_ENABLED":
                self.transcribe_enabled = _coerce_bool(raw, default=self.transcribe_enabled)
            elif key == "TRANSCRIBE_BACKEND" and raw is not None:
                self.transcribe_backend = str(raw)
            elif key == "TRANSCRIBE_MODEL_PATH" and raw is not None:
                self.transcribe_model_path = str(raw)
            elif key == "TRANSCRIBE_LANGUAGE" and raw is not None:
                self.transcribe_language = str(raw)
            elif key == "SUMMARIZE_AUDIO":
                self.summarize_audio = _coerce_bool(raw, default=self.summarize_audio)
            elif key == "AUDIO_KEYWORDS_LLM":
                self.audio_keywords_llm = _coerce_bool(raw, default=self.audio_keywords_llm)
            elif key == "IMAGES_VISION_ENABLED":
                self.images_vision_enabled = _coerce_bool(raw, default=self.images_vision_enabled)
            elif key == "KEYWORDS_TO_TAGS_ENABLED":
                self.keywords_to_tags_enabled = _coerce_bool(raw, default=self.keywords_to_tags_enabled)
            elif key == "TYPE_DETECT_FLOW" and raw is not None:
                self.type_detect_flow = str(raw)
            elif key == "TYPE_LLM_OVERRIDE":
                self.type_llm_override = _coerce_bool(raw, default=self.type_llm_override)
            elif key == "IMPORT_SUBDIR" and raw is not None:
                self.import_subdir = _sanitize_subdir(str(raw))
            elif key == "MOVE_ON_RENAME":
                self.move_on_rename = _coerce_bool(raw, default=self.move_on_rename)
            elif key == "COLLECTIONS_IN_SEPARATE_DIRS":
                self.collections_in_separate_dirs = _coerce_bool(raw, default=self.collections_in_separate_dirs)
            elif key == "COLLECTION_TYPE_SUBDIRS":
                self.collection_type_subdirs = _coerce_bool(raw, default=self.collection_type_subdirs)
            elif key == "TYPE_DIRS" and isinstance(raw, Mapping):
                self.type_dirs = {str(k): str(v) for k, v in raw.items()}
            elif key == "DEFAULT_USE_LLM":
                self.default_use_llm = _coerce_bool(raw, default=self.default_use_llm)
            elif key == "DEFAULT_PRUNE":
                self.default_prune = _coerce_bool(raw, default=self.default_prune)
            elif key == "OCR_LANGS_CFG" and raw is not None:
                self.ocr_langs_cfg = str(raw)
            elif key == "PDF_OCR_PAGES_CFG":
                try:
                    self.pdf_ocr_pages_cfg = max(1, int(raw))
                except Exception:
                    pass
            elif key == "ALWAYS_OCR_FIRST_PAGE_DISSERTATION":
                self.always_ocr_first_page_dissertation = _coerce_bool(
                    raw, default=self.always_ocr_first_page_dissertation
                )
            elif key == "PROMPTS" and isinstance(raw, Mapping):
                for p_key, p_val in raw.items():
                    if isinstance(p_val, str):
                        self.prompts[str(p_key)] = p_val
            elif key == "AI_RERANK_LLM":
                self.ai_rerank_llm = _coerce_bool(raw, default=self.ai_rerank_llm)
            elif key == "LLM_CACHE_ENABLED":
                self.llm_cache_enabled = _coerce_bool(raw, default=self.llm_cache_enabled)
            elif key == "LLM_CACHE_TTL_SECONDS":
                try:
                    self.llm_cache_ttl_seconds = max(1, int(raw))
                except Exception:
                    pass
            elif key == "LLM_CACHE_MAX_ITEMS":
                try:
                    self.llm_cache_max_items = max(1, int(raw))
                except Exception:
                    pass
            elif key == "LLM_CACHE_ONLY_MODE":
                self.llm_cache_only_mode = _coerce_bool(raw, default=self.llm_cache_only_mode)
            elif key == "SEARCH_CACHE_ENABLED":
                self.search_cache_enabled = _coerce_bool(raw, default=self.search_cache_enabled)
            elif key == "SEARCH_CACHE_TTL_SECONDS":
                try:
                    self.search_cache_ttl_seconds = max(1, int(raw))
                except Exception:
                    pass
            elif key == "SEARCH_CACHE_MAX_ITEMS":
                try:
                    self.search_cache_max_items = max(1, int(raw))
                except Exception:
                    pass
            elif key == "LM_MAX_INPUT_CHARS":
                try:
                    self.lm_max_input_chars = max(500, int(raw))
                except Exception:
                    pass
            elif key == "LM_MAX_OUTPUT_TOKENS":
                try:
                    self.lm_max_output_tokens = max(16, int(raw))
                except Exception:
                    pass
            elif key == "AZURE_OPENAI_API_VERSION" and raw is not None:
                token = str(raw).strip()
                if token:
                    self.azure_openai_api_version = token
            elif key == "SEARCH_FACET_TAG_KEYS":
                self.search_facet_tag_keys = _sanitize_list(raw) if raw is not None else None
            elif key == "GRAPH_FACET_TAG_KEYS":
                self.graph_facet_tag_keys = _sanitize_list(raw) if raw is not None else None
            elif key == "SEARCH_FACET_INCLUDE_TYPES":
                self.search_facet_include_types = _coerce_bool(
                    raw, default=self.search_facet_include_types
                )

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

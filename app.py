# REST API и маршруты импорта/экспорта перенесены в `routes.py` как Blueprint.
import os
import re
import json
import hashlib
import hmac
import logging
from logging.handlers import RotatingFileHandler
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
import itertools
import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Mapping

import click
from flask import Blueprint, Flask, request, redirect, url_for, jsonify, send_from_directory, send_file, Response, make_response, session, g, abort, current_app, has_app_context
from functools import wraps
from werkzeug.utils import secure_filename
import sqlite3
import threading

try:
    from flask import copy_current_app_context
except ImportError:  # запасной вариант для старых версий Flask
    from flask import current_app

    def copy_current_app_context(func):
        app_obj = current_app._get_current_object()

        @wraps(func)
        def wrapper(*args, **kwargs):
            with app_obj.app_context():
                return func(*args, **kwargs)

        return wrapper
from sqlalchemy import func, and_, or_, exists, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import aliased
from models import (
    db,
    File,
    Tag,
    TagSchema,
    ChangeLog,
    upsert_tag,
    file_to_dict,
    Collection,
    User,
    CollectionMember,
    UserActionLog,
    TaskRecord,
    LlmEndpoint,
    AiWordAccess,
    AiSearchSnippetCache,
    AiSearchKeywordFeedback,
    AiSearchMetric,
    RagDocument,
    RagDocumentVersion,
    RagDocumentChunk,
    RagChunkEmbedding,
    RagIngestFailure,
    RagSession,
)


@event.listens_for(Engine, "connect")
def _configure_sqlite_pragmas(dbapi_connection, connection_record):
    """Ensure required PRAGMA flags are enabled for SQLite connections."""
    try:
        import sqlite3 as _sqlite3
    except Exception:
        _sqlite3 = None
    if _sqlite3 is not None and isinstance(dbapi_connection, _sqlite3.Connection):
        try:
            import math as _math
            try:
                from agregator.rag.utils import bytes_to_vector as _bytes_to_vector  # type: ignore
            except Exception:
                _bytes_to_vector = None

            def _cosine_similarity_sqlite(blob_a, blob_b):
                if _bytes_to_vector is None or blob_a is None or blob_b is None:
                    return 0.0
                try:
                    vec_a = _bytes_to_vector(blob_a)
                    vec_b = _bytes_to_vector(blob_b)
                except Exception:
                    return 0.0
                if not vec_a or not vec_b or len(vec_a) != len(vec_b):
                    return 0.0
                dot = sum(a * b for a, b in zip(vec_a, vec_b))
                if dot == 0.0:
                    return 0.0
                norm_a = _math.sqrt(sum(a * a for a in vec_a))
                norm_b = _math.sqrt(sum(b * b for b in vec_b))
                if norm_a == 0.0 or norm_b == 0.0:
                    return 0.0
                return dot / (norm_a * norm_b)

            dbapi_connection.create_function("cosine_similarity", 2, _cosine_similarity_sqlite)
        except Exception:
            pass
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys = 1")
            cursor.execute("PRAGMA trusted_schema = 1")
            cursor.execute("PRAGMA recursive_triggers = 1")
        finally:
            cursor.close()


import fitz  # библиотека PyMuPDF
import requests
from agregator.config import AppConfig, load_app_config
from agregator.runtime_settings import runtime_settings_store
from agregator.services import (
    HttpSettings,
    configure_http,
    configure_llm_cache,
    configure_search_cache,
    configure_logging,
    get_rotating_log_handler as svc_get_rotating_log_handler,
    get_task_queue,
    llm_cache_get,
    llm_cache_set,
    CachedLLMResponse,
    FacetQueryParams,
    FacetService,
    SearchService,
    search_cache_get,
    search_cache_set,
    http_request,
    list_system_log_files as svc_list_system_log_files,
    resolve_log_name as svc_resolve_log_name,
    tail_log_file as svc_tail_log_file,
)
from agregator.rag import (
    ChunkConfig,
    ContextSelector,
    KeywordRetriever,
    ContextSection,
    ValidationResult,
    RagIndexer,
    VectorRetriever,
    EmbeddingBackend,
    load_embedding_backend,
    vector_to_bytes,
    build_system_prompt,
    build_user_prompt,
    fallback_answer,
    validate_answer,
    detect_language,
)
try:
    import docx
except ImportError:
    docx = None
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None
try:
    from ebooklib import epub
except ImportError:
    epub = None
try:
    import djvu.decode
except ImportError:
    djvu = None
try:
    import pytesseract  # необязательный OCR для PDF, состоящих из изображений
except ImportError:
    pytesseract = None
try:
    import mammoth
except ImportError:
    mammoth = None
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except Exception:
    FasterWhisperModel = None
try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except Exception:
    hf_snapshot_download = None
import subprocess, shutil, wave, tempfile
import time
from html import escape

# ------------------- RU morphology (Natasha/pymorphy2) -------------------
try:
    from razdel import tokenize as _ru_tokenize
except Exception:
    _ru_tokenize = None
try:
    import pymorphy2
    _morph = pymorphy2.MorphAnalyzer()
except Exception:
    _morph = None

logger = logging.getLogger(__name__)

PIPELINE_API_KEY = (os.getenv("PIPELINE_API_KEY") or "").strip()


def _extract_pipeline_token() -> Optional[str]:
    header = request.headers.get('Authorization')
    token: Optional[str] = None
    if header:
        lowered = header.lower()
        if lowered.startswith('bearer '):
            token = header[7:].strip()
        else:
            token = header.strip()
    if not token:
        token = request.headers.get('X-Agregator-Key')
        if token:
            token = token.strip()
    if not token:
        token = request.headers.get('X-Agregator-Token')
        if token:
            token = token.strip()
    return token or None


def _check_pipeline_access() -> tuple[bool, Optional[Response]]:
    user = _load_current_user()
    if user and getattr(user, 'role', '') == 'admin':
        return True, None
    if PIPELINE_API_KEY:
        token = _extract_pipeline_token()
        if token and hmac.compare_digest(token, PIPELINE_API_KEY):
            return True, None
        resp = jsonify({'ok': False, 'error': 'Forbidden'})
        resp.status_code = 403
        return False, resp
    return True, None


def _rt():
    """Быстрый доступ к текущим runtime-настройкам."""
    return runtime_settings_store.current


def _lm_max_input_chars() -> int:
    """Return current input character limit for LLM calls."""
    try:
        value = int(getattr(_rt(), 'lm_max_input_chars', 4000) or 4000)
    except Exception:
        value = 4000
    return max(1, value)


def _lm_max_output_tokens() -> int:
    """Return current output token limit for LLM calls."""
    try:
        value = int(getattr(_rt(), 'lm_max_output_tokens', 256) or 256)
    except Exception:
        value = 256
    return max(1, value)


def _lm_safe_context_tokens() -> int:
    """Return the soft limit of tokens we keep within the prompt window."""
    try:
        value = int(getattr(_rt(), 'lm_context_token_limit', 3200) or 3200)
    except Exception:
        value = 3200
    return max(512, value)


def _always_ocr_first_page_dissertation() -> bool:
    """Return flag for forcing OCR on the first page of dissertations."""
    try:
        return bool(getattr(_rt(), 'always_ocr_first_page_dissertation', False))
    except Exception:
        return False


def _refresh_runtime_globals() -> None:
    """Expose frequently used runtime flags as module globals for legacy code."""
    runtime = runtime_settings_store.current
    for key, attr in _RUNTIME_ATTR_MAP.items():
        try:
            value = getattr(runtime, attr)
        except Exception:
            value = None
        if key in {'LM_MAX_INPUT_CHARS', 'LM_MAX_OUTPUT_TOKENS'}:
            try:
                value = int(value)
            except Exception:
                value = 0
            if value <= 0:
                value = 4000 if key == 'LM_MAX_INPUT_CHARS' else 256
        if key in {
            'TRANSCRIBE_ENABLED',
            'SUMMARIZE_AUDIO',
            'AUDIO_KEYWORDS_LLM',
            'IMAGES_VISION_ENABLED',
            'KEYWORDS_TO_TAGS_ENABLED',
            'TYPE_LLM_OVERRIDE',
            'MOVE_ON_RENAME',
            'COLLECTIONS_IN_SEPARATE_DIRS',
            'COLLECTION_TYPE_SUBDIRS',
            'DEFAULT_USE_LLM',
            'DEFAULT_PRUNE',
            'AI_RERANK_LLM',
            'LLM_CACHE_ENABLED',
            'LLM_CACHE_ONLY_MODE',
            'SEARCH_CACHE_ENABLED',
            'ALWAYS_OCR_FIRST_PAGE_DISSERTATION',
            'SEARCH_FACET_INCLUDE_TYPES',
        }:
            value = bool(value)
        if key == 'PROMPTS':
            value = dict(value or {})
        globals()[key] = value


_RUNTIME_ATTR_MAP = {
    'SCAN_ROOT': 'scan_root',
    'EXTRACT_TEXT': 'extract_text',
    'LMSTUDIO_API_BASE': 'lmstudio_api_base',
    'LMSTUDIO_MODEL': 'lmstudio_model',
    'LMSTUDIO_API_KEY': 'lmstudio_api_key',
    'LM_DEFAULT_PROVIDER': 'lm_default_provider',
    'TRANSCRIBE_ENABLED': 'transcribe_enabled',
    'TRANSCRIBE_BACKEND': 'transcribe_backend',
    'TRANSCRIBE_MODEL_PATH': 'transcribe_model_path',
    'TRANSCRIBE_LANGUAGE': 'transcribe_language',
    'SUMMARIZE_AUDIO': 'summarize_audio',
    'AUDIO_KEYWORDS_LLM': 'audio_keywords_llm',
    'IMAGES_VISION_ENABLED': 'images_vision_enabled',
    'KEYWORDS_TO_TAGS_ENABLED': 'keywords_to_tags_enabled',
    'TYPE_DETECT_FLOW': 'type_detect_flow',
    'TYPE_LLM_OVERRIDE': 'type_llm_override',
    'IMPORT_SUBDIR': 'import_subdir',
    'MOVE_ON_RENAME': 'move_on_rename',
    'COLLECTIONS_IN_SEPARATE_DIRS': 'collections_in_separate_dirs',
    'COLLECTION_TYPE_SUBDIRS': 'collection_type_subdirs',
    'TYPE_DIRS': 'type_dirs',
    'DEFAULT_USE_LLM': 'default_use_llm',
    'DEFAULT_PRUNE': 'default_prune',
    'OCR_LANGS_CFG': 'ocr_langs_cfg',
    'PDF_OCR_PAGES_CFG': 'pdf_ocr_pages_cfg',
    'ALWAYS_OCR_FIRST_PAGE_DISSERTATION': 'always_ocr_first_page_dissertation',
    'PROMPTS': 'prompts',
    'AI_RERANK_LLM': 'ai_rerank_llm',
    'LLM_CACHE_ENABLED': 'llm_cache_enabled',
    'LLM_CACHE_TTL_SECONDS': 'llm_cache_ttl_seconds',
    'LLM_CACHE_MAX_ITEMS': 'llm_cache_max_items',
    'LLM_CACHE_ONLY_MODE': 'llm_cache_only_mode',
    'SEARCH_CACHE_ENABLED': 'search_cache_enabled',
    'SEARCH_CACHE_TTL_SECONDS': 'search_cache_ttl_seconds',
    'SEARCH_CACHE_MAX_ITEMS': 'search_cache_max_items',
    'LM_MAX_INPUT_CHARS': 'lm_max_input_chars',
    'LM_MAX_OUTPUT_TOKENS': 'lm_max_output_tokens',
    'AZURE_OPENAI_API_VERSION': 'azure_openai_api_version',
    'SEARCH_FACET_TAG_KEYS': 'search_facet_tag_keys',
    'GRAPH_FACET_TAG_KEYS': 'graph_facet_tag_keys',
    'SEARCH_FACET_INCLUDE_TYPES': 'search_facet_include_types',
}

LM_MAX_INPUT_CHARS = 4000
LM_MAX_OUTPUT_TOKENS = 256
ALWAYS_OCR_FIRST_PAGE_DISSERTATION = False
PROMPTS = {}

for _GLOBAL_NAME in _RUNTIME_ATTR_MAP:
    globals().setdefault(_GLOBAL_NAME, None)


def _scan_root_path() -> Path:
    return _rt().scan_root


def _import_subdir_value() -> str:
    return (_rt().import_subdir or '').strip()


def _collections_in_separate_dirs() -> bool:
    return bool(_rt().collections_in_separate_dirs)


def _ensure_rag_embedding_defaults() -> None:
    runtime = runtime_settings_store.current
    updates: dict[str, object] = {}
    backend_raw = (runtime.rag_embedding_backend or '').strip().lower()
    if backend_raw in ('', 'auto'):
        updates['RAG_EMBEDDING_BACKEND'] = 'lm-studio'
    model_raw = (runtime.rag_embedding_model or '').strip()
    if not model_raw or model_raw == 'intfloat/multilingual-e5-large':
        updates['RAG_EMBEDDING_MODEL'] = 'nomic-ai/nomic-embed-text-v1.5-GGUF'
        if (runtime.rag_embedding_dim or 0) < 256:
            updates['RAG_EMBEDDING_DIM'] = 768
    if not runtime.rag_embedding_endpoint:
        updates['RAG_EMBEDDING_ENDPOINT'] = runtime.lmstudio_api_base or ''
    if runtime.rag_embedding_api_key is None:
        updates['RAG_EMBEDDING_API_KEY'] = ''
    if updates:
        runtime_settings_store.apply_updates(updates)


_RUNTIME_ATTR_MAP = {
    'SCAN_ROOT': 'scan_root',
    'EXTRACT_TEXT': 'extract_text',
    'LMSTUDIO_API_BASE': 'lmstudio_api_base',
    'LMSTUDIO_MODEL': 'lmstudio_model',
    'LMSTUDIO_API_KEY': 'lmstudio_api_key',
    'LM_DEFAULT_PROVIDER': 'lm_default_provider',
    'TRANSCRIBE_ENABLED': 'transcribe_enabled',
    'TRANSCRIBE_BACKEND': 'transcribe_backend',
    'TRANSCRIBE_MODEL_PATH': 'transcribe_model_path',
    'TRANSCRIBE_LANGUAGE': 'transcribe_language',
    'SUMMARIZE_AUDIO': 'summarize_audio',
    'AUDIO_KEYWORDS_LLM': 'audio_keywords_llm',
    'IMAGES_VISION_ENABLED': 'images_vision_enabled',
    'KEYWORDS_TO_TAGS_ENABLED': 'keywords_to_tags_enabled',
    'TYPE_DETECT_FLOW': 'type_detect_flow',
    'TYPE_LLM_OVERRIDE': 'type_llm_override',
    'IMPORT_SUBDIR': 'import_subdir',
    'MOVE_ON_RENAME': 'move_on_rename',
    'COLLECTIONS_IN_SEPARATE_DIRS': 'collections_in_separate_dirs',
    'COLLECTION_TYPE_SUBDIRS': 'collection_type_subdirs',
    'TYPE_DIRS': 'type_dirs',
    'DEFAULT_USE_LLM': 'default_use_llm',
    'DEFAULT_PRUNE': 'default_prune',
    'OCR_LANGS_CFG': 'ocr_langs_cfg',
    'PDF_OCR_PAGES_CFG': 'pdf_ocr_pages_cfg',
    'ALWAYS_OCR_FIRST_PAGE_DISSERTATION': 'always_ocr_first_page_dissertation',
    'PROMPTS': 'prompts',
    'AI_RERANK_LLM': 'ai_rerank_llm',
    'LLM_CACHE_ENABLED': 'llm_cache_enabled',
    'LLM_CACHE_TTL_SECONDS': 'llm_cache_ttl_seconds',
    'LLM_CACHE_MAX_ITEMS': 'llm_cache_max_items',
    'LLM_CACHE_ONLY_MODE': 'llm_cache_only_mode',
    'SEARCH_CACHE_ENABLED': 'search_cache_enabled',
    'SEARCH_CACHE_TTL_SECONDS': 'search_cache_ttl_seconds',
    'SEARCH_CACHE_MAX_ITEMS': 'search_cache_max_items',
    'LM_MAX_INPUT_CHARS': 'lm_max_input_chars',
    'LM_MAX_OUTPUT_TOKENS': 'lm_max_output_tokens',
    'AZURE_OPENAI_API_VERSION': 'azure_openai_api_version',
    'SEARCH_FACET_TAG_KEYS': 'search_facet_tag_keys',
    'GRAPH_FACET_TAG_KEYS': 'graph_facet_tag_keys',
    'SEARCH_FACET_INCLUDE_TYPES': 'search_facet_include_types',
}


def __getattr__(name: str):
    attr = _RUNTIME_ATTR_MAP.get(name)
    if attr is not None:
        return getattr(runtime_settings_store.current, attr)
    raise AttributeError(f"module 'app' has no attribute {name!r}")


ROLE_ADMIN = 'admin'
ROLE_EDITOR = 'editor'
ROLE_VIEWER = 'viewer'
_ROLE_ALIASES = {
    'user': ROLE_EDITOR,
    'editor': ROLE_EDITOR,
    'viewer': ROLE_VIEWER,
    'admin': ROLE_ADMIN,
}


def _normalize_user_role(value: str | None, *, default: str = ROLE_VIEWER) -> str:
    if not value:
        return default
    token = str(value).strip().lower()
    return _ROLE_ALIASES.get(token, default if token not in _ROLE_ALIASES else token)


def _user_role(user: User | None) -> str:
    if not user:
        return ROLE_VIEWER
    return _normalize_user_role(getattr(user, 'role', None), default=ROLE_EDITOR)

PIPELINE_API_KEY = (os.getenv("PIPELINE_API_KEY") or "").strip()


def _extract_pipeline_token() -> Optional[str]:
    header = request.headers.get('Authorization')
    token: Optional[str] = None
    if header:
        lowered = header.lower()
        if lowered.startswith('bearer '):
            token = header[7:].strip()
        else:
            token = header.strip()
    if not token:
        token = request.headers.get('X-Agregator-Key')
        if token:
            token = token.strip()
    if not token:
        token = request.headers.get('X-Agregator-Token')
        if token:
            token = token.strip()
    return token or None


def _check_pipeline_access() -> tuple[bool, Optional[Response]]:
    user = _load_current_user()
    if _user_role(user) == ROLE_ADMIN:
        return True, None
    if PIPELINE_API_KEY:
        token = _extract_pipeline_token()
        if token and hmac.compare_digest(token, PIPELINE_API_KEY):
            return True, None
        resp = jsonify({'ok': False, 'error': 'Forbidden'})
        resp.status_code = 403
        return False, resp
    return True, None


class TimedCache:
    def __init__(self, max_items: int = 128, ttl: float = 60.0):
        self.max_items = max(1, int(max_items))
        self.ttl = float(ttl)
        self._store: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.time()
        dead = [key for key, (expires, _val) in self._store.items() if expires <= now]
        for key in dead:
            self._store.pop(key, None)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def get(self, key: tuple) -> object | None:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires, value = item
            if expires <= time.time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: tuple, value: object) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
            self._store.move_to_end(key)
            self._prune()

    def get_or_set(self, key: tuple, factory):
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


FACET_CACHE = TimedCache(max_items=256, ttl=60)
facet_service = FacetService(FACET_CACHE)
search_service = SearchService(
    db=db,
    file_model=File,
    tag_model=Tag,
    cache_get=search_cache_get,
    cache_set=search_cache_set,
    logger=logging.getLogger('agregator.search'),
)

_FACET_CACHE_REBUILD_LOCK = threading.Lock()
_FACET_CACHE_REBUILD_PENDING = False


def _invalidate_facets_cache(reason: str | None = None) -> None:
    try:
        facet_service.invalidate(reason)
    except Exception:
        pass
    try:
        search_service.invalidate_cache(reason or 'facets')
    except Exception:
        pass
    try:
        _schedule_facet_cache_rebuild(reason)
    except Exception:
        pass


def _facet_default_contexts() -> list[FacetQueryParams]:
    cfg = current_app.config

    def _normalize_keys(value):
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        return None

    search_keys = _normalize_keys(cfg.get('SEARCH_FACET_TAG_KEYS'))
    graph_keys = _normalize_keys(cfg.get('GRAPH_FACET_TAG_KEYS'))
    include_types = bool(cfg.get('SEARCH_FACET_INCLUDE_TYPES', True))
    return [
        FacetQueryParams(
            query='',
            material_type='',
            context='search',
            include_types=include_types,
            tag_filters=[],
            collection_filter=None,
            allowed_scope=None,
            allowed_keys_list=search_keys,
            year_from='',
            year_to='',
            size_min='',
            size_max='',
            sources={'tags': True},
            request_args=(),
        ),
        FacetQueryParams(
            query='',
            material_type='',
            context='graph',
            include_types=False,
            tag_filters=[],
            collection_filter=None,
            allowed_scope=None,
            allowed_keys_list=graph_keys,
            year_from='',
            year_to='',
            size_min='',
            size_max='',
            sources={'tags': True},
            request_args=(('context', ('graph',)),),
        ),
    ]


def _schedule_facet_cache_rebuild(reason: str | None = None) -> None:
    if not has_app_context():
        return
    global _FACET_CACHE_REBUILD_PENDING
    with _FACET_CACHE_REBUILD_LOCK:
        if _FACET_CACHE_REBUILD_PENDING:
            return
        _FACET_CACHE_REBUILD_PENDING = True

    @copy_current_app_context
    def _run_rebuild():
        global _FACET_CACHE_REBUILD_PENDING
        try:
            contexts = _facet_default_contexts()
            for params in contexts:
                facet_service.get_facets(
                    params,
                    search_candidate_fn=_search_candidate_ids,
                    like_filter_fn=_apply_like_filter,
                )
            app.logger.debug('Facet cache prewarmed%s', f" ({reason})" if reason else '')
        except Exception as exc:
            app.logger.warning('Facet cache prewarm failed: %s', exc)
        finally:
            with _FACET_CACHE_REBUILD_LOCK:
                _FACET_CACHE_REBUILD_PENDING = False

    try:
        get_task_queue().submit(_run_rebuild, description='facet_cache_prewarm')
    except Exception as exc:
        app.logger.debug('Facet cache prewarm scheduling failed: %s', exc)
        with _FACET_CACHE_REBUILD_LOCK:
            _FACET_CACHE_REBUILD_PENDING = False


def _current_allowed_collections() -> set[int] | None:
    allowed = getattr(g, 'allowed_collection_ids', None)
    if allowed is None:
        return None
    try:
        return set(int(x) for x in allowed)
    except Exception:
        return None


def _ensure_search_support() -> None:
    search_service.ensure_support()


def _rebuild_files_fts(conn=None):
    search_service.rebuild_files(connection=conn)


def _rebuild_tags_fts(conn=None):
    search_service.rebuild_tags(connection=conn)


def _sync_file_to_fts(file_obj: File | None):
    search_service.sync_file(file_obj)


def _delete_file_from_fts(file_id: int | None):
    search_service.delete_file(file_id)


def _search_candidate_ids(query: str, limit: int = 4000) -> list[int] | None:
    return search_service.candidate_ids(query, limit=limit)


def _apply_like_filter(base_query, query: str):
    return search_service.apply_like_filter(base_query, query)


def _apply_text_search_filter(base_query, query: str):
    return search_service.apply_text_search_filter(base_query, query)


class TimedCache:
    def __init__(self, max_items: int = 128, ttl: float = 60.0):
        self.max_items = max(1, int(max_items))
        self.ttl = float(ttl)
        self._store: OrderedDict[tuple, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self) -> None:
        now = time.time()
        dead = [key for key, (expires, _val) in self._store.items() if expires <= now]
        for key in dead:
            self._store.pop(key, None)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    def get(self, key: tuple) -> object | None:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires, value = item
            if expires <= time.time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: tuple, value: object) -> None:
        with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
            self._store.move_to_end(key)
            self._prune()

    def get_or_set(self, key: tuple, factory):
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

_RU_SYNONYMS = {
    'машина': ['автомобиль', 'авто', 'тачка', 'car'],
    'изображение': ['картинка', 'фото', 'фотография', 'image', 'picture', 'снимок', 'скриншот', 'screenshot'],
    'вуз': ['университет', 'институт', 'универ'],
    'статья': ['публикация', 'paper', 'publication', 'пейпер'],
    'диссертация': ['дисс', 'thesis', 'dissertation'],
    'журнал': ['journal', 'magazine', 'сборник'],
    'номер': ['выпуск', 'issue'],
    'страница': ['стр', 'страницы', 'pages', 'page'],
    'год': ['лет', 'г'],
}

def _ru_tokens(text: str) -> list[str]:
    s = (text or '').lower()
    if _ru_tokenize:
        try:
            return [t.text for t in _ru_tokenize(s)]
        except Exception:
            pass
    import re as _re
    return _re.sub(r"[^\w\d]+", " ", s).strip().split()

def _lemma(word: str) -> str:
    if _morph:
        try:
            p = _morph.parse(word)
            if p:
                return p[0].normal_form
        except Exception:
            pass
    return word.lower()

def _lemmas(text: str) -> list[str]:
    return [_lemma(w) for w in _ru_tokens(text)]

def _expand_synonyms(lemmas: list[str]) -> set[str]:
    out = set(lemmas)
    for l in list(lemmas):
        for s in _RU_SYNONYMS.get(l, []):
            out.add(_lemma(s))
    return out

LLM_ROUND_ROBIN: dict[str, itertools.cycle] = {}
LLM_ENDPOINT_SIGNATURE: dict[str, tuple[tuple[int, float, tuple[str, ...]], ...]] = {}
LLM_ENDPOINT_POOLS: dict[str, list[dict[str, str]]] = {}
LLM_ENDPOINT_UNIQUE: dict[str, list[dict[str, str]]] = {}
LLM_SCHEMA_READY = False
LLM_BUSY_HTTP_CODES = {409, 423, 429, 503}
LLM_BUSY_STATUS_VALUES = {'busy', 'processing', 'in_progress', 'queued', 'pending', 'rate_limited'}
LLM_PURPOSES = [
    {'id': 'summary', 'label': 'Резюме стенограмм'},
    {'id': 'keywords', 'label': 'Ключевые слова'},
    {'id': 'compose', 'label': 'Генерация ответов'},
    {'id': 'metadata', 'label': 'Извлечение метаданных'},
    {'id': 'vision', 'label': 'Анализ изображений'},
    {'id': 'rerank', 'label': 'Реранжирование поиска'},
    {'id': 'default', 'label': 'По умолчанию'},
]
LLM_PROVIDER_OPTIONS = [
    {'id': 'openai', 'label': 'OpenAI-совместимый (LM Studio, OpenAI, OpenRouter)'},
    {'id': 'azure_openai', 'label': 'Azure OpenAI'},
    {'id': 'ollama', 'label': 'Ollama (локальные модели)'},
]
TASK_RETENTION_WINDOW = timedelta(days=1)
TASK_FINAL_STATUSES = ('completed', 'error', 'cancelled')
TASK_STUCK_STATUSES = ('running', 'pending', 'queued', 'cancelling')

_RAG_RERANK_LOCK = threading.Lock()
_RAG_RERANK_CACHE: Dict[str, "CrossEncoderReranker"] = {}


def _reset_rag_rerank_cache(*, locked: bool = False) -> None:
    if not locked:
        with _RAG_RERANK_LOCK:
            _reset_rag_rerank_cache(locked=True)
        return
    for instance in _RAG_RERANK_CACHE.values():
        try:
            instance.close()
        except Exception:
            pass
    _RAG_RERANK_CACHE.clear()


def _get_rag_reranker() -> Optional["CrossEncoderReranker"]:
    runtime = _rt()
    backend_raw = str(getattr(runtime, "rag_rerank_backend", "none") or "").strip().lower()
    if backend_raw not in {"cross-encoder", "cross_encoder", "cross"}:
        if _RAG_RERANK_CACHE:
            _reset_rag_rerank_cache()
        return None
    model_name = str(getattr(runtime, "rag_rerank_model", "") or "").strip()
    if not model_name:
        return None
    device = getattr(runtime, "rag_rerank_device", None)
    batch_size = max(1, int(getattr(runtime, "rag_rerank_batch_size", 16) or 16))
    max_length = max(32, int(getattr(runtime, "rag_rerank_max_length", 512) or 512))
    max_chars = max(200, int(getattr(runtime, "rag_rerank_max_chars", 1200) or 1200))
    cache_key = f"{backend_raw}::{model_name}::{device or ''}::{batch_size}::{max_length}::{max_chars}"
    with _RAG_RERANK_LOCK:
        existing = _RAG_RERANK_CACHE.get(cache_key)
        if existing:
            return existing
        _reset_rag_rerank_cache(locked=True)
        try:
            from agregator.rag.rerank import CrossEncoderConfig, load_reranker

            reranker = load_reranker(
                backend_raw,
                config=CrossEncoderConfig(
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length,
                    truncate_chars=max_chars,
                ),
            )
        except Exception as exc:
            logger.warning("Не удалось инициализировать cross-encoder rerank (%s/%s): %s", backend_raw, model_name, exc)
            return None
        if reranker is None:
            return None
        _RAG_RERANK_CACHE[cache_key] = reranker
        return reranker

def _row_text_for_search(f: File) -> str:
    parts = [
        f.title or '', f.author or '', f.keywords or '', f.filename or '',
        getattr(f, 'text_excerpt', '') or '', getattr(f, 'abstract', '') or ''
    ]
    try:
        parts.extend([f"{t.key}={t.value}" for t in (f.tags or []) if t and t.key and t.value])
    except Exception:
        pass
    return ' '.join(parts)

# Простое кэширующее хранилище в памяти для расширения ключевых слов ИИ
AI_EXPAND_CACHE: dict[str, tuple[float, list[str], list[tuple[str, int]]]] = {}
AI_KEYWORD_PLAN_CACHE: dict[str, tuple[float, Dict[str, List[str]]]] = {}

# Весовые коэффициенты и параметры оценки (можно менять через переменные окружения)
def _getf(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

AI_SCORE_TITLE = _getf('AI_SCORE_TITLE', 2.5)
AI_SCORE_AUTHOR = _getf('AI_SCORE_AUTHOR', 1.5)
AI_SCORE_KEYWORDS = _getf('AI_SCORE_KEYWORDS', 1.2)
AI_SCORE_EXCERPT = _getf('AI_SCORE_EXCERPT', 1.0)
AI_SCORE_ABSTRACT = _getf('AI_SCORE_ABSTRACT', 1.0)
AI_SCORE_TAG = _getf('AI_SCORE_TAG', 1.0)
AI_SCORE_FTS = _getf('AI_SCORE_FTS', 1.8)
AI_SCORE_FTS_RAW = _getf('AI_SCORE_FTS_RAW', 2.2)
AI_BOOST_PHRASE = _getf('AI_BOOST_PHRASE', 3.0)
AI_BOOST_MULTI = _getf('AI_BOOST_MULTI', 0.6)  # дополнительный бонус за каждое уникальное слово
AI_BOOST_SNIPPET_COOCCUR = _getf('AI_BOOST_SNIPPET_COOCCUR', 0.8)
SNIPPET_CACHE_TTL_HOURS = int(os.getenv('AI_SNIPPET_CACHE_TTL_HOURS', '24'))
KEYWORD_IDF_MIN = float(os.getenv('KEYWORD_IDF_MIN', '1.25'))
CACHE_CLEANUP_INTERVAL_HOURS = int(os.getenv('AI_SNIPPET_CACHE_SWEEP_INTERVAL_HOURS', '24'))

def _now() -> float:
    return time.time()

def _sha256(s: str) -> str:
    try:
        return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _query_fingerprint(query: str, payload: dict | None = None) -> str:
    payload = payload or {}
    serialized = {
        'query': query.strip(),
        'top_k': payload.get('top_k'),
        'deep_search': bool(payload.get('deep_search')),
        'full_text': bool(payload.get('full_text')),
        'llm_snippets': bool(payload.get('llm_snippets')),
        'sources': payload.get('sources'),
        'filters': {
            'collection_ids': payload.get('collection_ids') or payload.get('collection_id'),
            'material_types': payload.get('material_types'),
            'year_from': payload.get('year_from'),
            'year_to': payload.get('year_to'),
            'tag_filters': payload.get('tag_filters'),
        }
    }
    try:
        data = json.dumps(serialized, sort_keys=True, ensure_ascii=False)
    except Exception:
        data = str(serialized)
    return _sha256(data)

def _get_cached_snippet(file_id: int, query_hash: str, llm_variant: bool):
    if not file_id or not query_hash:
        return None
    try:
        entry = AiSearchSnippetCache.query.filter_by(file_id=file_id, query_hash=query_hash, llm_variant=llm_variant).first()
    except Exception:
        entry = None
    if not entry:
        return None
    if entry.expires_at and entry.expires_at < datetime.utcnow():
        try:
            db.session.delete(entry)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return None
    return entry


def _store_snippet_cache(file_id: int, query_hash: str, llm_variant: bool, snippet: str, meta: dict | None = None, ttl_hours: int | None = None):
    if not file_id or not query_hash or not snippet:
        return
    ttl = ttl_hours if ttl_hours is not None else SNIPPET_CACHE_TTL_HOURS
    expires_at = datetime.utcnow() + timedelta(hours=max(1, ttl)) if ttl else None
    try:
        entry = AiSearchSnippetCache.query.filter_by(file_id=file_id, query_hash=query_hash, llm_variant=llm_variant).first()
        meta_json = json.dumps(meta, ensure_ascii=False) if meta else None
        if entry:
            entry.snippet = snippet
            entry.meta = meta_json
            entry.expires_at = expires_at
        else:
            entry = AiSearchSnippetCache(
                file_id=file_id,
                query_hash=query_hash,
                llm_variant=llm_variant,
                snippet=snippet,
                meta=meta_json,
                expires_at=expires_at,
            )
            db.session.add(entry)
        db.session.commit()
    except Exception:
        db.session.rollback()


def _prune_expired_snippet_cache() -> int:
    cutoff = datetime.utcnow()
    try:
        q = AiSearchSnippetCache.query.filter(
            AiSearchSnippetCache.expires_at.isnot(None),
            AiSearchSnippetCache.expires_at < cutoff
        )
        deleted = q.delete(synchronize_session=False)
        if deleted:
            db.session.commit()
            app.logger.info(f"[ai-search] snippet cache cleanup removed {deleted} rows")
        else:
            db.session.commit()
        return deleted or 0
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(f"[ai-search] snippet cache cleanup failed: {exc}")
        return 0


_CLEANUP_THREAD_STARTED = False


def _start_cache_cleanup_scheduler() -> None:
    global _CLEANUP_THREAD_STARTED
    if _CLEANUP_THREAD_STARTED or CACHE_CLEANUP_INTERVAL_HOURS <= 0:
        return

    interval = max(1, CACHE_CLEANUP_INTERVAL_HOURS) * 3600

    def loop():
        while True:
            try:
                with app.app_context():
                    _prune_expired_snippet_cache()
            except Exception:
                app.logger.exception("[ai-search] cache cleanup loop error")
            time.sleep(interval)

    threading.Thread(target=loop, name='ai-snippet-cache-cleaner', daemon=True).start()
    _CLEANUP_THREAD_STARTED = True

def _record_search_metric(query_hash: str, durations: dict[str, float], user: User | None, meta_extra: dict | None = None):
    try:
        known_fields = {'total', 'keywords', 'candidates', 'deep', 'llm_answer', 'llm_snippets'}
        extra_meta = {}
        for key, value in durations.items():
            if key not in known_fields and value is not None:
                extra_meta[key] = value
        if meta_extra:
            for key, value in meta_extra.items():
                if value is None:
                    continue
                extra_meta[key] = value
        metric = AiSearchMetric(
            query_hash=query_hash,
            user_id=getattr(user, 'id', None),
            total_ms=int(durations.get('total', 0) * 1000),
            keywords_ms=int(durations.get('keywords', 0) * 1000) if 'keywords' in durations else None,
            candidate_ms=int(durations.get('candidates', 0) * 1000) if 'candidates' in durations else None,
            deep_ms=int(durations.get('deep', 0) * 1000) if 'deep' in durations else None,
            llm_answer_ms=int(durations.get('llm_answer', 0) * 1000) if 'llm_answer' in durations else None,
            llm_snippet_ms=int(durations.get('llm_snippets', 0) * 1000) if 'llm_snippets' in durations else None,
            meta=json.dumps(extra_meta, ensure_ascii=False) if extra_meta else None,
        )
        db.session.add(metric)
        db.session.commit()
    except Exception:
        db.session.rollback()

# ------------------- Конфигурация -------------------

CONFIG = load_app_config()


def _apply_config_defaults(cfg: AppConfig) -> None:
    global BASE_DIR, LOGIN_BACKGROUNDS_DIR, RENAME_PATTERNS, DEFAULT_PROMPTS
    global FW_CACHE_DIR, SETTINGS_STORE_PATH, LOG_DIR, LOG_FILE_PATH
    global HTTP_DEFAULT_TIMEOUT, HTTP_CONNECT_TIMEOUT, HTTP_RETRIES, HTTP_BACKOFF_FACTOR
    global LOG_LEVEL, SENTRY_DSN, SENTRY_ENVIRONMENT

    runtime_settings_store.initialize(cfg)

    BASE_DIR = cfg.base_dir
    LOGIN_BACKGROUNDS_DIR = cfg.login_backgrounds_dir
    RENAME_PATTERNS = dict(cfg.rename_patterns)
    DEFAULT_PROMPTS = dict(cfg.default_prompts)
    FW_CACHE_DIR = cfg.fw_cache_dir
    SETTINGS_STORE_PATH = cfg.settings_store_path
    LOG_DIR = cfg.logs_dir
    LOG_FILE_PATH = cfg.log_file_path
    HTTP_DEFAULT_TIMEOUT = float(cfg.http_timeout)
    HTTP_CONNECT_TIMEOUT = float(cfg.http_connect_timeout)
    HTTP_RETRIES = int(cfg.http_retries)
    HTTP_BACKOFF_FACTOR = float(cfg.http_backoff_factor)
    LOG_LEVEL = cfg.log_level
    SENTRY_DSN = cfg.sentry_dsn
    SENTRY_ENVIRONMENT = cfg.sentry_environment

    _refresh_runtime_globals()
    _ensure_rag_embedding_defaults()


_apply_config_defaults(CONFIG)
LLM_PROVIDER_CHOICES = {'openai', 'openrouter', 'azure_openai', 'ollama'}
if runtime_settings_store.current.lm_default_provider not in LLM_PROVIDER_CHOICES:
    runtime_settings_store.current.lm_default_provider = 'openai'

# ------------------- Сохранение настроек во время работы -------------------

# ------------------- Сохранение настроек во время работы -------------------

# ------------------- Facet configuration (search & graph) -------------------

def _facet_config_state() -> dict:
    runtime = _rt()
    return {
        'search': {
            'include_types': bool(runtime.search_facet_include_types),
            'tag_keys': list(runtime.search_facet_tag_keys) if runtime.search_facet_tag_keys is not None else None,
        },
        'graph': {
            'tag_keys': list(runtime.graph_facet_tag_keys) if runtime.graph_facet_tag_keys is not None else None,
        }
    }


def _load_runtime_settings_from_disk() -> None:
    if not SETTINGS_STORE_PATH.exists():
        return
    try:
        data = json.loads(SETTINGS_STORE_PATH.read_text(encoding='utf-8'))
    except Exception as exc:
        logger.warning("Не удалось прочитать настройки: %s", exc)
        return
    runtime_settings_store.apply_updates(data)
    _reset_rag_rerank_cache()
    runtime = _rt()
    if runtime.lm_default_provider not in LLM_PROVIDER_CHOICES:
        runtime.lm_default_provider = 'openai'
    _refresh_runtime_globals()
    configure_llm_cache(
        enabled=runtime.llm_cache_enabled,
        max_items=runtime.llm_cache_max_items,
        ttl_seconds=runtime.llm_cache_ttl_seconds,
    )
    configure_search_cache(
        enabled=runtime.search_cache_enabled,
        max_items=runtime.search_cache_max_items,
        ttl_seconds=runtime.search_cache_ttl_seconds,
    )
    app_ref = globals().get('app')
    if app_ref:
        runtime.apply_to_flask_config(app_ref)


def _save_runtime_settings_to_disk() -> None:
    try:
        SETTINGS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot = runtime_settings_store.snapshot()
        SETTINGS_STORE_PATH.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    except Exception as exc:
        logger.warning("Не удалось сохранить настройки: %s", exc)


def _runtime_settings_snapshot() -> dict:
    return runtime_settings_store.snapshot()


def _apply_runtime_settings_to_config(app_obj: Flask) -> None:
    runtime_settings_store.current.apply_to_flask_config(app_obj)


# ------------------- Lightweight response helpers -------------------

def json_error(message: str, status: int = 400):
    """Shortcut для JSON-ошибок."""
    return jsonify({"error": str(message)}), int(status)


def _html_page(title: str, body: str, *, extra_head: str = "", status: int = 200) -> Response:
    head = f"""
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{escape(title)}</title>
    {extra_head}
    """
    html = f"""<!doctype html><html lang=\"ru\"><head>{head}</head><body>{body}</body></html>"""
    resp = make_response(html, status)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    return resp




_PREVIEW_HEAD = """
<style>
  :root { color-scheme: dark light; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#0d1117; color:#c9d1d9; }
  a { color:#58a6ff; }
  .preview-shell { min-height:100vh; display:flex; flex-direction:column; }
  .preview-shell.embedded { min-height:auto; }
  .preview-header { padding:12px 16px; display:flex; align-items:center; gap:12px; justify-content:space-between; background:#161b22; border-bottom:1px solid #30363d; }
  .preview-title { font-size:16px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .preview-actions { display:flex; gap:8px; }
  .btn { display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:8px 16px; border-radius:999px; font-size:13px;
         border:1px solid #30363d; background:#1f6feb; color:#fff; text-decoration:none; }
  .btn.secondary { background:transparent; color:#c9d1d9; }
  .preview-main { flex:1; overflow:auto; padding:16px; }
  .preview-shell.embedded .preview-main { padding:12px; }
  .preview-card { background:#161b22; border:1px solid #30363d; border-radius:14px; padding:16px; box-shadow:0 16px 48px rgba(0,0,0,0.35); }
  .preview-text { white-space:pre-wrap; max-height:70vh; overflow:auto; word-break:break-word; overflow-wrap:anywhere; background:#0d1117; padding:12px; border-radius:10px; border:1px solid #30363d; }
  .muted { color:#8b949e; font-size:13px; margin:12px 0 6px; }
  .section { margin-top:18px; }
  .section h3 { margin:0 0 8px; font-size:15px; font-weight:600; color:inherit; }
  img.responsive { max-width:100%; height:auto; border-radius:12px; border:1px solid #30363d; }
  .badge { display:inline-block; padding:4px 10px; border-radius:999px; background:#30363d; color:#c9d1d9; font-size:12px; margin-left:6px; }
  .tag { display:inline-block; padding:4px 10px; border-radius:999px; background:#30363d; color:#c9d1d9; font-size:12px; margin:2px 4px 2px 0; }
  mark { background:#bb8009; color:inherit; padding:0 2px; border-radius:3px; }
  embed, iframe { border:none; border-radius:10px; background:#0d1117; }
  audio { width:100%; }
  .docx-preview { background:#0d1117; border:1px solid #30363d; border-radius:10px; max-height:70vh; overflow:auto; padding:16px; line-height:1.55; }
  .docx-preview p { margin:0 0 0.9em; }
  .docx-preview h1, .docx-preview h2, .docx-preview h3, .docx-preview h4 { margin-top:1.2em; margin-bottom:0.6em; }
  .docx-preview ul, .docx-preview ol { padding-left:1.3em; }
  @media (prefers-color-scheme: light) {
    body { background:#f5f6f8; color:#101828; }
    .preview-header { background:#ffffff; border-color:rgba(15,23,42,0.09); }
    .preview-card { background:#ffffff; border-color:rgba(15,23,42,0.08); box-shadow:0 12px 36px rgba(15,23,42,0.2); }
    .preview-text { background:#f8fafc; border-color:rgba(15,23,42,0.08); }
    .btn { background:#1f6feb; color:#fff; border:1px solid #1f6feb; }
    .btn.secondary { background:#fff; color:#0f172a; border-color:rgba(15,23,42,0.14); }
    .muted { color:#64748b; }
    .badge, .tag { background:rgba(15,23,42,0.08); color:#0f172a; }
    .docx-preview { background:#f8fafc; border-color:rgba(15,23,42,0.08); }
  }
</style>
<script>
(function(){
  try {
    const params = new URLSearchParams(window.location.search);
    const mark = (params.get('mark') || '').trim();
    if (!mark) return;
    const terms = Array.from(new Set(mark.split(/[\\s,|]+/).filter(x => x && x.length >= 2))).slice(0, 5);
    if (!terms.length) return;
    const esc = s => s.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
    const re = new RegExp('(' + terms.map(esc).join('|') + ')', 'gi');
    document.querySelectorAll('.preview-text, .docx-preview').forEach(el => {
      try { el.innerHTML = el.innerHTML.replace(re, '<mark>$1</mark>'); } catch (_) {}
    });
  } catch (err) {
    console.warn('highlight error', err);
  }
})();
</script>
""".strip()


def _sanitize_html_fragment(html_fragment: str) -> str:
    if not html_fragment:
        return ''
    cleaned = re.sub(r'(?is)<script.*?>.*?</script>', '', html_fragment)
    cleaned = re.sub(r'(?is)<style.*?>.*?</style>', '', cleaned)
    cleaned = re.sub(r'(?i) on[a-z]+="[^"]*"', '', cleaned)
    cleaned = re.sub(r"(?i) on[a-z]+='[^']*'", '', cleaned)
    cleaned = re.sub(r'(?i)javascript:', '', cleaned)
    return cleaned


def _docx_to_html(path: Path, limit_chars: int = 60000) -> str | None:
    if mammoth is None:
        return None
    try:
        with path.open('rb') as docx_file:
            result = mammoth.convert_to_html(docx_file)
        html_fragment = result.value or ''
        sanitized = _sanitize_html_fragment(html_fragment)
        if limit_chars:
            return sanitized[:limit_chars]
        return sanitized
    except Exception as exc:
        app.logger.warning(f"DOCX preview conversion failed for {path}: {exc}")
        return None


def _render_preview(
    rel_url: str,
    *,
    is_pdf: bool,
    is_text: bool,
    is_audio: bool,
    is_image: bool,
    is_docx: bool,
    content: str,
    thumbnail_url: str | None,
    abstract: str,
    audio_url: str | None,
    duration: str | None,
    image_url: str | None,
    keywords: str | None,
    embedded: bool,
    docx_html: str | None
) -> Response:
    safe_rel = escape(rel_url)
    download_url = escape(url_for('download_file', rel_path=rel_url))
    view_url = escape(url_for('view_file', rel_path=rel_url))
    shell_class = "preview-shell embedded" if embedded else "preview-shell"

    sections: list[str] = []

    def section(title: str, html_content: str) -> None:
        sections.append(f'<div class="section"><h3>{escape(title)}</h3>{html_content}</div>')

    if is_pdf:
        sections.append(f'<embed src="{view_url}" type="application/pdf" width="100%" style="min-height:70vh;" />')
        if content:
            section('Фрагмент содержимого', f'<div class="preview-text">{escape(content)}</div>')
    elif is_audio:
        if audio_url:
            audio = f'<audio controls preload="metadata"><source src="{escape(audio_url)}" /></audio>'
            if duration:
                audio += f'<span class="badge">{escape(duration)}</span>'
            sections.append(audio)
        if abstract:
            section('Резюме', f'<div class="preview-text">{escape(abstract)}</div>')
        if keywords:
            chips = ''.join(f'<span class="tag">{escape(k.strip())}</span>' for k in keywords.split(',') if k.strip())
            if chips:
                section('Ключевые слова', chips)
        if content and not abstract:
            section('Фрагмент стенограммы', f'<div class="preview-text">{escape(content)}</div>')
    elif is_image:
        img_src = escape(image_url or url_for('media_file', rel_path=rel_url))
        sections.append(f'<img class="responsive" src="{img_src}" alt="image preview" />')
        if abstract:
            section('Описание', f'<div class="preview-text">{escape(abstract)}</div>')
        if keywords:
            chips = ''.join(f'<span class="tag">{escape(k.strip())}</span>' for k in keywords.split(',') if k.strip())
            if chips:
                section('Ключевые слова', chips)
    elif is_docx:
        if docx_html:
            section('Документ', f'<div class="docx-preview">{docx_html}</div>')
        elif content:
            section('Текст', f'<div class="preview-text">{escape(content)}</div>')
        else:
            sections.append('<div class="muted">Не удалось построить предпросмотр DOCX.</div>')
    elif is_text:
        section('Текст', f'<div class="preview-text">{escape(content)}</div>')
    elif thumbnail_url:
        sections.append(f'<img class="responsive" src="{escape(thumbnail_url)}" alt="thumbnail" />')
        if content:
            section('Фрагмент содержимого', f'<div class="preview-text">{escape(content)}</div>')
    else:
        if content:
            sections.append('<div class="muted">Предпросмотр для этого формата ограничен. Ниже показан доступный текст:</div>')
            sections.append(f'<div class="preview-text">{escape(content)}</div>')
        else:
            sections.append('<div class="muted">Предпросмотр не поддерживается для этого типа файла.</div>')

    if abstract and not any('Резюме' in s for s in sections):
        section('Резюме', f'<div class="preview-text">{escape(abstract)}</div>')
    if keywords and not any('Ключевые слова' in s for s in sections):
        chips = ''.join(f'<span class="tag">{escape(k.strip())}</span>' for k in keywords.split(',') if k.strip())
        if chips:
            section('Ключевые слова', chips)

    header_actions = f'<a class="btn" href="{download_url}" download>Скачать</a>'
    body = f"""
    <div class=\"{shell_class}\">
      <header class=\"preview-header\">
        <div class=\"preview-title\">Предпросмотр: {safe_rel}</div>
        <div class=\"preview-actions\">{header_actions}</div>
      </header>
      <main class=\"preview-main\">
        <div class=\"preview-card\">{''.join(sections)}</div>
      </main>
    </div>
    """
    return _html_page(f"Предпросмотр — {rel_url}", body, extra_head=_PREVIEW_HEAD)

# ------------------- Вспомогательные функции -------------------

def _normalize_author(val):
    """Convert various author representations (list/dict/etc.) to a string.
    Returns None for empty results.
    """
    try:
        if val is None:
            return None
        # список/кортеж имён или словарей
        if isinstance(val, (list, tuple, set)):
            out = []
            for x in val:
                if x is None:
                    continue
                if isinstance(x, dict):
                    if 'name' in x and x['name']:
                        s = str(x['name']).strip()
                    else:
                        s = " ".join(str(v).strip() for v in x.values() if v)
                else:
                    s = str(x).strip()
                if s:
                    out.append(s)
            return ", ".join(out) if out else None
        # один словарь вида {first,last} или {name}
        if isinstance(val, dict):
            if 'name' in val and val['name']:
                s = str(val['name']).strip()
                return s or None
            s = " ".join(str(v).strip() for v in val.values() if v)
            return s or None
        s = str(val).strip()
        return s or None
    except Exception:
        return None

def _normalize_year(val):
    """Normalize year to a short string; prefer 4-digit if present."""
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return str(int(val))
        s = str(val).strip()
        m = re.search(r"\b(\d{4})\b", s)
        if m:
            return m.group(1)
        return s[:16] if s else None
    except Exception:
        return None

# --------- Расширенное извлечение тегов (общие и специфичные по типу) ---------
def extract_richer_tags(material_type: str, text: str, filename: str = "") -> dict:
    t = (text or "")
    tl = t.lower()
    tags: dict[str, str] = {}
    # предположение языка
    try:
        cyr = sum(1 for ch in t if ('а' <= ch.lower() <= 'я') or (ch in 'ёЁ'))
        lat = sum(1 for ch in t if 'a' <= ch.lower() <= 'z')
        if cyr + lat > 20:
            tags['lang'] = 'ru' if cyr >= lat else 'en'
    except Exception:
        pass
    # распространённые идентификаторы
    m = re.search(r"\b(10\.\d{4,9}\/[\w\-\.:;()\/[\]A-Za-z0-9]+)", t)
    if m:
        tags.setdefault('doi', m.group(1))
    m = re.search(r"\bISBN[:\s]*([0-9\- ]{10,20})", t, flags=re.I)
    if m:
        tags.setdefault('isbn', m.group(1).strip())
    m = re.search(r"\bISSN[:\s]*([0-9\- ]{8,15})\b", t, flags=re.I)
    if m:
        tags.setdefault('issn', m.group(1).strip())
    m = re.search(r"\bУДК[:\s]*([\d\.:\-]+)\b", t, flags=re.I)
    if m:
        tags.setdefault('udk', m.group(1))
    m = re.search(r"\bББК[:\s]*([A-ZА-Я0-9\.-/]+)\b", t, flags=re.I)
    if m:
        tags.setdefault('bbk', m.group(1))

    mt = (material_type or '').strip().lower()
    if mt in ("dissertation", "dissertation_abstract"):
        # код специальности вида 05.13.11
        m = re.search(r"\b(\d{2}\.\d{2}\.\d{2})\b", t)
        if m:
            tags.setdefault('specialty', m.group(1))
        if 'автореферат' in tl:
            tags.setdefault('kind', 'автореферат')
    elif mt == 'article':
        m = re.search(r"(journal|transactions|вестник|журнал)[:\s\-]+([^\n\r]{3,80})", tl, flags=re.I)
        if m:
            tags.setdefault('journal', m.group(2).strip().title())
        m = re.search(r"\b(\d+)\b.*?№\s*(\d+)\b.*?([\d]+)[\-–—]([\d]+)", t)
        if m:
            tags.setdefault('volume_issue', f"{m.group(1)}/{m.group(2)}")
            tags.setdefault('pages', f"{m.group(3)}–{m.group(4)}")
    elif mt == 'journal':
        m = re.search(r"(journal|transactions|вестник|журнал)[:\s\-]+([^\n\r]{3,80})", tl, flags=re.I)
        if m:
            tags.setdefault('journal', m.group(2).strip().title())
        m = re.search(r"\b(\d+)\b.*?№\s*(\d+)\b", t)
        if m:
            tags.setdefault('volume_issue', f"{m.group(1)}/{m.group(2)}")
        m = re.search(r"№\s*(\d+)\b", t)
        if m:
            tags.setdefault('number', m.group(1))
        toc_entries = _extract_journal_toc_entries(t)
        if toc_entries:
            formatted: list[str] = []
            for entry in toc_entries[:25]:
                title = entry.get('title', '').strip()
                if not title:
                    continue
                piece = title
                authors = entry.get('authors')
                page = entry.get('page')
                if authors:
                    piece = f"{authors}: {piece}"
                if page:
                    piece = f"{piece} (стр. {page})"
                formatted.append(piece)
            if formatted:
                tags.setdefault('toc', formatted)
    elif mt == 'textbook':
        m = re.search(r"учебн(?:ое|ик|ое\s+пособие)\s*[:\-]?\s*([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault('discipline', m.group(1).strip().title())
    elif mt == 'monograph':
        m = re.search(r"(серия|series)\s*[:\-]?\s*([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault('series', m.group(2).strip().title())
    elif mt == 'standard':
        pats = [
            r"\bГОСТ\s*R?\s*\d{1,5}(?:\.\d+)*[-–—]\d{2,4}\b",
            r"\bСТБ\s*\d{1,5}(?:\.\d+)*[-–—]\d{2,4}\b",
            r"\bСТО\s*[^\s]*\s*\d{1,6}[-–—]?\d{2,4}\b",
            r"\bISO\s*\d{3,5}(?:-\d+)*(?::\d{4})?\b",
            r"\bIEC\s*\d{3,5}(?:-\d+)*(?::\d{4})?\b",
            r"\bСП\s*\d{1,4}\.\d{1,4}-\d{4}\b",
            r"\bСанПиН\s*\d+[\.-]\d+[\.-]\d+\b",
            r"\bТУ\s*[A-Za-zА-Яа-я0-9\./-]+\b",
        ]
        for pat in pats:
            m = re.search(pat, t, flags=re.I)
            if m:
                tags.setdefault('standard', m.group(0))
                break
        if re.search(r"утратил[аи]?\s+силу|замен(ен|яет)|взамен", tl):
            tags.setdefault('status', 'replaced')
        elif re.search(r"введ(ен|ена)\s+впервые|действующ", tl):
            tags.setdefault('status', 'active')
    elif mt == 'proceedings':
        m = re.search(r"(материалы\s+конференции|proceedings\s+of\s+the|international\s+conference|symposium\s+on|workshop\s+on)[^\n\r]{0,120}", tl, flags=re.I)
        if m:
            tags.setdefault('conference', m.group(0).strip().title())
    elif mt == 'report':
        if re.search(r"техническое\s+задание\b|ТЗ\b", tl):
            tags.setdefault('doc_kind', 'Техническое задание')
        if re.search(r"пояснительная\s+записка\b", tl):
            tags.setdefault('doc_kind', 'Пояснительная записка')
    elif mt == 'patent':
        m = re.search(r"\b(?:RU|SU|US|WO|EP)\s?\d{4,10}[A-Z]?\d?\b", t)
        if m:
            tags.setdefault('patent_no', m.group(0))
        m = re.search(r"\b[A-H][0-9]{2}[A-Z]\s*\d+\/\d+\b", t)
        if m:
            tags.setdefault('ipc', m.group(0).replace(' ', ''))
    elif mt == 'presentation':
        if re.search(r"(слайды|slides|powerpoint|презентация)", tl):
            tags.setdefault('slides', 'yes')
    return tags

def _fw_alias_to_repo(ref: str) -> str | None:
    r = (ref or '').strip().lower()
    # Распространённые псевдонимы
    alias = {
        'tiny': 'Systran/faster-whisper-tiny',
        'base': 'Systran/faster-whisper-base',
        'small': 'Systran/faster-whisper-small',
        'medium': 'Systran/faster-whisper-medium',
        'large-v2': 'Systran/faster-whisper-large-v2',
        'large-v3': 'Systran/faster-whisper-large-v3',
        # Сжатые английские варианты
        'distil-small.en': 'Systran/faster-distil-whisper-small.en',
        'distil-medium.en': 'Systran/faster-distil-whisper-medium.en',
        'distil-large-v2': 'Systran/faster-distil-whisper-large-v2',
    }
    if r in alias:
        return alias[r]
    # Также принимаем форму org/name
    if '/' in r and len(r.split('/', 1)[0]) > 0:
        return ref
    return None

def _ensure_faster_whisper_model(model_ref: str) -> str:
    """Гарантировать локальный путь к модели faster-whisper.
    Если указан локальный каталог — используем его. Если это алиас или repo id — скачиваем в кэш.
    Возвращает путь к модели или пустую строку при ошибке.
    """
    try:
        if not model_ref:
            return ''
        p = Path(model_ref).expanduser()
        if p.exists() and p.is_dir():
            return str(p)
        repo = _fw_alias_to_repo(model_ref)
        if not repo:
            return ''
        if hf_snapshot_download is None:
            app.logger.warning("Пакет huggingface_hub не установлен — автозагрузка модели faster-whisper недоступна")
            return ''
        # вычислить целевую директорию внутри кэша
        FW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = repo.replace('/', '__')
        target_dir = FW_CACHE_DIR / safe_name
        if not target_dir.exists() or not any(target_dir.iterdir()):
            # загрузить/снять снимок в target_dir
            hf_snapshot_download(repo_id=repo, local_dir=str(target_dir), local_dir_use_symlinks=False, revision=None)
        return str(target_dir)
    except Exception as e:
        app.logger.warning(f"Failed to resolve faster-whisper model '{model_ref}': {e}")
        return ''

app = Flask(__name__)


def _initialize_database_structures(app_obj: Flask) -> None:
    """Ensure ORM metadata, FTS tables and task queue state are ready."""
    with app_obj.app_context():
        db.create_all()
        _ensure_search_support()
        _reset_inflight_tasks()


def setup_app(config: AppConfig | None = None, *, ensure_database: bool = True) -> Flask:
    global CONFIG, LOG_DIR, LOG_FILE_PATH
    cfg = config or CONFIG
    if config is not None:
        CONFIG = cfg
        _apply_config_defaults(cfg)
        if runtime_settings_store.current.lm_default_provider not in LLM_PROVIDER_CHOICES:
            runtime_settings_store.current.lm_default_provider = 'openai'
    LOG_DIR = cfg.logs_dir
    LOG_FILE_PATH = cfg.log_file_path
    configure_http(
        HttpSettings(
            timeout=cfg.http_timeout,
            connect_timeout=cfg.http_connect_timeout,
            retries=cfg.http_retries,
            backoff_factor=cfg.http_backoff_factor,
        )
    )
    configure_llm_cache(
        enabled=cfg.llm_cache_enabled,
        max_items=cfg.llm_cache_max_items,
        ttl_seconds=cfg.llm_cache_ttl_seconds,
    )
    configure_search_cache(
        enabled=cfg.search_cache_enabled,
        max_items=cfg.search_cache_max_items,
        ttl_seconds=cfg.search_cache_ttl_seconds,
    )
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        fallback_dir = cfg.base_dir
        fallback_dir.mkdir(parents=True, exist_ok=True)
        LOG_DIR = fallback_dir
        LOG_FILE_PATH = LOG_DIR / 'agregator.log'
    configure_logging(
        app,
        LOG_FILE_PATH,
        level=LOG_LEVEL,
        sentry_dsn=SENTRY_DSN or None,
        sentry_environment=SENTRY_ENVIRONMENT or None,
    )
    # Гарантируем запуск очереди фоновых задач
    get_task_queue().start()
    app.secret_key = cfg.flask_secret_key
    app.config.update(cfg.to_flask_config())
    app.config['MAX_CONTENT_LENGTH'] = cfg.max_content_length
    db.init_app(app)
    try:
        with app.app_context():
            with db.engine.connect() as conn:
                conn.execute(text("PRAGMA foreign_keys = 1"))
                conn.execute(text("PRAGMA trusted_schema = 1"))
                conn.execute(text("PRAGMA recursive_triggers = 1"))
            db.engine.dispose()
    except Exception as pragma_exc:
        logger.warning('Failed to apply SQLite PRAGMA settings: %s', pragma_exc)
    _load_runtime_settings_from_disk()
    _apply_runtime_settings_to_config(app)
    if 'admin_api' not in app.blueprints:
        app.register_blueprint(admin_bp)
    if 'users_api' not in app.blueprints:
        app.register_blueprint(users_bp)
    if ensure_database:
        _initialize_database_structures(app)
    with app.app_context():
        _schedule_facet_cache_rebuild('startup')
    return app


def _get_rotating_log_handler() -> RotatingFileHandler | None:
    return svc_get_rotating_log_handler(app, LOG_FILE_PATH)


def _list_system_log_files() -> list[dict[str, str | int | float | bool | None]]:
    files = []
    for entry in svc_list_system_log_files(LOG_DIR, LOG_FILE_PATH):
        files.append(
            {
                'name': entry.get('name'),
                'size': entry.get('size'),
                'modified_at': entry.get('modified_at'),
                'rotated': entry.get('rotated'),
            }
        )
    return files


def _tail_log_file(path: Path, max_lines: int = 200) -> list[str]:
    return svc_tail_log_file(path, max_lines=max_lines)


def _resolve_log_name(name: str | None) -> Path | None:
    return svc_resolve_log_name(LOG_DIR, LOG_FILE_PATH, name)
# ------------------- Утилиты -------------------

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".docx", ".rtf", ".mp3", ".wav", ".m4a", ".flac", ".ogg",
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ------------------- Пользователи и доступ -------------------

DEFAULT_ADMIN_USER = CONFIG.default_admin_user
DEFAULT_ADMIN_PASSWORD = CONFIG.default_admin_password
LEGACY_ACCESS_CODE = CONFIG.legacy_access_code
SESSION_KEY = 'user_id'

_PUBLIC_PREFIXES = ('/static/', '/assets/', '/login-backgrounds/')
_PUBLIC_PATHS = {'/favicon.ico', '/api/auth/login', '/api/login-backgrounds'}

def _log_user_action(user: User | None, action: str, entity: str | None = None, entity_id: int | None = None, detail: str | None = None) -> None:
    try:
        rec = UserActionLog(user_id=user.id if user else None, action=action, entity=entity, entity_id=entity_id, detail=detail)
        db.session.add(rec)
        db.session.commit()
    except Exception:
        db.session.rollback()

def _user_to_payload(user: User) -> dict:
    role = _normalize_user_role(getattr(user, 'role', ROLE_VIEWER), default=ROLE_EDITOR)
    return {
        'id': user.id,
        'username': user.username,
        'role': role,
        'full_name': user.full_name,
        'created_at': user.created_at.isoformat() if user.created_at else None,
        'updated_at': user.updated_at.isoformat() if getattr(user, 'updated_at', None) else None,
        'aiword_access': _has_aiword_access(user),
        'can_upload': _user_can_upload(user),
        'can_import': _user_can_upload(user),
        'permissions': {
            'can_admin': role == ROLE_ADMIN,
            'can_edit': role in (ROLE_ADMIN, ROLE_EDITOR),
            'can_view': True,
        },
    }

def ensure_default_admin() -> None:
    try:
        with app.app_context():
            db.create_all()
            if User.query.count() == 0:
                password = DEFAULT_ADMIN_PASSWORD or LEGACY_ACCESS_CODE or 'admin123'
                user = User(username=DEFAULT_ADMIN_USER, role=ROLE_ADMIN)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                app.logger.warning(
                    "Создан администратор по умолчанию '%s'. Рекомендуется немедленно сменить пароль.",
                    DEFAULT_ADMIN_USER,
                )
    except Exception as exc:
        app.logger.error("Не удалось создать администратора по умолчанию: %s", exc)

def _load_current_user() -> User | None:
    uid = session.get(SESSION_KEY)
    if not uid:
        g.current_user = None
        g.allowed_collection_ids = set()
        return None
    try:
        user = User.query.get(int(uid))
    except Exception:
        user = None
    if not user:
        session.pop(SESSION_KEY, None)
        g.current_user = None
        g.allowed_collection_ids = set()
        return None
    g.current_user = user
    _ensure_personal_collection(user)
    g.allowed_collection_ids = _compute_allowed_collection_ids(user)
    return user


def _compute_allowed_collection_ids(user: User | None):
    if not user:
        return set()
    if _user_role(user) == ROLE_ADMIN:
        return None
    ids: set[int] = set()
    try:
        own = db.session.query(Collection.id).filter(Collection.owner_id == user.id).all()
        ids.update(int(cid) for (cid,) in own)
    except Exception:
        pass
    try:
        membership = db.session.query(CollectionMember.collection_id, CollectionMember.role).filter(CollectionMember.user_id == user.id).all()
        ids.update(int(cid) for (cid, _role) in membership)
    except Exception:
        pass
    try:
        public = db.session.query(Collection.id).filter(Collection.is_private == False).all()
        ids.update(int(cid) for (cid,) in public)
    except Exception:
        pass
    return ids


def _ensure_personal_collection(user: User) -> None:
    try:
        personal = Collection.query.filter_by(owner_id=user.id, is_private=True).first()
        if not personal:
            name = f"Личная коллекция {user.username}"[:120]
            base_slug = _slugify(name)
            if not base_slug:
                base_slug = f"user-{user.id}"
            slug = base_slug
            i = 2
            while Collection.query.filter_by(slug=slug).first() is not None:
                slug = f"{base_slug}-{i}"
                i += 1
            personal = Collection(name=name, slug=slug, owner_id=user.id, is_private=True, searchable=True, graphable=True)
            db.session.add(personal)
            db.session.flush()
        member = CollectionMember.query.filter_by(collection_id=personal.id, user_id=user.id).first()
        if not member:
            db.session.add(CollectionMember(collection_id=personal.id, user_id=user.id, role='owner'))
        db.session.commit()
    except Exception:
        db.session.rollback()


def _has_collection_access(collection_id: int, write: bool = False) -> bool:
    user = _load_current_user()
    if not user:
        return False
    if _user_role(user) == ROLE_ADMIN:
        return True
    allowed = getattr(g, 'allowed_collection_ids', set())
    if allowed is None or collection_id in allowed:
        if not write:
            return True
        try:
            col = Collection.query.get(collection_id)
            if col and col.owner_id == user.id:
                return True
        except Exception:
            pass
        try:
            member = CollectionMember.query.filter_by(collection_id=collection_id, user_id=user.id).first()
            if member and member.role in ('owner', 'editor'):
                return True
        except Exception:
            pass
    return False


def _apply_file_access_filter(query):
    allowed = getattr(g, 'allowed_collection_ids', set())
    if allowed is None:
        return query
    if not allowed:
        return query.filter(File.collection_id == -1)
    return query.filter(File.collection_id.in_(allowed))


def _parse_collection_param(value: str | None, require_write: bool = False) -> int | None:
    if not value:
        return None
    try:
        cid = int(value)
    except Exception:
        abort(400)
    if require_write:
        if not _has_collection_access(cid, write=True):
            abort(403)
    else:
        if not _has_collection_access(cid):
            abort(403)
    return cid


def _aiword_allowed_user_ids() -> set[int]:
    try:
        return {row.user_id for row in AiWordAccess.query.all()}
    except Exception:
        return set()


def _ensure_aiword_access() -> User:
    user = _load_current_user()
    if not user:
        abort(401)
    if _user_role(user) == ROLE_ADMIN:
        return user
    allowed = _aiword_allowed_user_ids()
    if user.id in allowed:
        return user
    abort(403)


def _has_aiword_access(user: User | None) -> bool:
    if not user:
        return False
    if _user_role(user) == ROLE_ADMIN:
        return True
    try:
        return AiWordAccess.query.filter_by(user_id=user.id).first() is not None
    except Exception:
        return False


def _user_can_upload(user: User | None) -> bool:
    if not user:
        return False
    role = _normalize_user_role(getattr(user, 'role', None), default=ROLE_EDITOR)
    if role == ROLE_ADMIN:
        return True
    if role == ROLE_VIEWER:
        return False
    try:
        if Collection.query.filter(Collection.owner_id == user.id).limit(1).first():
            return True
    except Exception:
        pass
    try:
        member = CollectionMember.query.filter(
            CollectionMember.user_id == user.id,
            CollectionMember.role.in_(('owner', 'editor'))
        ).limit(1).first()
        if member:
            return True
    except Exception:
        pass
    return False


def _llm_parse_purposes(raw: str | None) -> list[str]:
    if raw is None:
        return ['default']
    parts = [p.strip().lower() for p in re.split(r'[;,\s]+', raw) if p.strip()]
    return parts or ['default']


def _llm_normalize_purposes(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = _llm_parse_purposes(value)
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            token = str(item).strip().lower()
            if token:
                parts.append(token)
    else:
        return None
    seen: set[str] = set()
    ordered: list[str] = []
    for token in parts:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ','.join(ordered) if ordered else None


def _invalidate_llm_cache() -> None:
    LLM_ROUND_ROBIN.clear()
    LLM_ENDPOINT_SIGNATURE.clear()
    LLM_ENDPOINT_POOLS.clear()
    LLM_ENDPOINT_UNIQUE.clear()


def _ensure_llm_pool(purpose: str | None) -> None:
    desired = (purpose or 'default').lower()
    _ensure_llm_schema_once()
    base_default = (LMSTUDIO_API_BASE or '').rstrip('/')
    model_default = LMSTUDIO_MODEL
    key_default = LMSTUDIO_API_KEY
    try:
        all_eps = LlmEndpoint.query.order_by(LlmEndpoint.created_at.asc()).all()
        candidates: list[tuple[LlmEndpoint, list[str]]] = []
        for ep in all_eps:
            purposes = _llm_parse_purposes(ep.purpose)
            if desired in purposes or (desired != 'default' and 'default' in purposes):
                candidates.append((ep, purposes))
        if not candidates and desired != 'default':
            for ep in all_eps:
                purposes = _llm_parse_purposes(ep.purpose)
                if 'default' in purposes:
                    candidates.append((ep, purposes))
        if candidates:
            sig = tuple(sorted((ep.id, float(ep.weight or 1.0), (ep.provider or ''), tuple(sorted(purposes))) for ep, purposes in candidates))
            if LLM_ENDPOINT_SIGNATURE.get(desired) != sig:
                pool: list[dict[str, str]] = []
                unique: list[dict[str, str]] = []
                seen: set[tuple] = set()
                for ep, _purposes in candidates:
                    provider_fallback = (_rt().lm_default_provider or 'openai')
                    entry = {
                        'id': ep.id,
                        'name': ep.name or f'endpoint-{ep.id}',
                        'base_url': (ep.base_url or base_default).rstrip('/'),
                        'model': ep.model or model_default,
                        'api_key': ep.api_key or key_default,
                        'provider': (ep.provider or provider_fallback).strip().lower() or provider_fallback,
                    }
                    weight = max(1, int(round(float(ep.weight or 1.0))))
                    for _ in range(weight):
                        pool.append(entry)
                    ident = (entry['id'], entry['base_url'], entry['model'], entry.get('api_key'), entry.get('provider'))
                    if ident not in seen:
                        seen.add(ident)
                        unique.append(entry)
                if not pool:
                    default_entry = {
                        'id': None,
                        'name': 'default',
                        'base_url': base_default,
                        'model': model_default,
                        'api_key': key_default,
                        'provider': (_rt().lm_default_provider or 'openai'),
                    }
                    pool = [default_entry]
                    unique = [default_entry]
                LLM_ENDPOINT_POOLS[desired] = pool
                LLM_ENDPOINT_UNIQUE[desired] = unique
                LLM_ROUND_ROBIN[desired] = itertools.cycle(pool)
                LLM_ENDPOINT_SIGNATURE[desired] = sig
            return
    except Exception:
        pass
    if desired not in LLM_ENDPOINT_POOLS:
        default_entry = {
            'id': None,
            'name': 'default',
            'base_url': base_default,
            'model': model_default,
            'api_key': key_default,
            'provider': (_rt().lm_default_provider or 'openai'),
        }
        LLM_ENDPOINT_POOLS[desired] = [default_entry]
        LLM_ENDPOINT_UNIQUE[desired] = [default_entry]
        LLM_ROUND_ROBIN[desired] = itertools.cycle(LLM_ENDPOINT_POOLS[desired])
        LLM_ENDPOINT_SIGNATURE[desired] = tuple()


def _select_llm_endpoint(purpose: str | None = None) -> tuple[str, str, str]:
    desired = (purpose or 'default').lower()
    _ensure_llm_pool(desired)
    pool = LLM_ENDPOINT_POOLS.get(desired) or []
    if not pool:
        return LMSTUDIO_API_BASE, LMSTUDIO_MODEL, LMSTUDIO_API_KEY
    rotation = LLM_ROUND_ROBIN.get(desired)
    if rotation is None:
        rotation = itertools.cycle(pool)
        LLM_ROUND_ROBIN[desired] = rotation
    choice = next(rotation)
    base = choice.get('base_url') or LMSTUDIO_API_BASE
    model = choice.get('model') or LMSTUDIO_MODEL
    key = choice.get('api_key') or LMSTUDIO_API_KEY
    return base, model, key


def _llm_iter_choices(purpose: str | None):
    desired = (purpose or 'default').lower()
    _ensure_llm_pool(desired)
    pool = LLM_ENDPOINT_POOLS.get(desired) or []
    if not pool:
        yield {
            'id': None,
            'name': 'default',
            'base_url': (LMSTUDIO_API_BASE or '').rstrip('/'),
            'model': LMSTUDIO_MODEL,
            'api_key': LMSTUDIO_API_KEY,
            'provider': (_rt().lm_default_provider or 'openai'),
        }
        return
    rotation = LLM_ROUND_ROBIN.get(desired)
    if rotation is None:
        rotation = itertools.cycle(pool)
        LLM_ROUND_ROBIN[desired] = rotation
    unique = LLM_ENDPOINT_UNIQUE.get(desired) or []
    max_iters = len(unique) if unique else len(pool) or 1
    seen: set[tuple] = set()
    for _ in range(max_iters):
        choice = next(rotation)
        ident = (choice.get('id'), choice.get('base_url'), choice.get('model'), choice.get('api_key'), choice.get('provider'))
        if ident in seen:
            continue
        seen.add(ident)
        yield choice


def _llm_choice_label(choice: dict) -> str:
    name = str(choice.get('name') or '').strip()
    base = str(choice.get('base_url') or '').strip()
    model = str(choice.get('model') or '').strip()
    ident = choice.get('id')
    parts = []
    if name:
        parts.append(name)
    if model:
        parts.append(model)
    if base:
        parts.append(base)
    if not parts:
        parts.append(f'id={ident}')
    return ' | '.join(parts)


def _llm_choice_provider(choice: dict) -> str:
    provider = str(choice.get('provider') or '').strip().lower()
    aliases = {
        'azure': 'azure_openai',
        'azure-openai': 'azure_openai',
    }
    provider = aliases.get(provider, provider)
    if provider not in LLM_PROVIDER_CHOICES:
        provider = runtime_settings_store.current.lm_default_provider
    if provider not in LLM_PROVIDER_CHOICES:
        provider = 'openai'
    return provider


def _llm_response_indicates_busy(response) -> bool:
    if response is None:
        return False
    try:
        status_code = int(response.status_code)
    except Exception:
        status_code = None
    if status_code in LLM_BUSY_HTTP_CODES:
        return True
    payload = None
    try:
        payload = response.json() if hasattr(response, 'json') else None
    except ValueError:
        payload = None
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        status = str(payload.get('status') or '').lower()
        message = str(payload.get('message') or '').lower()
        if status in LLM_BUSY_STATUS_VALUES:
            return True
        if any(token in message for token in ('busy', 'processing', 'queue', 'rate')):
            return True
    text = str(getattr(response, 'text', '') or '').lower()
    if any(token in text for token in ('busy', 'processing', 'queue full')):
        return True
    return False


def _llm_choice_url(choice: dict) -> str:
    base = str(choice.get('base_url') or LMSTUDIO_API_BASE or '').strip()
    if not base:
        return ''
    base = base.rstrip('/')
    if not base:
        return ''
    provider = _llm_choice_provider(choice)
    if provider == 'ollama':
        if re.search(r"/api/(chat|generate)$", base):
            return base
        return f"{base}/api/chat"
    if provider == 'azure_openai':
        url = base
        if not re.search(r"/chat/completions(?:\?|$)", url):
            url = f"{url}/chat/completions"
        if 'api-version=' not in url:
            api_version = choice.get('api_version') or AZURE_OPENAI_API_VERSION
            sep = '&' if '?' in url else '?'
            url = f"{url}{sep}api-version={api_version}"
        return url
    if base.endswith('/chat/completions'):
        return base
    if base.endswith('/v1'):
        return f"{base}/chat/completions"
    return f"{base}/chat/completions"


def _llm_choice_headers(choice: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    key = choice.get('api_key')
    if key:
        provider = _llm_choice_provider(choice)
        if provider == 'azure_openai':
            headers["api-key"] = key
        else:
            headers["Authorization"] = f"Bearer {key}"
    return headers


def _llm_timeout_pair(total_timeout: int | float | None) -> tuple[float, float]:
    try:
        base = float(total_timeout) if total_timeout is not None else 120.0
    except Exception:
        base = 120.0
    base = max(5.0, min(base, 300.0))
    connect = min(10.0, max(3.0, base * 0.25))
    read = max(connect + 2.0, base)
    return connect, read


def _llm_send_chat(
    choice: dict,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int | None,
    top_p: float = 1.0,
    timeout: int = 120,
    extra_payload: dict | None = None,
    cache_bucket: str | None = None,
    cache_only: bool | None = None,
):
    provider = _llm_choice_provider(choice)
    url = _llm_choice_url(choice)
    if not url:
        raise ValueError('empty_url')
    model = choice.get('model') or LMSTUDIO_MODEL
    if provider == 'ollama':
        payload = {
            'model': model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': float(temperature),
                'top_p': float(top_p),
            },
        }
        if max_tokens is not None:
            try:
                payload['options']['num_predict'] = max(0, int(max_tokens))
            except Exception:
                pass
        if extra_payload:
            payload.update(extra_payload)
    else:
        payload = {
            'model': model,
            'messages': messages,
            'temperature': float(temperature),
            'top_p': float(top_p),
        }
        if max_tokens is not None:
            try:
                payload['max_tokens'] = max(0, int(max_tokens))
            except Exception:
                pass
        if extra_payload:
            payload.update(extra_payload)
    cache_enabled = bool(LLM_CACHE_ENABLED)
    cache_only = LLM_CACHE_ONLY_MODE if cache_only is None else bool(cache_only)
    cache_key: str | None = None
    if cache_enabled:
        cache_key_material = {
            'bucket': cache_bucket or (choice.get('name') or provider),
            'provider': provider,
            'url': url,
            'model': model,
            'payload': payload,
        }
        cache_key = hashlib.sha256(json.dumps(cache_key_material, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        cached_resp = llm_cache_get(cache_key)
        if cached_resp is not None:
            cached_resp.headers.setdefault('X-LLM-Cache', 'hit')
            return provider, cached_resp
        if cache_only:
            raise RuntimeError('llm_cache_only_mode')

    response = http_request(
        'POST',
        url,
        headers=_llm_choice_headers(choice),
        json=payload,
        timeout=_llm_timeout_pair(timeout),
        logger=app.logger,
    )

    if response is None:
        raise RuntimeError('llm_response_none')

    if response.status_code >= 400:
        try:
            preview = (response.text or '')[:200]
            app.logger.warning(
                "LLM HTTP %s (%s): %s",
                response.status_code,
                _llm_choice_label(choice),
                preview,
            )
        except Exception:
            pass

    if cache_enabled and cache_key and response.status_code < 400:
        try:
            json_payload = response.json()
        except Exception:
            json_payload = None
        cached = CachedLLMResponse(
            status_code=response.status_code,
            data=json_payload,
            text=response.text,
            headers=dict(response.headers or {}),
        )
        cached.headers.setdefault('X-LLM-Cache', 'store')
        llm_cache_set(cache_key, cached)
    return provider, response


def _llm_extract_content(provider: str, data) -> str:
    if not isinstance(data, dict):
        return ''
    if provider == 'ollama':
        message = data.get('message')
        if isinstance(message, dict):
            content = message.get('content')
            if content:
                return str(content)
        for key in ('response', 'content', 'text'):
            value = data.get(key)
            if value:
                return str(value)
        return ''
    choices = data.get('choices')
    if isinstance(choices, list) and choices:
        message = choices[0].get('message') if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get('content')
            if content:
                return str(content)
    return ''


def _is_public_path(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
        return True
    if path in _PUBLIC_PATHS:
        return True
    if path == '/' or path.startswith('/access'):
        return True
    if request.method == 'GET' and path.startswith('/app'):
        return True
    return False

@app.before_request
def _access_gate():
    path = request.path or '/'
    user = _load_current_user()
    if _is_public_path(path):
        return None
    if path == '/api/training/problem-pipeline':
        if request.method == 'OPTIONS':
            return None
        allowed, failure_response = _check_pipeline_access()
        if allowed:
            return None
        return _add_pipeline_cors_headers(failure_response)
    if user:
        return None
    if path.startswith('/api/') or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        resp = jsonify({'ok': False, 'error': 'Не авторизовано'})
        resp.status_code = 401
        return _add_pipeline_cors_headers(resp)
    return redirect('/app/login')

@app.route('/api/auth/login', methods=['POST'])
def api_auth_login():
    data = request.get_json(silent=True) or request.form or {}
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    if not username or not password:
        return jsonify({'ok': False, 'error': 'Введите логин и пароль'}), 400
    try:
        user = User.query.filter(func.lower(User.username) == username.lower()).first()
    except Exception:
        user = None
    if not user or not user.check_password(password):
        session.pop(SESSION_KEY, None)
        return jsonify({'ok': False, 'error': 'Неверный логин или пароль'}), 401
    session[SESSION_KEY] = user.id
    session.permanent = True
    g.current_user = user
    _log_user_action(user, 'login')
    return jsonify({'ok': True, 'user': _user_to_payload(user)})

@app.route('/api/auth/logout', methods=['POST'])
def api_auth_logout():
    user = _load_current_user()
    session.pop(SESSION_KEY, None)
    g.current_user = None
    _log_user_action(user, 'logout')
    return jsonify({'ok': True})

@app.route('/api/auth/me')
def api_auth_me():
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False}), 401
    return jsonify({'ok': True, 'user': _user_to_payload(user)})

@app.route('/api/profile', methods=['GET', 'POST'])
def api_profile():
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Не авторизовано'}), 401
    if request.method == 'GET':
        return jsonify({'ok': True, 'user': _user_to_payload(user)})
    data = request.get_json(silent=True) or {}
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''
    user.full_name = (data.get('full_name') or '').strip() or None
    db.session.commit()
    try:
        _log_user_action(user, 'profile_update', 'user', user.id, detail=json.dumps({'full_name': user.full_name}))
    except Exception:
        pass
    return jsonify({'ok': True, 'user': _user_to_payload(user)})

@app.route('/api/auth/password', methods=['POST'])
def api_auth_change_password():
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Не авторизовано'}), 401
    data = request.get_json(silent=True) or {}
    current = data.get('current_password') or ''
    new_password = (data.get('new_password') or '').strip()
    if not user.check_password(current):
        return jsonify({'ok': False, 'error': 'Неверный текущий пароль'}), 400
    if len(new_password) < 6:
        return jsonify({'ok': False, 'error': 'Пароль должен содержать не менее 6 символов'}), 400
    user.set_password(new_password)
    db.session.commit()
    _log_user_action(user, 'change_password')
    return jsonify({'ok': True})

def _require_admin() -> User:
    user = _load_current_user()
    if not user or user.role != 'admin':
        abort(403)
    return user

def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _require_admin()
        return fn(*args, **kwargs)
    return wrapper

admin_bp = Blueprint('admin_api', __name__, url_prefix='/api/admin')
users_bp = Blueprint('users_api', __name__, url_prefix='/api/users')

def _admin_count(exclude_id: int | None = None) -> int:
    q = User.query.filter(User.role == 'admin')
    if exclude_id is not None:
        q = q.filter(User.id != exclude_id)
    try:
        return q.count()
    except Exception:
        return 0

@users_bp.route('/', methods=['GET', 'POST'])
def api_users_collection():
    _require_admin()
    if request.method == 'GET':
        users = User.query.order_by(func.lower(User.username)).all()
        return jsonify({'ok': True, 'users': [_user_to_payload(u) for u in users]})
    data = request.get_json(silent=True) or {}
    username = (data.get('username') or '').strip()
    password = (data.get('password') or '').strip()
    full_name = (data.get('full_name') or '').strip()
    role = _normalize_user_role(data.get('role'), default=ROLE_EDITOR)
    if len(username) < 3:
        return jsonify({'ok': False, 'error': 'Логин должен содержать минимум 3 символа'}), 400
    if len(password) < 6:
        return jsonify({'ok': False, 'error': 'Пароль должен содержать минимум 6 символов'}), 400
    if len(full_name) < 5:
        return jsonify({'ok': False, 'error': 'Укажите ФИО (минимум 5 символов)'}), 400
    if User.query.filter(func.lower(User.username) == username.lower()).first():
        return jsonify({'ok': False, 'error': 'Пользователь с таким логином уже существует'}), 409
    user = User(username=username, role=role, full_name=full_name)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    _log_user_action(
        _load_current_user(),
        'user_create',
        'user',
        user.id,
        detail=json.dumps({'username': username, 'role': role, 'full_name': full_name}, ensure_ascii=False)
    )
    return jsonify({'ok': True, 'user': _user_to_payload(user)}), 201


@users_bp.route('/search')
@require_admin
def api_users_search():
    q = (request.args.get('q') or '').strip()
    try:
        limit = min(max(int(request.args.get('limit', '20')), 1), 50)
    except Exception:
        limit = 20
    qry = User.query
    if q:
        like = f"%{q}%"
        qry = qry.filter(or_(User.username.ilike(like), User.full_name.ilike(like)))
    users = qry.order_by(func.lower(User.username)).limit(limit).all()
    return jsonify({'ok': True, 'users': [{
        'id': u.id,
        'username': u.username,
        'full_name': u.full_name,
        'role': _normalize_user_role(u.role, default=ROLE_EDITOR),
    } for u in users]})


@users_bp.route('/roles', methods=['GET'])
@require_admin
def api_users_roles():
    roles = [
        {'id': ROLE_ADMIN, 'label': 'Администратор'},
        {'id': ROLE_EDITOR, 'label': 'Редактор'},
        {'id': ROLE_VIEWER, 'label': 'Наблюдатель'},
    ]
    return jsonify({'ok': True, 'roles': roles})

@users_bp.route('/<int:user_id>', methods=['PATCH', 'DELETE'])
def api_users_detail(user_id: int):
    admin = _require_admin()
    user = User.query.get_or_404(user_id)
    if request.method == 'DELETE':
        if user.id == admin.id:
            return jsonify({'ok': False, 'error': 'Нельзя удалить собственную учётную запись'}), 400
        if _user_role(user) == ROLE_ADMIN and _admin_count(exclude_id=user.id) == 0:
            return jsonify({'ok': False, 'error': 'Нельзя удалить последнего администратора'}), 400
        db.session.delete(user)
        db.session.commit()
        _log_user_action(admin, 'user_delete', 'user', user_id)
        return jsonify({'ok': True})
    data = request.get_json(silent=True) or {}
    updated = False
    new_role = data.get('role')
    if new_role is not None:
        role = _normalize_user_role(str(new_role), default=_user_role(user))
        current_role = _user_role(user)
        if role != current_role:
            if current_role == ROLE_ADMIN and role != ROLE_ADMIN and _admin_count(exclude_id=user.id) == 0:
                return jsonify({'ok': False, 'error': 'Должен остаться хотя бы один администратор'}), 400
            user.role = role
            updated = True
    new_password = data.get('password')
    if new_password:
        pw = str(new_password).strip()
        if len(pw) < 6:
            return jsonify({'ok': False, 'error': 'Пароль должен содержать минимум 6 символов'}), 400
        user.set_password(pw)
        updated = True
    if not updated:
        return jsonify({'ok': False, 'error': 'Нет изменений'}), 400
    db.session.commit()
    _log_user_action(admin, 'user_update', 'user', user.id, detail=json.dumps({'role': _user_role(user), 'password_changed': bool(new_password)}))
    return jsonify({'ok': True, 'user': _user_to_payload(user)})


@admin_bp.route('/users', methods=['GET', 'POST'])
def api_admin_users():
    return api_users_collection()


@admin_bp.route('/users/search')
@require_admin
def api_admin_users_search():
    return api_users_search()


@admin_bp.route('/users/<int:user_id>', methods=['PATCH', 'DELETE'])
def api_admin_user_detail(user_id: int):
    return api_users_detail(user_id)

# ------------------- Инициализация и миграции коллекций -------------------
def _slugify(s: str) -> str:
    s = (s or '').strip().lower()
    s = re.sub(r"[^a-z0-9_\-а-яё]+", "-", s)
    s = re.sub(r"-+", "-", s).strip('-')
    if not s:
        s = 'collection'
    return s

def ensure_collections_schema():
    """Проверить наличие модели/таблицы коллекций и поля files.collection_id, создать базовую коллекцию.
    Привязать базовую коллекцию к файлам, у которых она отсутствует.
    """
    try:
        with app.app_context():
            # Создать таблицы, если их ещё нет
            db.create_all()
            # Добавить столбец collection_id в files, если его нет
            try:
                from sqlalchemy import text as _text
                with db.engine.begin() as conn:
                    rows = list(conn.execute(_text("PRAGMA table_info(files)")))
                    cols = {r[1] for r in rows}
                    if 'collection_id' not in cols:
                        conn.execute(_text("ALTER TABLE files ADD COLUMN collection_id INTEGER"))
                    rows = list(conn.execute(_text("PRAGMA table_info(collections)")))
                    ccols = {r[1] for r in rows}
                    if 'owner_id' not in ccols:
                        conn.execute(_text("ALTER TABLE collections ADD COLUMN owner_id INTEGER"))
                    if 'is_private' not in ccols:
                        conn.execute(_text("ALTER TABLE collections ADD COLUMN is_private BOOLEAN NOT NULL DEFAULT 0"))
                    rows = list(conn.execute(_text("PRAGMA table_info(users)")))
                    ucols = {r[1] for r in rows}
                    if 'full_name' not in ucols:
                        conn.execute(_text("ALTER TABLE users ADD COLUMN full_name TEXT"))
            except Exception:
                pass
            # Убедиться, что базовая коллекция существует
            base = Collection.query.filter_by(slug='base').first()
            if not base:
                base = Collection(name='Базовая коллекция', slug='base', searchable=True, graphable=True)
                db.session.add(base)
                db.session.commit()
            # Назначить базовую коллекцию всем файлам с NULL в collection_id
            try:
                File.query.filter(File.collection_id.is_(None)).update({File.collection_id: base.id})
                db.session.commit()
            except Exception:
                db.session.rollback()
    except Exception:
        # по возможности продолжаем дальше
        pass


def ensure_llm_schema():
    """Добавить дополнительные поля, используемые LLM-эндпоинтами."""
    try:
        with app.app_context():
            db.create_all()
            try:
                from sqlalchemy import text as _text
                with db.engine.begin() as conn:
                    rows = list(conn.execute(_text("PRAGMA table_info(llm_endpoints)")))
                    cols = {r[1] for r in rows}
                    if 'provider' not in cols:
                        conn.execute(_text("ALTER TABLE llm_endpoints ADD COLUMN provider TEXT DEFAULT 'openai'"))
            except Exception:
                pass
    except Exception:
        pass


def _ensure_llm_schema_once():
    global LLM_SCHEMA_READY
    if LLM_SCHEMA_READY:
        return
    ensure_llm_schema()
    LLM_SCHEMA_READY = True

# ------------------- Вспомогательные функции безопасности путей -------------------
def _resolve_under_base(rel_path: str) -> tuple[Path, Path | None]:
    """Безопасно разрешать rel_path внутри UPLOAD_FOLDER. Возвращает (base_dir, abs_path или None, если путь вне папки)."""
    base_dir = Path(app.config.get('UPLOAD_FOLDER') or '.').resolve()
    try:
        # Нормализовать разделители и убрать ведущие слэши
        rel = str(rel_path).replace('\\', '/').lstrip('/')
        abs_path = (base_dir / rel).resolve()
        abs_path.relative_to(base_dir)
        return base_dir, abs_path
    except Exception:
        return base_dir, None

# ------------------- Routes -------------------

@app.after_request
def _force_utf8(resp):
    try:
        ct = (resp.headers.get('Content-Type') or '').lower()
        if ct.startswith('text/html'):
            resp.headers['Content-Type'] = 'text/html; charset=utf-8'
        elif ct.startswith('application/json'):
            # обеспечить кодировку utf-8 и для JSON
            resp.headers['Content-Type'] = 'application/json; charset=utf-8'
        return resp
    except Exception:
        return resp


@app.route('/health', methods=['GET'])
def health_check():
    checks: dict[str, dict[str, object]] = {}
    overall_status = 'ok'

    # Проверка базы данных
    try:
        db.session.execute(text('SELECT 1'))
        db.session.commit()
        checks['database'] = {'status': 'ok'}
    except Exception as exc:
        db.session.rollback()
        checks['database'] = {'status': 'error', 'detail': str(exc)}
        overall_status = 'degraded'

    # Проверка очереди задач
    queue = get_task_queue()
    queue_stats = queue.stats()
    queue_status = 'ok' if queue_stats.get('started') and not queue_stats.get('shutdown') else 'warning'
    if queue_status != 'ok':
        overall_status = 'degraded'
    checks['task_queue'] = {'status': queue_status, 'stats': queue_stats}

    # Проверка фонового шедулера очистки кэша
    if CACHE_CLEANUP_INTERVAL_HOURS > 0 and not _CLEANUP_THREAD_STARTED:
        checks['cache_cleanup'] = {'status': 'warning', 'detail': 'cleanup thread not started'}
        overall_status = 'degraded'
    else:
        checks['cache_cleanup'] = {'status': 'ok'}

    return jsonify({'status': overall_status, 'checks': checks})


@app.route('/metrics', methods=['GET'])
def metrics():
    lines: list[str] = []
    queue_stats = get_task_queue().stats()
    lines.append(f"task_queue_queued {queue_stats.get('queued', 0)}")
    lines.append(f"task_queue_workers {queue_stats.get('workers', 0)}")
    lines.append(f"task_queue_started {int(bool(queue_stats.get('started')))}")
    lines.append(f"task_queue_shutdown {int(bool(queue_stats.get('shutdown')))}")

    try:
        totals = (
            db.session.query(TaskRecord.status, func.count(TaskRecord.id))
            .group_by(TaskRecord.status)
            .all()
        )
        for status, count in totals:
            metric_name = f"tasks_total_status_{(status or 'unknown').replace('-', '_')}"
            lines.append(f"{metric_name} {count}")
        db_status = '1'
    except Exception:
        db.session.rollback()
        db_status = '0'
    lines.append(f"database_available {db_status}")

    payload = "\n".join(lines) + "\n"
    return Response(payload, mimetype='text/plain; version=0.0.4; charset=utf-8')


# Загрузка файлов через веб-интерфейс
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    user = _load_current_user()
    if not user:
        abort(401)
    is_admin = _user_role(user) == ROLE_ADMIN
    if request.method == "GET":
        allowed = getattr(g, 'allowed_collection_ids', set())
        cols_q = Collection.query.order_by(Collection.name.asc())
        if allowed is not None:
            if not allowed:
                cols_q = cols_q.filter(Collection.id == -1)
            else:
                cols_q = cols_q.filter(Collection.id.in_(allowed))
        cols = cols_q.all()
        return jsonify({
            "allowed_extensions": sorted(ALLOWED_EXTS),
            "collections": [{"id": c.id, "name": c.name} for c in cols],
        })

    file = request.files.get("file")
    if not file or not file.filename:
        return json_error("Файл не выбран", 400)

    try:
        collection = _resolve_upload_collection(user, request.form, is_admin)
    except PermissionError:
        abort(403)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        return json_error(f"Не удалось подготовить коллекцию: {exc}", 500)

    try:
        save_result = _save_upload_to_disk(file)
    except ValueError as exc:
        return json_error(str(exc), 415)
    except Exception as exc:
        return json_error(f"Не удалось сохранить файл: {exc}", 500)

    try:
        import_result = _import_file_record(
            save_result['path'],
            save_result['rel_path'],
            collection,
            user_id=user.id if user else None,
            generate_metadata=True,
        )
    except Exception as exc:
        return json_error(f"Ошибка обработки файла: {exc}", 500)

    try:
        _log_user_action(
            _load_current_user(),
            'file_upload',
            'file',
            import_result.get('file_id'),
            detail=json.dumps({'filename': import_result.get('preview', {}).get('filename'), 'collection_id': getattr(collection, 'id', None)}),
        )
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "file_id": import_result.get('file_id'),
        "rel_path": import_result.get('rel_path'),
        "preview": import_result.get('preview'),
    })


@app.route('/api/import/jobs', methods=['POST'])
def api_import_jobs_create():
    user = _load_current_user()
    if not user:
        abort(401)
    if not _user_can_upload(user):
        abort(403)

    file = request.files.get('file')
    if not file or not file.filename:
        return json_error('Файл не выбран', 400)

    is_admin = _user_role(user) == ROLE_ADMIN
    try:
        collection = _resolve_upload_collection(user, request.form, is_admin)
    except PermissionError:
        abort(403)
    except ValueError as exc:
        return json_error(str(exc), 400)
    except Exception as exc:
        return json_error(f"Не удалось подготовить коллекцию: {exc}", 500)

    try:
        save_result = _save_upload_to_disk(file)
    except ValueError as exc:
        return json_error(str(exc), 415)
    except Exception as exc:
        return json_error(f"Не удалось сохранить файл: {exc}", 500)

    initial_preview = _build_initial_preview(save_result, collection)
    payload = {
        'kind': 'import_file',
        'submitted_by': user.id,
        'collection_id': getattr(collection, 'id', None),
        'filename': save_result.get('filename'),
        'rel_path': save_result.get('rel_path'),
        'initial_preview': initial_preview,
    }
    try:
        task = TaskRecord(name='import_file', status='queued', payload=json.dumps(payload, ensure_ascii=False), progress=0.0)
        db.session.add(task)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return json_error(f"Не удалось создать задачу импорта: {exc}", 500)

    spec = {
        'path': str(save_result['path']),
        'rel_path': save_result['rel_path'],
        'collection_id': getattr(collection, 'id', None),
        'user_id': user.id,
        'filename': save_result.get('filename'),
    }
    _enqueue_import_job(task.id, spec, description=save_result.get('filename'))
    return jsonify({'ok': True, 'task_id': task.id, 'initial_preview': initial_preview})


def _task_payload_json(task: TaskRecord) -> dict:
    if not task.payload:
        return {}
    try:
        return json.loads(task.payload)
    except Exception:
        return {}


@app.route('/api/import/jobs', methods=['GET'])
def api_import_jobs_list():
    user = _load_current_user()
    if not user:
        abort(401)
    role = _user_role(user)
    tasks_query = TaskRecord.query.filter(TaskRecord.name == 'import_file').order_by(TaskRecord.created_at.desc())
    tasks = tasks_query.limit(50 if role == ROLE_ADMIN else 20).all()
    items = []
    for task in tasks:
        payload = _task_payload_json(task)
        if role != ROLE_ADMIN and payload.get('submitted_by') not in (user.id, None):
            continue
        info = _task_to_dict(task)
        info['payload_json'] = payload
        items.append(info)
    return jsonify({'ok': True, 'tasks': items})


@app.route('/api/import/jobs/<int:task_id>', methods=['GET'])
def api_import_jobs_status(task_id: int):
    user = _load_current_user()
    if not user:
        abort(401)
    task = TaskRecord.query.get_or_404(task_id)
    payload = _task_payload_json(task)
    if _user_role(user) != ROLE_ADMIN and payload.get('submitted_by') != user.id:
        abort(403)
    info = _task_to_dict(task)
    info['payload_json'] = payload
    return jsonify({'ok': True, 'task': info})


def _resolve_upload_collection(user: User | None, form, is_admin: bool) -> Collection | None:
    allowed = getattr(g, 'allowed_collection_ids', set())
    col: Collection | None = None
    new_col = (form.get('new_collection') or '').strip()
    col_id = form.get('collection_id')

    try:
        if new_col:
            slug = _slugify(new_col)
            col = Collection.query.filter((Collection.slug == slug) | (Collection.name == new_col)).first()
            if not col:
                requested_private = str(form.get('private', '')).lower() in ('1', 'true', 'yes', 'on')
                is_private_flag = requested_private or not is_admin
                col = Collection(
                    name=new_col,
                    slug=slug,
                    searchable=True,
                    graphable=True,
                    owner_id=user.id if user else None,
                    is_private=is_private_flag,
                )
                db.session.add(col)
                db.session.commit()
                if user:
                    try:
                        if not CollectionMember.query.filter_by(collection_id=col.id, user_id=user.id).first():
                            db.session.add(CollectionMember(collection_id=col.id, user_id=user.id, role='owner'))
                            db.session.commit()
                    except Exception:
                        db.session.rollback()
                if allowed is not None:
                    try:
                        allowed = set(allowed)
                        allowed.add(col.id)
                        g.allowed_collection_ids = allowed
                    except Exception:
                        pass
        elif col_id:
            try:
                col = Collection.query.get(int(col_id))
            except Exception:
                col = None
    except Exception:
        db.session.rollback()
        raise

    if not col and user:
        try:
            col = Collection.query.filter_by(owner_id=user.id, is_private=True).first()
        except Exception:
            col = None
    if not col:
        col = Collection.query.filter_by(slug='base').first() or Collection.query.first()

    if col:
        if allowed is not None and col.id not in allowed:
            if not _has_collection_access(col.id, write=True):
                raise PermissionError('Недостаточно прав для загрузки в коллекцию')
        else:
            if not _has_collection_access(col.id, write=True):
                raise PermissionError('Недостаточно прав для загрузки в коллекцию')
    return col


def _save_upload_to_disk(file_storage) -> dict:
    safe_name = secure_filename(file_storage.filename or '')
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise ValueError(f"Недопустимый тип файла: {ext}")

    scan_root = _scan_root_path()
    subdir = _import_subdir_value()
    base_dir = scan_root / subdir if subdir else scan_root
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / safe_name
    stem = Path(safe_name).stem
    counter = 1
    while target.exists():
        target = base_dir / f"{stem}_{counter}{ext}"
        counter += 1
    file_storage.save(target)

    try:
        rel_path = str(target.relative_to(scan_root))
    except Exception:
        rel_path = target.name

    stat = target.stat()
    return {
        'path': target,
        'rel_path': rel_path,
        'ext': ext,
        'size': stat.st_size,
        'filename': target.name,
    }


def _build_initial_preview(save_result: dict, collection: Collection | None) -> dict:
    return {
        'filename': save_result.get('filename'),
        'ext': save_result.get('ext'),
        'size': int(save_result.get('size', 0) or 0),
        'collection_id': getattr(collection, 'id', None),
        'collection_name': getattr(collection, 'name', None) if collection else None,
    }


def _guess_language(text: str) -> str | None:
    if not text:
        return None
    cyr = sum(1 for ch in text if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё')
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    total = cyr + lat
    if total < 20:
        return None
    return 'ru' if cyr >= lat else 'en'


def _extract_text_sample(path: Path, ext: str, limit: int = 6000) -> str:
    ext = ext.lower()
    try:
        if ext in {'.txt', '.md'}:
            return path.read_text(encoding='utf-8', errors='ignore')[:limit]
        if ext == '.pdf':
            return extract_text_pdf(path, limit_chars=limit)
        if ext == '.docx':
            return extract_text_docx(path, limit_chars=limit)
        if ext == '.rtf':
            return extract_text_rtf(path, limit_chars=limit)
        if ext == '.epub':
            return extract_text_epub(path, limit_chars=limit)
        if ext == '.djvu':
            return extract_text_djvu(path, limit_chars=limit)
    except Exception:
        return ''
    return ''


def _import_file_record(
    save_path: Path,
    rel_path: str,
    collection: Collection | None,
    user_id: int | None = None,
    *,
    generate_metadata: bool = True,
) -> dict:
    ext = save_path.suffix.lower()
    size = save_path.stat().st_size
    mtime = save_path.stat().st_mtime
    try:
        sha1 = sha1_of_file(save_path)
    except Exception:
        sha1 = None

    fobj = File.query.filter_by(path=str(save_path)).first()
    if not fobj:
        fobj = File(path=str(save_path))
        db.session.add(fobj)

    fobj.rel_path = rel_path
    fobj.filename = save_path.stem
    fobj.ext = ext
    fobj.size = size
    fobj.mtime = mtime
    fobj.sha1 = sha1
    if collection:
        fobj.collection_id = collection.id

    try:
        db.session.flush()
    except Exception as exc:
        db.session.rollback()
        raise exc

    text_sample = _extract_text_sample(save_path, ext)
    if text_sample:
        fobj.text_excerpt = text_sample[:4000]

    metadata_summary: dict[str, object] | None = None
    if generate_metadata:
        metadata = None
        sample_for_llm = text_sample or ''
        if not sample_for_llm and ext in AUDIO_EXTS:
            sample_for_llm = ''
        if sample_for_llm:
            try:
                metadata = call_lmstudio_for_metadata(sample_for_llm[:15000], save_path.name)
            except Exception as exc:
                logger.debug('LLM metadata generation failed for %s: %s', save_path.name, exc)
        if isinstance(metadata, dict) and metadata:
            metadata_summary = metadata
            mt_meta = normalize_material_type(metadata.get("material_type"))
            if ext in AUDIO_EXTS:
                fobj.material_type = 'audio'
            elif ext in IMAGE_EXTS:
                fobj.material_type = 'image'
            elif TYPE_LLM_OVERRIDE and mt_meta:
                fobj.material_type = mt_meta
            title_meta = (metadata.get('title') or '').strip()
            if title_meta:
                fobj.title = title_meta
            author_meta = _normalize_author(metadata.get('author'))
            if author_meta:
                fobj.author = author_meta
            year_meta = _normalize_year(metadata.get('year'))
            if year_meta:
                fobj.year = year_meta
            advisor = metadata.get('advisor')
            if advisor:
                advisor_str = str(advisor).strip()
                if advisor_str:
                    fobj.advisor = advisor_str
            kws = metadata.get('keywords') or []
            if isinstance(kws, list) and kws:
                fobj.keywords = ", ".join([str(x) for x in kws][:50])
            try:
                db.session.flush()
                if metadata.get('novelty'):
                    upsert_tag(fobj, 'научная новизна', str(metadata['novelty']))
                for extra_key in ('literature', 'organizations', 'classification'):
                    extra_val = metadata.get(extra_key)
                    if isinstance(extra_val, list) and extra_val:
                        upsert_tag(fobj, extra_key, "; ".join([str(x) for x in extra_val]))
            except Exception:
                db.session.rollback()
                db.session.add(fobj)

    try:
        _sync_file_to_fts(fobj)
    except Exception as exc:
        logger.debug('FTS sync failed for file %s: %s', getattr(fobj, 'id', None), exc)
    _invalidate_facets_cache('import')
    db.session.commit()

    language = _guess_language(fobj.text_excerpt or '') if getattr(fobj, 'text_excerpt', None) else None

    preview = {
        'filename': save_path.name,
        'ext': ext,
        'size': size,
        'collection_id': getattr(collection, 'id', None),
        'collection_name': getattr(collection, 'name', None) if collection else None,
        'title': fobj.title,
        'author': fobj.author,
        'material_type': fobj.material_type,
        'language': language,
        'text_excerpt': (fobj.text_excerpt or '')[:400],
    }
    if metadata_summary:
        preview['metadata'] = metadata_summary

    return {
        'file_id': fobj.id,
        'rel_path': rel_path,
        'preview': preview,
    }


def _enqueue_import_job(task_id: int, spec: dict, description: str | None = None) -> None:
    @copy_current_app_context
    def _runner():
        _run_import_job(task_id, spec)

    get_task_queue().submit(_runner, description=description or f"import-{spec.get('filename', '')}")


def _run_import_job(task_id: int, spec: dict) -> None:
    task = TaskRecord.query.get(task_id)
    if not task:
        return
    try:
        task.status = 'running'
        task.started_at = datetime.utcnow()
        task.progress = 0.05
        db.session.commit()
    except Exception:
        db.session.rollback()

    path = Path(spec.get('path'))
    rel_path = spec.get('rel_path')
    collection = _get_collection_instance(spec.get('collection_id'))
    user_id = spec.get('user_id')

    try:
        if not path.exists():
            raise FileNotFoundError(f"Файл {path} недоступен")
        info = _import_file_record(path, rel_path, collection, user_id=user_id, generate_metadata=True)
        payload = {
            'kind': 'import_file',
            'submitted_by': user_id,
            'collection_id': getattr(collection, 'id', None),
            'filename': spec.get('filename'),
            'rel_path': rel_path,
            'result': info,
        }
        task = TaskRecord.query.get(task_id)
        if task:
            task.status = 'completed'
            task.progress = 1.0
            task.finished_at = datetime.utcnow()
            task.payload = json.dumps(payload, ensure_ascii=False)
            db.session.commit()
        if user_id:
            try:
                actor = User.query.get(user_id)
                if actor:
                    _log_user_action(actor, 'file_import_async', 'file', info.get('file_id'), detail=json.dumps({'filename': spec.get('filename'), 'collection_id': getattr(collection, 'id', None)}))
            except Exception:
                pass
    except Exception as exc:
        db.session.rollback()
        task = TaskRecord.query.get(task_id)
        if task:
            task.status = 'error'
            task.error = str(exc)
            task.finished_at = datetime.utcnow()
            payload = {
                'kind': 'import_file',
                'submitted_by': user_id,
                'collection_id': getattr(collection, 'id', None),
                'filename': spec.get('filename'),
                'rel_path': rel_path,
                'error': str(exc),
            }
            try:
                task.payload = json.dumps(payload, ensure_ascii=False)
            except Exception:
                task.payload = None
            db.session.commit()
        app.logger.warning('Import job %s failed: %s', task_id, exc)

# Утилита: безопасная относительная дорожка (для загрузки папок)
def _sanitize_relpath(p: str) -> str:
    p = (p or '').replace('\\', '/').lstrip('/')
    parts = [seg for seg in p.split('/') if seg not in ('', '.', '..')]
    return '/'.join(parts)


def _get_collection_instance(collection_ref) -> Collection | None:
    if isinstance(collection_ref, Collection):
        return collection_ref
    try:
        cid = int(collection_ref)
    except Exception:
        cid = None
    if cid:
        try:
            return Collection.query.get(cid)
        except Exception:
            return None
    return None


def _collection_dir_slug(col: Collection | None) -> str:
    if not col:
        return ''
    slug = (col.slug or '').strip()
    if slug:
        return slug
    name = (col.name or '').strip()
    if name:
        return _slugify(name)
    return f"collection-{col.id or 'unknown'}"


def _collection_root_dir(col: Collection | None, *, ensure: bool = True) -> Path:
    root = _scan_root_path()
    if not _collections_in_separate_dirs() or col is None:
        if ensure:
            root.mkdir(parents=True, exist_ok=True)
        return root
    slug = _collection_dir_slug(col) or f"collection-{col.id or 'unknown'}"
    target = root / 'collections' / slug
    if ensure:
        target.mkdir(parents=True, exist_ok=True)
    return target


def _import_base_dir_for_collection(col: Collection | None) -> Path:
    base_root = _collection_root_dir(col)
    sub = _import_subdir_value()
    return (base_root / sub) if sub else base_root

@app.route('/import', methods=['GET', 'POST'])
def import_files():
    """Импорт нескольких файлов и папок (через webkitdirectory) с возможным автосканом."""
    user = _load_current_user()
    if not user:
        abort(401)
    is_admin = _user_role(user) == ROLE_ADMIN
    scan_root = _scan_root_path()
    if request.method == 'GET':
        allowed = getattr(g, 'allowed_collection_ids', set())
        q = Collection.query.order_by(Collection.name.asc())
        if allowed is not None:
            if not allowed:
                q = q.filter(Collection.id == -1)
            else:
                q = q.filter(Collection.id.in_(allowed))
        cols = q.all()
        return jsonify({
            "allowed_extensions": sorted(ALLOWED_EXTS),
            "collections": [{"id": c.id, "name": c.name} for c in cols],
        })

    files = request.files.getlist('files')
    if not files:
        return json_error('Файлы не выбраны', 400)

    col_id = request.form.get('collection_id')
    new_col = (request.form.get('new_collection') or '').strip()
    col: Collection | None = None
    allowed = getattr(g, 'allowed_collection_ids', set())
    try:
        if new_col:
            slug = _slugify(new_col)
            col = Collection.query.filter((Collection.slug == slug) | (Collection.name == new_col)).first()
            if not col:
                requested_private = str(request.form.get('private', '')).lower() in ('1','true','yes','on')
                is_private_flag = requested_private or not is_admin
                col = Collection(name=new_col, slug=slug, searchable=True, graphable=True,
                                 owner_id=user.id, is_private=is_private_flag)
                db.session.add(col)
                db.session.commit()
                if user:
                    try:
                        if not CollectionMember.query.filter_by(collection_id=col.id, user_id=user.id).first():
                            db.session.add(CollectionMember(collection_id=col.id, user_id=user.id, role='owner'))
                            db.session.commit()
                    except Exception:
                        db.session.rollback()
                if allowed is not None:
                    try:
                        allowed = set(allowed)
                        allowed.add(col.id)
                        g.allowed_collection_ids = allowed
                    except Exception:
                        pass
        elif col_id:
            try:
                col = Collection.query.get(int(col_id))
            except Exception:
                col = None
    except Exception as e:
        db.session.rollback()
        return json_error(f"Не удалось создать коллекцию: {e}", 500)
    if not col and user:
        try:
            col = Collection.query.filter_by(owner_id=user.id, is_private=True).first()
        except Exception:
            col = None
    if not col:
        col = Collection.query.filter_by(slug='base').first() or Collection.query.first()
    if col:
        if allowed is not None and col.id not in allowed:
            allowed = None
        if not _has_collection_access(col.id, write=True):
            abort(403)

    base_dir = _import_base_dir_for_collection(col)
    saved = 0
    skipped = 0
    duplicates_found = 0
    saved_rel_paths: list[str] = []
    base_dir.mkdir(parents=True, exist_ok=True)

    for fs in files:
        raw_name = fs.filename or ''
        rel = _sanitize_relpath(raw_name)
        ext = Path(rel).suffix.lower()
        if not rel or ext not in ALLOWED_EXTS:
            skipped += 1
            continue
        dest = base_dir / rel
        if dest.exists():
            skipped += 1
            duplicates_found += 1
            app.logger.info(f'Skip duplicate by path: {dest}')
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fs.save(dest)
        except Exception as e:
            app.logger.warning(f'Failed to save %s: %s', rel, e)
            skipped += 1
            continue

        try:
            try:
                sha1 = sha1_of_file(dest)
            except Exception:
                sha1 = None

            duplicate_rec = None
            if sha1:
                try:
                    q = File.query.filter(File.sha1 == sha1)
                    if col:
                        q = q.filter(File.collection_id == col.id)
                    else:
                        q = q.filter(File.collection_id.is_(None))
                    duplicate_rec = q.first()
                except Exception:
                    duplicate_rec = None
            if duplicate_rec:
                try:
                    existing_path = Path(duplicate_rec.path) if duplicate_rec.path else None
                except Exception:
                    existing_path = None
                try:
                    if existing_path is None or existing_path.resolve() != dest.resolve():
                        try:
                            dest.unlink()
                        except Exception:
                            pass
                        duplicates_found += 1
                        skipped += 1
                        app.logger.info(f'Skip duplicate by hash: {dest}')
                        continue
                except Exception:
                    pass

            try:
                relp = str(dest.relative_to(scan_root))
            except Exception:
                relp = dest.name
            fobj = File.query.filter_by(path=str(dest)).first()
            if not fobj:
                fobj = File(
                    path=str(dest),
                    rel_path=relp,
                    filename=dest.stem,
                    ext=dest.suffix.lower(),
                    size=dest.stat().st_size,
                    mtime=dest.stat().st_mtime,
                    sha1=sha1,
                    collection_id=(col.id if col else None),
                )
                db.session.add(fobj)
            else:
                if col:
                    fobj.collection_id = col.id
            db.session.commit()
            saved_rel_paths.append(relp)
            saved += 1
        except Exception as e:
            db.session.rollback()
            app.logger.warning(f'Failed to register %s in DB: %s', rel, e)
            skipped += 1
            try:
                if dest.exists():
                    dest.unlink()
            except Exception:
                pass

    start_scan = request.form.get('start_scan') == 'on'
    extract_text = request.form.get('extract_text', 'on') == 'on'
    use_llm = request.form.get('use_llm') == 'on'
    prune = request.form.get('prune', 'on') == 'on'
    scan_started = False
    if start_scan and saved_rel_paths:
        if not is_admin:
            try:
                _log_user_action('scan_start_denied', 'collection', col.id if col else None,
                                 detail=json.dumps({'requested': len(saved_rel_paths)}))
            except Exception:
                pass
        else:
            paths_for_scan = [str((scan_root / p).resolve()) for p in saved_rel_paths]
            get_task_queue().submit(
                _run_scan_with_progress,
                extract_text,
                use_llm,
                prune,
                0,
                paths_for_scan,
                description="scan-import-batch",
            )
            scan_started = True

    try:
        _log_user_action(
            _load_current_user(),
            'import_batch',
            'collection',
            col.id if col else None,
            detail=json.dumps({'saved': saved, 'skipped': skipped, 'duplicates': duplicates_found})
        )
    except Exception:
        pass
    return jsonify({
        "status": "ok",
        "saved": saved,
        "skipped": skipped,
        "duplicates": duplicates_found,
        "scan_started": scan_started,
        "imported": saved_rel_paths,
    })

# Просмотр и скачивание файлов
@app.route("/download/<path:rel_path>")
def download_file(rel_path):
    base_dir, abs_path = _resolve_under_base(rel_path)
    # Пытаемся заранее найти запись в БД и подготовить резерв, если файл переместили
    try:
        rp = str(rel_path)
        rp_alt = rp.replace('/', '\\') if ('/' in rp) else rp.replace('\\', '/')
        f = File.query.filter(or_(File.rel_path == rp, File.rel_path == rp_alt)).first()
    except Exception:
        f = None
    if (abs_path is None or not abs_path.exists()) and f is not None:
        try:
            filename_only = Path(f.rel_path or rp).name
            type_dirs = app.config.get('TYPE_DIRS') or {}
            sub = type_dirs.get((f.material_type or '').strip().lower())
            candidates = []
            if sub:
                candidates.append((base_dir / sub / filename_only).resolve())
            for v in set(type_dirs.values()):
                candidates.append((base_dir / v / filename_only).resolve())
            candidates.append((base_dir / filename_only).resolve())
            for cand in candidates:
                try:
                    cand.relative_to(base_dir)
                except Exception:
                    continue
                if cand.exists():
                    abs_path = cand
                    try:
                        f.path = str(cand)
                        f.rel_path = str(cand.relative_to(base_dir))
                        db.session.commit()
                    except Exception:
                        db.session.rollback()
                    # обновляем rel_path для дальнейших URL
                    rel_path = f.rel_path
                    break
            # В крайнем случае ищем по имени файла во всём base_dir (может быть медленно)
            if not (abs_path and abs_path.exists()):
                try:
                    for cand in base_dir.rglob(filename_only):
                        if cand.is_file():
                            abs_path = cand.resolve()
                            try:
                                abs_path.relative_to(base_dir)
                            except Exception:
                                continue
                            try:
                                f.path = str(cand)
                                f.rel_path = str(cand.relative_to(base_dir))
                                db.session.commit()
                            except Exception:
                                db.session.rollback()
                            rel_path = f.rel_path
                            break
                except Exception:
                    pass
        except Exception:
            pass
    if abs_path is None or not abs_path.exists():
        return ("Файл не найден.", 404)
    return send_from_directory(str(base_dir), str(Path(rel_path).as_posix()), as_attachment=True)

@app.route("/media/<path:rel_path>")
def media_file(rel_path):
    """Отдавать файл встраиваемо (для аудиоплеера в предпросмотре), без принудительного скачивания."""
    base_dir, abs_path = _resolve_under_base(rel_path)
    if abs_path is None or not abs_path.exists():
        # Встроенный 404 упрощает работу iframe/плеера
        return ("Not Found", 404)
    return send_from_directory(str(base_dir), str(Path(rel_path).as_posix()), as_attachment=False)

@app.route("/view/<path:rel_path>")
def view_file(rel_path):
    base_dir, abs_path = _resolve_under_base(rel_path)
    if abs_path is None or not abs_path.exists():
        return ("Файл не найден.", 404)
    ext = abs_path.suffix.lower()
    if ext == ".pdf":
        return send_from_directory(str(base_dir), str(Path(rel_path).as_posix()))
    elif ext in {".txt", ".md"}:
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = "Не удалось прочитать файл."
        return Response(content, mimetype="text/plain; charset=utf-8")
    return json_error("Просмотр этого типа файлов не поддерживается", 415)


@app.route('/preview/<path:rel_path>')
def preview_file(rel_path):
    base_dir, abs_path = _resolve_under_base(rel_path)
    if abs_path is None or not abs_path.exists():
        body = (
            "<div style=\"min-height:100vh;display:flex;align-items:center;justify-content:center;"
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;"
            "padding:24px;text-align:center;\">"
            "<div style=\"max-width:400px;background:#161b22;padding:28px;border-radius:18px;"
            "border:1px solid #30363d;box-shadow:0 20px 48px rgba(0,0,0,0.45);\">"
            "Файл не найден или недоступен.</div></div>"
        )
        return _html_page("Файл не найден", body, status=404)

    ext = abs_path.suffix.lower()
    is_pdf = ext == '.pdf'
    is_text = ext in {'.txt', '.md'}
    is_image = ext in IMAGE_EXTS
    is_audio = ext in AUDIO_EXTS
    is_docx = ext == '.docx'
    content = ''
    thumbnail_url = None
    abstract = ''
    audio_url = None
    duration = None
    image_url = None
    docx_html = None

    try:
        # Кэш‑каталог для текстовых фрагментов
        cache_dir = Path(app.static_folder) / 'cache' / 'text_excerpts'
        cache_dir.mkdir(parents=True, exist_ok=True)
        # по возможности используем sha1 в качестве ключа
        sha = None
        try:
            import hashlib
            with abs_path.open('rb') as fh:
                b = fh.read(1024*16)
                sha = hashlib.sha1(b).hexdigest()
        except Exception:
            sha = None

        cache_key = (sha or rel_path.replace('/', '_'))
        cache_file = cache_dir / (cache_key + '.txt')
        if cache_file.exists():
            content = cache_file.read_text(encoding='utf-8', errors='ignore')
        else:
            if is_pdf:
                content = extract_text_pdf(abs_path, limit_chars=4000)[:4000]
            elif ext == '.docx':
                content = extract_text_docx(abs_path, limit_chars=4000)
            elif ext == '.rtf':
                content = extract_text_rtf(abs_path, limit_chars=4000)
            elif ext == '.epub':
                content = extract_text_epub(abs_path, limit_chars=4000)
            elif ext == '.djvu':
                content = extract_text_djvu(abs_path, limit_chars=4000)
            elif is_text:
                content = abs_path.read_text(encoding='utf-8', errors='ignore')[:4000]
            # для аудио позднее предпочтём данные из БД
            try:
                cache_file.write_text(content, encoding='utf-8')
            except Exception:
                pass
        if is_docx and mammoth is not None:
            html_cache_dir = Path(app.static_folder) / 'cache' / 'docx_html'
            html_cache_dir.mkdir(parents=True, exist_ok=True)
            html_cache_file = html_cache_dir / (cache_key + '.html')
            if html_cache_file.exists():
                try:
                    docx_html = html_cache_file.read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    docx_html = None
            if docx_html is None:
                generated_html = _docx_to_html(abs_path)
                if generated_html:
                    docx_html = generated_html
                    try:
                        html_cache_file.write_text(docx_html, encoding='utf-8')
                    except Exception:
                        pass
    except Exception:
        content = ''

    # Пробуем найти запись File, чтобы получить id для ссылки «Подробнее» и ключевые слова
    # Находим запись в БД по rel_path; учитываем различия слэшей/обратных слэшей (Windows)
    rp = str(rel_path)
    rp_alt = rp.replace('/', '\\') if ('/' in rp) else rp.replace('\\', '/')
    f = File.query.filter(or_(File.rel_path == rp, File.rel_path == rp_alt)).first()
    file_id = f.id if f else None
    keywords_str = (f.keywords or '') if f else ''
    if f and (is_audio or (f.material_type or '') == 'audio'):
        is_audio = True
        abstract = (f.abstract or '')
        # Если нет кэша, берём фрагмент транскрипта из БД
        if not content:
            content = (f.text_excerpt or '')[:4000]
        # аудиоплеер ссылается на endpoint скачивания
        # используем rel_path с прямыми слэшами для URL
        audio_url = url_for('media_file', rel_path=str(rel_path).replace('\\','/'))
        try:
            duration = audio_duration_hhmmss(abs_path)
        except Exception:
            duration = None
    if f and (is_image or (f.material_type or '') == 'image'):
        is_image = True
        abstract = (f.abstract or '')
        image_url = url_for('media_file', rel_path=str(rel_path).replace('\\','/'))

    # Сгенерировать миниатюру для PDF (с кэшем)
    if is_pdf:
        try:
            thumb_path = Path(app.static_folder) / 'thumbnails' / (Path(rel_path).stem + '.png')
            if not thumb_path.exists():
                try:
                    import fitz
                    doc = fitz.open(str(abs_path))
                    pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1))
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    pix.save(str(thumb_path))
                except Exception as e:
                    app.logger.warning(f"Thumbnail generation failed: {e}")
            if thumb_path.exists():
                thumbnail_url = url_for('static', filename=f'thumbnails/{thumb_path.name}')
        except Exception:
            thumbnail_url = None

    rel_url = str(rel_path).replace('\\','/')
    embedded = str(request.args.get('embedded', '')).strip().lower() in ('1', 'true', 'yes', 'on')
    return _render_preview(
        rel_url,
        is_pdf=is_pdf,
        is_text=is_text,
        is_audio=is_audio,
        is_image=is_image,
        is_docx=is_docx,
        content=content,
        thumbnail_url=thumbnail_url,
        abstract=abstract,
        audio_url=audio_url,
        duration=duration,
        image_url=image_url,
        keywords=keywords_str,
        embedded=embedded,
        docx_html=docx_html,
    )

FILENAME_PATTERNS = [
    # "Author - Title (2021).pdf"
    re.compile(r"^(?P<author>.+?)\s*-\s*(?P<title>.+?)\s*\((?P<year>\d{4})\)$"),
    # "Author_Title_2020.pdf" или "Author Title 2020.pdf"
    re.compile(r"^(?P<author>.+?)[_ ]+(?P<title>.+?)[_ ]+(?P<year>\d{4})$"),
]

def sha1_of_file(fp: Path, chunk=1<<20):
    h = hashlib.sha1()
    with fp.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def extract_text_pdf(fp: Path, limit_chars=40000, force_ocr_first_page: bool = False):
    """Извлечение текста из PDF.
    Fallback: если текста крайне мало — OCR до первых 5 страниц (если установлен pytesseract).
    Также логируем включение/отсутствие OCR и затраченное время в прогресс-лог (если он активен).
    """
    try:
        import time as _time
        doc = fitz.open(fp)
        text_parts = []
        max_ocr_pages = int(os.getenv('PDF_OCR_PAGES', '5'))
        used_ocr_pages = 0
        ocr_time_total = 0.0
        # Попробуем первые N страниц улучшить OCR-ом, если обычный текст слишком скуден
        for idx, page in enumerate(doc):
            raw = page.get_text("text") or ""
            # Для диссертаций может потребоваться OCR первой страницы независимо от настроек
            if idx == 0 and force_ocr_first_page and pytesseract is not None:
                try:
                    t0 = _time.time()
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                        pix.save(tf.name)
                        ocr = pytesseract.image_to_string(tf.name, lang=os.getenv('OCR_LANGS', 'rus+eng'))
                        if (ocr or '').strip():
                            # если есть оригинальный текст, добавляем OCR в начало; иначе заменяем
                            if len((raw or '').strip()) < 200:
                                raw = (ocr or '')
                            else:
                                raw = (ocr or '') + "\n" + raw
                            used_ocr_pages += 1
                    ocr_time_total += (_time.time() - t0)
                except Exception as oe:
                    app.logger.info(f"OCR(first page) failed for {fp}: {oe}")
            # Эвристический OCR первых N страниц, когда исходный текст слишком короткий
            if idx < max_ocr_pages and pytesseract is not None and len((raw or '').strip()) < 30:
                try:
                    t0 = _time.time()
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                        pix.save(tf.name)
                        ocr = pytesseract.image_to_string(tf.name, lang=os.getenv('OCR_LANGS', 'rus+eng'))
                        if (ocr or '').strip():
                            raw = (ocr or '')
                            used_ocr_pages += 1
                    ocr_time_total += (_time.time() - t0)
                except Exception as oe:
                    app.logger.info(f"OCR failed for page {idx} {fp}: {oe}")
            text_parts.append(raw)
            if sum(len(x) for x in text_parts) >= limit_chars:
                break
        text = "\n".join(text_parts)
        # Эвристика: выявляем «кашу» в тексте (защита от копирования/битые глифы) и переходим к OCR
        try:
            def _ratio_latin_cyr(s: str) -> float:
                if not s:
                    return 0.0
                good = sum(1 for ch in s if ('a' <= ch.lower() <= 'z') or ('а' <= ch.lower() <= 'я') or (ch in 'ёЁ'))
                return good / max(1, len(s))
            # очень мало латинских/кириллических букв и много символов
            rat = _ratio_latin_cyr(text)
            if len(text) >= 400 and rat < 0.15 and pytesseract is not None:
                tstart = _time.time()
                text_ocr = []
                pages_to_ocr = min(len(doc), int(os.getenv('PDF_OCR_PAGES', '5')))
                for idx in range(pages_to_ocr):
                    page = doc[idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                        pix.save(tf.name)
                        ocr = pytesseract.image_to_string(tf.name, lang=os.getenv('OCR_LANGS', 'rus+eng'))
                        if ocr:
                            text_ocr.append(ocr)
                if text_ocr:
                    text = ("\n".join(text_ocr) or text)[:limit_chars]
                    used_ocr_pages = max(used_ocr_pages, pages_to_ocr)
                    ocr_time_total += (_time.time() - tstart)
                    try:
                        _scan_log("OCR: заменил искажённый текст извлечённым OCR")
                    except Exception:
                        pass
        except Exception:
            pass
        # Резервный проход: если после обработки текст остаётся коротким — выполняем OCR до 5 страниц без условия длины
        if len(text.strip()) < 200 and pytesseract is not None:
            try:
                tstart = _time.time()
                text_ocr = []
                pages_to_ocr = min(len(doc), int(os.getenv('PDF_OCR_PAGES', '5')))
                for idx in range(pages_to_ocr):
                    page = doc[idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tf:
                        pix.save(tf.name)
                        ocr = pytesseract.image_to_string(tf.name, lang=os.getenv('OCR_LANGS', 'rus+eng'))
                        if ocr:
                            text_ocr.append(ocr)
                text = ("\n".join(text_ocr) or text)[:limit_chars]
                used_ocr_pages = max(used_ocr_pages, pages_to_ocr)
                ocr_time_total += (_time.time() - tstart)
            except Exception as oe:
                app.logger.info(f"OCR fallback failed {fp}: {oe}")

        # Логирование в прогресс (если доступно)
        try:
            if used_ocr_pages > 0:
                _scan_log(f"OCR: использовано страниц {used_ocr_pages}, время {int(ocr_time_total*1000)} мс")
            elif pytesseract is None:
                _scan_log("OCR недоступен (pytesseract не установлен)")
        except Exception:
            pass
        return text[:limit_chars]
    except Exception as e:
        app.logger.warning(f"PDF extract failed for {fp}: {e}")
        return ""

def extract_text_docx(fp: Path, limit_chars=40000):
    if not docx:
        return ""
    try:
        d = docx.Document(str(fp))
        text = "\n".join([p.text for p in d.paragraphs])
        return text[:limit_chars]
    except Exception as e:
        app.logger.warning(f"DOCX extract failed for {fp}: {e}")
        return ""

def extract_text_rtf(fp: Path, limit_chars=40000):
    if not rtf_to_text:
        return ""
    try:
        text = rtf_to_text(fp.read_text(encoding="utf-8", errors="ignore"))
        return text[:limit_chars]
    except Exception as e:
        app.logger.warning(f"RTF extract failed for {fp}: {e}")
        return ""

def extract_text_epub(fp: Path, limit_chars=40000):
    if not epub:
        return ""
    try:
        book = epub.read_epub(str(fp))
        text = ""
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                text += item.get_content().decode(errors="ignore")
                if len(text) >= limit_chars:
                    break
        return text[:limit_chars]
    except Exception as e:
        app.logger.warning(f"EPUB extract failed for {fp}: {e}")
        return ""

def extract_text_djvu(fp: Path, limit_chars=40000):
    if not djvu:
        return ""
    try:
        with djvu.decode.open(str(fp)) as d:
            text = ""
            for page in d.pages:
                text += page.get_text()
                if len(text) >= limit_chars:
                    break
        return text[:limit_chars]
    except Exception as e:
        app.logger.warning(f"DjVu extract failed for {fp}: {e}")
        return ""

def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None

def _convert_to_wav_pcm16(src: Path, dst: Path, rate=16000):
    if not _ffmpeg_available():
        raise RuntimeError("ffmpeg not found for audio conversion")
    subprocess.run(['ffmpeg','-y','-i',str(src),'-ac','1','-ar',str(rate),'-f','wav',str(dst)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

def _ffprobe_duration_seconds(src: Path) -> float:
    if shutil.which('ffprobe') is None:
        return 0.0
    try:
        out = subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(src)], stderr=subprocess.DEVNULL)
        return float(out.strip())
    except Exception:
        return 0.0

def audio_duration_hhmmss(src: Path) -> str:
    """Return duration string HH:MM:SS for common audio; best-effort using wave/ffprobe."""
    try:
        if src.suffix.lower() == '.wav':
            with wave.open(str(src), 'rb') as wf:
                frames = wf.getnframes(); rate = wf.getframerate() or 1
                secs = frames / float(rate)
        else:
            secs = _ffprobe_duration_seconds(src)
    except Exception:
        secs = 0.0
    secs = int(round(secs))
    h = secs // 3600; m = (secs % 3600) // 60; s = secs % 60
    return (f"{h:02d}:{m:02d}:{s:02d}") if h else (f"{m:02d}:{s:02d}")

def transcribe_audio(fp: Path, limit_chars=40000,
                     backend_override: str | None = None,
                     model_path_override: str | None = None,
                     lang_override: str | None = None,
                     vad_override: bool | None = None) -> str:
    # Позволяем диагностике работать даже при выключенном глобальном флаге, когда заданы overrides
    if not TRANSCRIBE_ENABLED and backend_override is None:
        return ""
    backend = (backend_override or TRANSCRIBE_BACKEND or '').lower()
    model_path = (model_path_override if model_path_override is not None else TRANSCRIBE_MODEL_PATH) or ''
    lang = (lang_override if (lang_override is not None) else TRANSCRIBE_LANGUAGE) or 'ru'
    try:
        if backend == 'faster-whisper' and FasterWhisperModel:
            # Определить путь к модели (каталог / алиас / repo id)
            resolved = ''
            if model_path:
                if Path(model_path).expanduser().exists():
                    resolved = str(Path(model_path).expanduser())
                else:
                    resolved = _ensure_faster_whisper_model(model_path)
            # Запасной вариант: используем small, если путь не определился
            if not resolved:
                resolved = _ensure_faster_whisper_model(os.getenv('FASTER_WHISPER_DEFAULT_MODEL', 'small'))
            if not resolved:
                app.logger.warning("Не удалось определить путь к модели faster-whisper — укажите TRANSCRIBE_MODEL_PATH или установите huggingface_hub")
                return ""
            model = FasterWhisperModel(resolved, device="cpu", compute_type="int8")

            def _fw_try(vad=True, lng=lang):
                try:
                    segs, info = model.transcribe(str(fp), language=lng, vad_filter=vad)
                    txt = " ".join(((s.text or '').strip()) for s in segs)
                    return (txt or '').strip()
                except Exception as _e:
                    app.logger.info(f"Попытка распознавания faster-whisper не удалась (vad={vad}, lang={lng}): {_e}")
                    return ''

            # Последовательность попыток (или принудительно заданный режим VAD)
            if vad_override is True or vad_override is False:
                text = _fw_try(vad=bool(vad_override), lng=lang)
                if not text:
                    text = _fw_try(vad=bool(vad_override), lng=None)
            else:
                text = _fw_try(vad=True, lng=lang)
                if not text:
                    text = _fw_try(vad=False, lng=lang)
                if not text:
                    text = _fw_try(vad=False, lng=None)
            return (text or '')[:limit_chars]
    except Exception as e:
        app.logger.warning(f"Транскрибация не удалась для {fp}: {e}")
    return ""

def call_lmstudio_summarize(text: str, filename: str) -> str:
    text = (text or "")[: int(os.getenv("SUMMARY_TEXT_LIMIT", "12000"))]
    text = text[:_lm_max_input_chars()]
    if not text:
        return ""
    system = PROMPTS.get('summarize_audio_system') or DEFAULT_PROMPTS.get('summarize_audio_system', '')
    user = f"Файл: {filename}\nСтенограмма:\n{text}"
    last_error: Exception | None = None
    for choice in _llm_iter_choices('summary'):
        label = _llm_choice_label(choice)
        provider = _llm_choice_provider(choice)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            provider_name = provider  # сохраняем имя провайдера для логов
            _provider, response = _llm_send_chat(
                choice,
                messages,
                temperature=0.2,
                max_tokens=min(400, _lm_max_output_tokens()),
                timeout=120,
                cache_bucket='summary',
            )
            if _llm_response_indicates_busy(response):
                app.logger.info(f"LLM summary endpoint занята ({label}), переключаемся")
                last_error = RuntimeError('busy')
                continue
            response.raise_for_status()
            data = response.json()
            content = _llm_extract_content(_provider, data)
            if content:
                return content
            try:
                preview = (response.text or '')[:200]
                app.logger.warning(
                    "LLM summary empty content (%s, provider=%s): %s",
                    label,
                    provider_name,
                    preview,
                )
            except Exception:
                pass
        except ValueError as ve:
            last_error = ve
            app.logger.warning(f"Суммаризация не удалась ({label}): {ve}")
            continue
        except Exception as e:
            last_error = e
            if isinstance(e, RuntimeError) and str(e) == 'llm_cache_only_mode':
                app.logger.info(f"Суммаризация пропущена (режим cache-only, {label})")
            else:
                app.logger.warning(f"Суммаризация не удалась ({label}, {provider_name}): {e}")
            continue
    if last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
        app.logger.warning(f"Суммаризация не удалась: {last_error}")
    return ""

def call_lmstudio_compose(system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 400) -> str:
    last_error: Exception | None = None
    for choice in _llm_iter_choices('compose'):
        label = _llm_choice_label(choice)
        provider = _llm_choice_provider(choice)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            provider_name = provider
            _provider, response = _llm_send_chat(
                choice,
                messages,
                temperature=float(temperature),
                max_tokens=min(int(max_tokens), _lm_max_output_tokens()),
                timeout=120,
                cache_bucket='compose',
            )
            if _llm_response_indicates_busy(response):
                app.logger.info(f"LLM compose endpoint занята ({label}), переключаемся")
                last_error = RuntimeError('busy')
                continue
            response.raise_for_status()
            data = response.json()
            content = _llm_extract_content(_provider, data)
            if content:
                return content
        except ValueError as ve:
            last_error = ve
            app.logger.warning(f"Compose LLM endpoint некорректен ({label}): {ve}")
            continue
        except Exception as e:
            last_error = e
            if isinstance(e, RuntimeError) and str(e) == 'llm_cache_only_mode':
                app.logger.info(f"Compose-запрос пропущен (режим cache-only, {label})")
            else:
                app.logger.warning(f"LM compose не удался ({label}, {provider_name}): {e}")
    if last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
        app.logger.warning(f"LLM compose failed: {last_error}")
    return ""
def call_lmstudio_keywords(text: str, filename: str):
    """Извлечь короткий список ключевых слов из текста или поискового запроса через LM Studio."""
    text = (text or "").strip()
    if not text:
        return []
    text = text[:int(os.getenv("KWS_TEXT_LIMIT", "8000"))]
    text = text[:_lm_max_input_chars()]
    is_query = str(filename or '').strip().lower() == 'ai-search'
    if is_query:
        system = PROMPTS.get('ai_search_keywords_system') or DEFAULT_PROMPTS.get('ai_search_keywords_system', '')
        user = (
            "Поисковый запрос пользователя:\n"
            f"{text}\n\n"
            "Верни JSON-массив вида [\"тег1\", \"тег2\", ...] без пояснений."
        )
    else:
        system = PROMPTS.get('keywords_system') or DEFAULT_PROMPTS.get('keywords_system', '')
        user = f"Файл: {filename}\nСтенограмма:\n{text}"
    last_error: Exception | None = None
    for choice in _llm_iter_choices('keywords'):
        label = _llm_choice_label(choice)
        provider = _llm_choice_provider(choice)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            provider_name = provider
            _provider, response = _llm_send_chat(
                choice,
                messages,
                temperature=0.0,
                max_tokens=min(200, _lm_max_output_tokens()),
                timeout=90,
                cache_bucket='keywords_query' if is_query else 'keywords_transcript',
            )
            if _llm_response_indicates_busy(response):
                app.logger.info(f"LLM keywords endpoint занята ({label}), переключаемся")
                last_error = RuntimeError('busy')
                continue
            response.raise_for_status()
            data = response.json()
            content = _llm_extract_content(_provider, data)
            try:
                obj = json.loads(content)
                if isinstance(obj, list):
                    return [str(x) for x in obj][:12]
            except Exception:
                pass
            m = re.search(r"```json\s*(\[.*?\])\s*```", content, flags=re.S)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, list):
                        return [str(x) for x in obj][:12]
                except Exception:
                    pass
            rough = re.split(r"[\n;,]", content)
            res = [w.strip(" \t\r\n-•") for w in rough if w.strip()]
            if res:
                return res[:12]
        except ValueError as ve:
            last_error = ve
            app.logger.warning(f"LLM keywords endpoint некорректен ({label}): {ve}")
            continue
        except Exception as e:
            last_error = e
            if isinstance(e, RuntimeError) and str(e) == 'llm_cache_only_mode':
                app.logger.info(f"Извлечение ключевых слов пропущено (cache-only, {label})")
            else:
                app.logger.warning(f"Извлечение ключевых слов (LLM) не удалось ({label}, {provider_name}): {e}")
    if last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
        app.logger.warning(f"Извлечение ключевых слов (LLM) не удалось: {last_error}")
    return []

def call_lmstudio_vision(image_path: Path, filename: str):
    """Распознавание и описание изображения через LM Studio (совместимый с OpenAI Vision)."""
    try:
        import base64
    except Exception as e:
        app.logger.warning(f"Визуальное распознавание недоступно: {e}")
        return {}

    mime = "image/png"
    suf = image_path.suffix.lower()
    if suf in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif suf in (".webp",):
        mime = "image/webp"
    elif suf in (".bmp",):
        mime = "image/bmp"
    elif suf in (".tif", ".tiff"):
        mime = "image/tiff"
    try:
        raw = image_path.read_bytes()
    except Exception as e:
        app.logger.warning(f"Визуальное распознавание: не удалось прочитать файл {image_path}: {e}")
        return {}
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode('ascii')

    system = PROMPTS.get('vision_system') or DEFAULT_PROMPTS.get('vision_system', '')
    user_content = [
        {"type": "text", "text": f"Файл: {filename}. Опиши и укажи ключевые слова."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    def _extract_kws_from_text(txt: str):
        try:
            t = (txt or '').replace('*', ' ')
            m = re.search(r"(?i)ключ[^\n\r:]*?:\s*(.+)", t)
            if m:
                line = m.group(1).strip()
                line = re.split(r"\n|\r|\u2028|\u2029|\s{2,}", line)[0]
                parts = [p.strip(" \t\r\n,.;·•-") for p in line.split(',')]
                parts = [p for p in parts if p]
                cleaned = re.sub(r"(?i)ключ[^\n\r:]*?:.*", "", t)
                return parts[:16], cleaned.strip()
        except Exception:
            return [], txt
        return [], txt

    last_error: Exception | None = None
    for choice in _llm_iter_choices('vision'):
        label = _llm_choice_label(choice)
        provider = _llm_choice_provider(choice)
        if provider not in ('openai', 'openrouter', 'azure_openai'):
            app.logger.info(f"LLM vision endpoint пропускается ({label}): провайдер {provider} не поддерживается")
            continue
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        try:
            _provider, response = _llm_send_chat(
                choice,
                messages,
                temperature=0.2,
                max_tokens=500,
                timeout=180,
                cache_bucket='vision',
            )
            if _llm_response_indicates_busy(response):
                app.logger.info(f"LLM vision endpoint занята ({label}), переключаемся")
                last_error = RuntimeError('busy')
                continue
            response.raise_for_status()
            data = response.json()
            content = _llm_extract_content(_provider, data)
            try:
                obj = json.loads(content)
                if isinstance(obj, dict):
                    kws_list = obj.get("keywords") if isinstance(obj.get("keywords"), list) else []
                    if not kws_list:
                        kws_list, cleaned = _extract_kws_from_text(obj.get("description") or "")
                        if kws_list:
                            obj["description"] = cleaned
                            obj["keywords"] = kws_list
                    if "keywords" in obj and isinstance(obj["keywords"], list):
                        obj["keywords"] = [str(x) for x in obj["keywords"]][:16]
                    return obj
            except Exception:
                pass
            m = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.S)
            if m:
                try:
                    obj = json.loads(m.group(1))
                    if isinstance(obj, dict):
                        kws_list = obj.get("keywords") if isinstance(obj.get("keywords"), list) else []
                        if not kws_list:
                            kws_list, cleaned = _extract_kws_from_text(obj.get("description") or "")
                            if kws_list:
                                obj["description"] = cleaned
                                obj["keywords"] = kws_list
                        if "keywords" in obj and isinstance(obj["keywords"], list):
                            obj["keywords"] = [str(x) for x in obj["keywords"]][:16]
                        return obj
                except Exception:
                    pass
            kws_guess, cleaned = _extract_kws_from_text(content or '')
            return {"description": cleaned.strip()[:2000], "keywords": kws_guess}
        except ValueError as ve:
            last_error = ve
            app.logger.warning(f"LLM vision endpoint некорректен ({label}): {ve}")
            continue
        except Exception as e:
            last_error = e
            if isinstance(e, RuntimeError) and str(e) == 'llm_cache_only_mode':
                app.logger.info(f"Визуальное распознавание пропущено (cache-only, {label})")
            else:
                app.logger.warning(f"Визуальное распознавание не удалось ({label}): {e}")
    if last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
        app.logger.warning(f"Визуальное распознавание не удалось: {last_error}")
    return {}

def call_lmstudio_for_metadata(text: str, filename: str):
    """
    Вызов OpenAI-совместимого API (LM Studio) для извлечения метаданных.
    Возвращает dict. Терпимо относится к не-JSON ответам: пытается вытащить из ```json ...``` блока.
    """
    text = (text or "")[: int(os.getenv("LLM_TEXT_LIMIT", "15000"))]
    text = text[:_lm_max_input_chars()]

    system = PROMPTS.get('metadata_system') or DEFAULT_PROMPTS.get('metadata_system', '')
    type_list, type_hints = _material_type_prompt_context()
    if "{{TYPE_LIST}}" in system:
        system = system.replace("{{TYPE_LIST}}", type_list or "document")
    elif type_list:
        system = f"{system}\nДоступные типы материалов: {type_list}."
    if "{{TYPE_HINTS}}" in system:
        system = system.replace("{{TYPE_HINTS}}", type_hints or "")
    elif type_hints:
        system = f"{system}\nТипы и подсказки:\n{type_hints}"
    system = system.strip()

    last_error: Exception | None = None
    for choice in _llm_iter_choices('metadata'):
        label = _llm_choice_label(choice)
        provider = _llm_choice_provider(choice)
        provider_name = provider
        current_text = text

        max_retries = 3
        backoff = 1.0
        busy = False
        for attempt in range(1, max_retries + 1):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Файл: {filename}\nФрагмент текста:\n{current_text}"},
            ]
            try:
                _provider, response = _llm_send_chat(
                    choice,
                    messages,
                    temperature=0.0,
                    max_tokens=min(800, _lm_max_output_tokens()),
                    timeout=120,
                    cache_bucket='metadata',
                )
                if _llm_response_indicates_busy(response):
                    app.logger.info(f"LLM metadata endpoint занята ({label}), переключаемся")
                    busy = True
                    last_error = RuntimeError('busy')
                    break
                text_snippet = (response.text or '')[:2000]
                if response.status_code != 200:
                    app.logger.warning(f"LLM HTTP {response.status_code} ({label}, попытка {attempt}): {text_snippet}")
                    response.raise_for_status()
                data = response.json()
                content = _llm_extract_content(_provider, data)
                try:
                    return json.loads(content)
                except Exception:
                    pass
                m = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        app.logger.warning("Не удалось разобрать JSON внутри блока ```json из ответа LLM")
                m = re.search(r"(\{.*\})", content, flags=re.S)
                if m:
                    try:
                        return json.loads(m.group(1))
                    except Exception:
                        app.logger.warning("Не удалось разобрать JSON‑фрагмент из ответа LLM")
                app.logger.warning(f"LLM вернул не‑JSON контент ({label}, первые 300 символов): {content[:300]}")
                return {}
            except RuntimeError as e:
                last_error = e
                if str(e) == 'llm_cache_only_mode':
                    app.logger.info(f"LLM metadata пропущено (cache-only, {label})")
                else:
                    app.logger.warning(f"LLM metadata unexpected runtime error ({label}, {provider_name}): {e}")
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                app.logger.warning(f"Исключение при запросе к LLM ({label}, {provider_name}, попытка {attempt}): {e}")
                if isinstance(e, requests.HTTPError):
                    resp = getattr(e, 'response', None)
                    detail = ''
                    if resp is not None:
                        try:
                            detail = resp.text or ''
                        except Exception:
                            detail = ''
                    if detail and 'number of tokens to keep' in detail.lower() and len(current_text) > 800:
                        current_text = current_text[: max(len(current_text) // 2, 800)]
                        app.logger.info(f"LLM metadata: сокращаем контекст до {len(current_text)} символов и повторяем")
                        continue
                if attempt < max_retries:
                    import time
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            except ValueError as e:
                app.logger.warning(f"LLM metadata endpoint некорректен ({label}, {provider_name}, попытка {attempt}): {e}")
                last_error = e
                break
            except Exception as e:
                app.logger.warning(f"Unexpected error calling LLM ({label}, {provider_name}, попытка {attempt}): {e}")
                last_error = e
                break
        if busy:
            continue
    if last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
        app.logger.warning(f"Метаданные через LLM не получены: {last_error}")
    return {}

def upsert_tag(file_obj: File, key: str, value: str):
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return
    t = Tag.query.filter_by(file_id=file_obj.id, key=key, value=value).first()
    if not t:
        t = Tag(file_id=file_obj.id, key=key, value=value)
        db.session.add(t)
        _invalidate_facets_cache('tag upsert')

def _upsert_keyword_tags(file_obj: File):
    """Разложить строку ключевых слов на отдельные теги 'ключевое слово'."""
    try:
        # Сначала удалим существующие теги ключевых слов, чтобы не плодить дубли
        try:
            Tag.query.filter_by(file_id=file_obj.id).filter(
                or_(Tag.key == 'ключевое слово', Tag.key == 'keywords')
            ).delete(synchronize_session=False)
            _invalidate_facets_cache('tag cleanup')
        except Exception:
            pass
        raw = (file_obj.keywords or '')
        if not raw:
            return
        parts = re.split(r"[\n,;]+", str(raw))
        seen = set()
        for kw in parts:
            w = (kw or '').strip()
            if not w:
                continue
            wl = w.lower()
            if wl in seen:
                continue
            seen.add(wl)
            upsert_tag(file_obj, 'ключевое слово', w)
        try:
            search_service.sync_file(file_obj)
        except Exception:
            pass
    except Exception:
        pass

def normalize_material_type(s: str) -> str:
    s = (s or '').strip().lower()
    mapping = {
        'диссертация': 'dissertation',
        'автореферат': 'dissertation_abstract',
        'автореферат диссертации': 'dissertation_abstract',
        'статья': 'article', 'article': 'article', 'paper': 'article',
        'журнал': 'journal', 'journal': 'journal', 'magazine': 'journal', 'вестник': 'journal',
        'учебник': 'textbook', 'пособие': 'textbook', 'учебное пособие': 'textbook',
        'монография': 'monograph',
        'отчет': 'report', 'отчёт': 'report', 'report': 'report',
        'патент': 'patent', 'patent': 'patent',
        'презентация': 'presentation', 'presentation': 'presentation',
        'тезисы': 'proceedings', 'proceedings': 'proceedings', 'труды': 'proceedings',
        'стандарт': 'standard', 'gost': 'standard', 'gost r': 'standard', 'standard': 'standard',
        'заметки': 'note', 'note': 'note',
        'document': 'document', 'документ': 'document',
        'image': 'image', 'изображение':'image', 'картинка':'image'
    }
    # пытаемся найти прямое соответствие
    if s in mapping:
        return mapping[s]
    # частичные совпадения
    for k, v in mapping.items():
        if k in s:
            return v
    return s or 'document'


def _looks_like_author_line(line: str) -> bool:
    line = (line or '').strip()
    if not line or len(line.split()) > 16:
        return False
    if re.search(r"\d{2,}|стр\.|с\.\s*\d", line, flags=re.I):
        return False
    # Классические форматы: Иванов И.И.; Иванов Иван Иванович; Ivanov I.I.; John Smith
    patterns = [
        r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.",
        r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+",
        r"[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.",
        r"[A-Z][a-z]+\s+[A-Z][a-z]+",
    ]
    return any(re.search(pat, line) for pat in patterns)


MIN_JOURNAL_TOC_ENTRIES = 5
MIN_JOURNAL_PAGES = 20


def _extract_journal_toc_entries(text: str, limit: int = 30) -> list[dict[str, str]]:
    """Извлекает записи оглавления журнала: авторы, название, страница."""
    if not text:
        return []
    lower = text.lower()
    markers = ["оглавлен", "содержани", "contents", "table of contents"]
    positions = [lower.find(m) for m in markers if lower.find(m) != -1]
    if not positions:
        return []
    start = min(positions)
    slice_end = min(len(text), start + 15000)
    lines = text[start:slice_end].splitlines()
    entries: list[dict[str, str]] = []
    pending_authors: str | None = None
    blank_streak = 0
    for raw in lines[1:]:
        if len(entries) >= limit:
            break
        line = raw.strip()
        if not line:
            blank_streak += 1
            if blank_streak >= 3 and entries:
                break
            continue
        blank_streak = 0
        cleaned = re.sub(r"^[\dIVXLCM\s\.\-–·•]+", "", line, flags=re.IGNORECASE)
        cleaned = cleaned.replace("…", " ")
        cleaned = re.sub(r"[·•]{2,}", " ", cleaned)
        cleaned = re.sub(r"\.{2,}", " ", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if len(cleaned) < 4:
            continue
        page_match = re.search(r"(\d{1,4})(?:\s*(?:стр\.|с\.)?)?$", cleaned, flags=re.IGNORECASE)
        if not page_match:
            if _looks_like_author_line(cleaned):
                pending_authors = cleaned
            continue
        page = page_match.group(1)
        body = cleaned[:page_match.start()].strip(" .•—-–\t")
        if not body:
            continue
        authors = None
        title = body
        # Попытки отделить авторов
        separators = [" — ", " - ", " – ", " —", ": ", "; "]
        for sep in separators:
            if sep in body:
                left, right = body.split(sep, 1)
                if _looks_like_author_line(left):
                    authors = left.strip()
                    title = right.strip()
                    break
        if authors is None:
            m = re.match(r"^((?:[A-ZА-ЯЁ][A-Za-zА-Яа-яё\-']+(?:\s+[A-ZА-ЯЁ]\.[A-ZА-ЯЁ]\.)?(?:,\s*)?)+)\s+(.+)$", body)
            if m and _looks_like_author_line(m.group(1)):
                authors = m.group(1).strip()
                title = m.group(2).strip()
            elif pending_authors and not _looks_like_author_line(body):
                authors = pending_authors.strip()
                title = body.strip()
                pending_authors = None
        entry = {
            'title': title.strip(),
            'page': page.strip(),
        }
        if authors:
            entry['authors'] = authors
        entries.append(entry)
        pending_authors = None
    return entries


def _material_type_profiles() -> List[Dict[str, Any]]:
    runtime = _rt()
    items = getattr(runtime, "material_types", None) or []
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for entry in items:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "").strip()
        if not key or key in seen:
            continue
        cleaned.append(entry)
        seen.add(key)
    if "document" not in seen:
        cleaned.append({
            "key": "document",
            "label": "Документ",
            "enabled": True,
            "priority": -100,
            "threshold": 0.0,
        })
    return cleaned


def _material_type_rank_key(entry: Dict[str, Any]) -> Tuple[float, int, int, int, int]:
    score = float(entry.get("score") or 0.0)
    priority = int(entry.get("priority") or 0)
    special_hit = 1 if entry.get("special_match") else 0
    signal = int(entry.get("raw_text_count") or 0) + int(entry.get("raw_filename_count") or 0)
    index = int(entry.get("index") or 0)
    return (score, priority, special_hit, signal, -index)


def _evaluate_material_type_profile(
    profile: Dict[str, Any],
    ext_clean: str,
    text_lower: str,
    filename_lower: str,
    journal_toc: List[dict[str, str]],
    idx: int,
) -> Optional[Dict[str, Any]]:
    key = str(profile.get("key") or "").strip()
    if not key:
        return None
    if not profile.get("enabled", True):
        return None

    flow = [str(v or "").strip().lower() for v in profile.get("flow", []) if str(v or "").strip()]
    allow_extension = (not flow) or any(token in {"extension", "any"} for token in flow)
    allow_filename = (not flow) or any(token in {"filename", "any"} for token in flow)
    allow_heuristics = (not flow) or any(token in {"heuristics", "text", "any"} for token in flow)

    priority = int(profile.get("priority") or 0)
    threshold = float(profile.get("threshold", 1.0))
    extension_weight = float(profile.get("extension_weight", 2.0))
    filename_weight = float(profile.get("filename_weight", 1.5))
    text_weight = float(profile.get("text_weight", 1.0))

    require_extension = bool(profile.get("require_extension", False))
    require_filename = bool(profile.get("require_filename", False))
    require_text = bool(profile.get("require_text", False))

    extensions = [
        str(item or "").strip().lower().lstrip(".")
        for item in profile.get("extensions", [])
        if str(item or "").strip()
    ]
    extension_match = bool(ext_clean and extensions and ext_clean in extensions)
    if require_extension and not extension_match:
        return None

    filename_keywords = [
        str(item or "").strip().lower()
        for item in profile.get("filename_keywords", [])
        if str(item or "").strip()
    ]
    filename_count = sum(1 for kw in filename_keywords if kw and kw in filename_lower)
    if require_filename and filename_keywords and filename_count == 0:
        return None

    text_keywords = [
        str(item or "").strip().lower()
        for item in profile.get("text_keywords", [])
        if str(item or "").strip()
    ]
    text_count = sum(1 for kw in text_keywords if kw and kw in text_lower)
    if require_text and text_keywords and text_count == 0:
        return None

    exclude_keywords = [
        str(item or "").strip().lower()
        for item in profile.get("exclude_keywords", [])
        if str(item or "").strip()
    ]
    if exclude_keywords and any(ex in text_lower or ex in filename_lower for ex in exclude_keywords):
        return None

    special = profile.get("special") or {}
    special_match = False
    special_weight = 0.0
    if special.get("journal_toc_required"):
        min_entries = int(special.get("min_toc_entries") or MIN_JOURNAL_TOC_ENTRIES)
        if len(journal_toc) >= max(1, min_entries):
            special_match = True
            special_weight = float(special.get("weight") or 2.0)
        else:
            return None

    score = 0.0
    if allow_extension and extension_match:
        score += extension_weight
    if allow_filename and filename_count:
        score += filename_weight * filename_count
    if allow_heuristics and text_count:
        score += text_weight * text_count
    if special_match:
        score += special_weight

    effective_threshold = max(0.0, threshold if threshold is not None else 0.0)
    if score < effective_threshold and effective_threshold > 0 and key != "document":
        return None

    extension_stage = allow_extension and extension_match
    filename_stage = allow_filename and filename_count > 0
    heuristics_stage = allow_heuristics and (text_count > 0 or filename_count > 0 or special_match)

    if not (extension_stage or filename_stage or heuristics_stage):
        if key == "document":
            heuristics_stage = True
        else:
            return None

    return {
        "key": key,
        "score": float(score),
        "priority": priority,
        "extension": bool(extension_stage),
        "filename": bool(filename_stage),
        "heuristics": bool(heuristics_stage),
        "special_match": bool(special_match),
        "raw_extension_match": bool(extension_match),
        "raw_filename_count": int(filename_count),
        "raw_text_count": int(text_count),
        "index": idx,
    }


def _collect_material_type_matches(ext: str, text_excerpt: str, filename: str) -> List[Dict[str, Any]]:
    profiles = _material_type_profiles()
    if not profiles:
        return []
    ext_clean = (ext or "").lower().lstrip(".")
    text_lower = (text_excerpt or "").lower()
    filename_lower = (filename or "").lower()
    needs_toc = any((profile.get("special") or {}).get("journal_toc_required") for profile in profiles)
    journal_toc = _extract_journal_toc_entries(text_excerpt or "") if needs_toc else []
    matches: List[Dict[str, Any]] = []
    for idx, profile in enumerate(profiles):
        evaluation = _evaluate_material_type_profile(
            profile,
            ext_clean,
            text_lower,
            filename_lower,
            journal_toc,
            idx,
        )
        if evaluation:
            matches.append(evaluation)
    return matches


def _material_type_prompt_context() -> Tuple[str, str]:
    profiles = _material_type_profiles()
    visible = [entry for entry in profiles if entry.get("enabled", True)]
    type_keys: List[str] = []
    hints: List[str] = []
    for entry in visible:
        key = str(entry.get("key") or "").strip()
        if not key:
            continue
        type_keys.append(key)
        label = str(entry.get("label") or "").strip()
        hint = str(entry.get("llm_hint") or entry.get("description") or label or key).strip()
        if hint:
            hints.append(f"{key}: {hint}")
    type_list = ", ".join(type_keys)
    type_hints = "\n".join(f"- {line}" for line in hints if line)
    return type_list, type_hints


def _journal_safety_check(material_type: str | None, text: str | None, page_count: int | None, entries: list[dict[str, str]] | None = None) -> str | None:
    mt = (material_type or '').strip().lower()
    if mt != 'journal':
        return material_type
    entries = entries if entries is not None else (_extract_journal_toc_entries(text or '') if text else [])
    if len(entries) < MIN_JOURNAL_TOC_ENTRIES:
        return 'article'
    if page_count is not None and page_count < MIN_JOURNAL_PAGES:
        return 'article'
    return material_type

def guess_material_type(ext: str, text_excerpt: str, filename: str = "") -> str:
    matches = _collect_material_type_matches(ext, text_excerpt, filename)
    heuristics_matches = [
        entry for entry in matches if entry.get("heuristics") and entry.get("key") != "document"
    ]
    if heuristics_matches:
        best = max(heuristics_matches, key=_material_type_rank_key)
        return best["key"]
    doc_entry = next((entry for entry in matches if entry.get("key") == "document"), None)
    if doc_entry:
        return doc_entry["key"]
    return "document"


def _detect_type_pre_llm(ext: str, text_excerpt: str, filename: str) -> str | None:
    flow = [p.strip() for p in (TYPE_DETECT_FLOW or '').split(',') if p.strip()]
    if not flow:
        flow = ['extension', 'filename', 'heuristics']
    matches = _collect_material_type_matches(ext, text_excerpt, filename)
    if not matches:
        return None
    doc_candidate: Optional[Dict[str, Any]] = None
    for stage in flow:
        stage_key = str(stage or "").strip().lower()
        if not stage_key:
            continue
        if stage_key == 'llm':
            continue
        if stage_key not in {'extension', 'filename', 'heuristics', 'text'}:
            stage_key = 'heuristics'
        actual_key = 'heuristics' if stage_key in {'heuristics', 'text'} else stage_key
        candidates = [entry for entry in matches if entry.get(actual_key)]
        if not candidates:
            continue
        non_doc = [entry for entry in candidates if entry.get("key") != "document"]
        target = non_doc or candidates
        best = max(target, key=_material_type_rank_key)
        if best["key"] == "document":
            if doc_candidate is None:
                doc_candidate = best
            continue
        return best["key"]
    if doc_candidate:
        return doc_candidate["key"]
    return None

def looks_like_dissertation_filename(fn: str) -> bool:
    fl = (fn or '').lower()
    if not fl:
        return False
    if any(tok in fl for tok in ["автореферат", "autoreferat", "автoref"]):
        return True
    if any(tok in fl for tok in ["диссер", "dissert", "thesis"]):
        return True
    return False

def extract_tags_for_type(material_type: str, text: str, filename: str = "") -> dict:
    """Извлечение тегов по типу материала простыми регулярками.
    Возвращает dict: {key: value}.
    """
    t = (text or "")
    tl = t.lower()
    tags = {}
    # Общие сведения
    # Приблизительная оценка языка: доля кириллицы против латиницы
    try:
        cyr = sum(1 for ch in t if ('а' <= ch.lower() <= 'я') or (ch in 'ёЁ'))
        lat = sum(1 for ch in t if 'a' <= ch.lower() <= 'z')
        if cyr + lat > 20:
            tags.setdefault('lang', 'ru' if cyr >= lat else 'en')
    except Exception:
        pass
    # Распространённые идентификаторы
    # Общие
    # DOI (цифровой идентификатор объекта)
    m = re.search(r"\b(10\.\d{4,9}\/[\w\-\.:;()\/[\]A-Za-z0-9]+)", t)
    if m:
        tags.setdefault("doi", m.group(1))
    # ISBN (международный книжный номер)
    m = re.search(r"\bISBN[:\s]*([0-9\- ]{10,20})", t, flags=re.I)
    if m:
        tags.setdefault("isbn", m.group(1).strip())

    if material_type in ("dissertation", "dissertation_abstract"):
        # Научный руководитель (варианты написания)
        m = re.search(r"научн[ыийо]{1,3}\s*[-–:]?\s*руководител[ья][\s:–]+(.{3,80})", tl, flags=re.I)
        if m:
            tags.setdefault("научный руководитель", m.group(1).strip().title())
        # Специальность/шифр ВАК вида 05.13.11 или 01.02.03
        m = re.search(r"(?:шифр|специальн)[^\n\r]{0,30}?(\d{2}\.\d{2}\.\d{2})", tl)
        if not m:
            m = re.search(r"\b(\d{2}\.\d{2}\.\d{2})\b", tl)
        if m:
            tags.setdefault("специальность", m.group(1))
        # Организация/вуз (ФГБОУ ВО, университет, институт, академия, НИУ, МГУ, СПбГУ)
        m = re.search(r"(фгбоу\s*во|университет|институт|академия|ниу|мгу|спбгу)[^\n\r]{0,100}", tl)
        if m:
            tags.setdefault("организация", m.group(0).strip().title())
        # Кафедра
        m = re.search(r"кафедра\s+([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault("кафедра", m.group(1).strip().title())
        # Степень
        if re.search(r"кандидат[а]?\b", tl):
            tags.setdefault("степень", "кандидат")
        elif re.search(r"доктор[а]?\b", tl):
            tags.setdefault("степень", "доктор")
        # Новизна (короткий фрагмент)
        m = re.search(r"научн[ао]я\s+новизн[аы][^\n\r:]*[:\-]\s*(.{20,300})", tl)
        if m:
            tags.setdefault("научная новизна", m.group(1).strip().capitalize())
        # Цель исследования
        m = re.search(r"цель\s+исследован[ияи][^:\n\r]*[:\-]\s*(.{10,200})", tl)
        if m:
            tags.setdefault("цель", m.group(1).strip().capitalize())
        # Объект/предмет исследования
        m = re.search(r"объект\s+исследован[ияи][^:\n\r]*[:\-]\s*(.{10,200})", tl)
        if m:
            tags.setdefault("объект", m.group(1).strip().capitalize())
        m = re.search(r"предмет\s+исследован[ияи][^:\n\r]*[:\-]\s*(.{10,200})", tl)
        if m:
            tags.setdefault("предмет", m.group(1).strip().capitalize())
        # Задачи исследования
        m = re.search(r"задач[аи]\s+исследован[ияи][^:\n\r]*[:\-]\s*(.{10,200})", tl)
        if m:
            tags.setdefault("задачи", m.group(1).strip().capitalize())
        # Положения, выносимые на защиту
        m = re.search(r"вынос[иы]м[ае][^\n\r]{0,40}на\s+защит[уы][^:\n\r]*[:\-]\s*(.{20,300})", tl)
        if m:
            tags.setdefault("на защиту", m.group(1).strip().capitalize())
    elif material_type == "article":
        # Журнал/Вестник
        m = re.search(r"(журнал|вестник|труды|transactions|journal)[:\s\-]+([^\n\r]{3,80})", tl, flags=re.I)
        if m:
            tags.setdefault("журнал", m.group(2).strip().title())
        # Номер/том/страницы
        m = re.search(r"том\s*(\d+)\b.*?№\s*(\d+)\b.*?с\.?\s*(\d+)[\-–](\d+)", tl)
        if m:
            tags.setdefault("том/номер", f"{m.group(1)}/{m.group(2)}")
            tags.setdefault("страницы", f"{m.group(3)}–{m.group(4)}")
        else:
            m = re.search(r"№\s*(\d+)\b.*?с\.?\s*(\d+)[\-–](\d+)", tl)
            if m:
                tags.setdefault("номер", m.group(1))
                tags.setdefault("страницы", f"{m.group(2)}–{m.group(3)}")
    elif material_type == "textbook":
        # Дисциплина
        m = re.search(r"по\s+дисциплин[еы]\s*[:\-]?\s*([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault("дисциплина", m.group(1).strip().title())
        # Издательство
        m = re.search(r"издательств[оа]\s*[:\-]?\s*([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault("издательство", m.group(1).strip().title())
    elif material_type == "monograph":
        # Издательство/город/год часто в шапке
        m = re.search(r"(издательство|изд.)\s*[:\-]?\s*([^\n\r]{3,80})", tl)
        if m:
            tags.setdefault("издательство", m.group(2).strip().title())

    return tags

def prune_missing_files():
    """Удаляет из БД файлы, которых нет на диске."""
    removed = 0
    for f in File.query.all():
        try:
            if not Path(f.path).exists():
                fid = f.id
                db.session.delete(f)
                removed += 1
                try:
                    _delete_file_from_fts(fid)
                except Exception as exc:
                    logger.debug('fts delete failed for missing file %s: %s', fid, exc)
        except Exception:
            fid = getattr(f, 'id', None)
            db.session.delete(f)
            removed += 1
            try:
                _delete_file_from_fts(fid)
            except Exception as exc:
                logger.debug('fts delete failed for missing file %s: %s', fid, exc)
    db.session.commit()
    return removed

# ------------------- Routes -------------------

@app.route("/")
def index():
    return redirect(url_for('app_react'))

def _serve_spa() -> Response:
    """Отдаёт собранный React UI либо dev-index, если билд отсутствует."""
    try:
        dist = BASE_DIR / 'frontend' / 'dist'
        idx = dist / 'index.html'
        if idx.exists():
            return send_file(str(idx))
    except Exception:
        pass
    dev_idx = BASE_DIR / 'frontend' / 'index.html'
    if dev_idx.exists():
        return send_file(str(dev_idx))
    body = (
        "<div style=\"min-height:100vh;display:flex;align-items:center;justify-content:center;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;"
        "padding:24px;text-align:center;\">"
        "<div style=\"max-width:520px;background:rgba(22,27,34,0.85);padding:32px;border-radius:18px;"
        "border:1px solid #30363d;box-shadow:0 20px 48px rgba(0,0,0,0.45);\">"
        "<h1 style=\"margin-top:0;font-size:22px;\">React-интерфейс не собран</h1>"
        "<p style=\"line-height:1.6;\">В каталоге <code>frontend/dist</code> не найден файл <code>index.html</code>."
        " Соберите клиент командой <code>npm install && npm run build</code> внутри директории <code>frontend</code> "
        "или запустите dev-сервер (<code>npm run dev</code>) и проксируйте его вручную.</p>"
        "</div></div>"
    )
    return _html_page("Agregator — UI не найден", body, status=404)


@app.route("/app")
def app_react():
    """Точка входа для React SPA."""
    return _serve_spa()


@app.route("/app/<path:_path>")
def app_react_catchall(_path: str):
    """Внутренний роутинг React: любые пути внутри /app/*."""
    return _serve_spa()


def _collect_login_backgrounds() -> list[dict[str, str]]:
    """Enumerate available HTML backgrounds for the login screen."""
    entries: list[dict[str, str]] = []
    try:
        directory = LOGIN_BACKGROUNDS_DIR
        if not directory.exists():
            return entries
        for item in sorted(directory.glob('*.html')):
            if not item.is_file():
                continue
            label = item.stem.replace('_', ' ').replace('-', ' ').strip().title()
            try:
                text = item.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                text = ''
            match = re.search(r'<title[^>]*>(.*?)</title>', text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                raw = match.group(1).strip()
                if raw:
                    label = re.sub(r'\s+', ' ', raw)
            entries.append({
                'name': item.name,
                'label': label,
                'url': f"/login-backgrounds/{item.name}",
            })
    except Exception as exc:
        try:
            current_app.logger.warning('login backgrounds scan failed: %s', exc)
        except Exception:
            pass
    return entries


@app.route('/api/login-backgrounds')
def api_login_backgrounds():
    """Expose available login background animations."""
    return jsonify(_collect_login_backgrounds())


@app.route('/login-backgrounds/<path:filename>')
def login_background_file(filename: str):
    """Serve individual login background HTML files."""
    if '/' in filename or '\\' in filename:
        abort(404)
    if not filename.lower().endswith('.html'):
        abort(404)
    directory = LOGIN_BACKGROUNDS_DIR
    if not directory.exists():
        abort(404)
    try:
        response = send_from_directory(str(directory), filename)
    except FileNotFoundError:
        abort(404)
    try:
        # Разрешаем встраивание в iframe на этом же домене.
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        csp = response.headers.get('Content-Security-Policy')
        if csp:
            if 'frame-ancestors' not in csp:
                response.headers['Content-Security-Policy'] = csp.rstrip(';') + "; frame-ancestors 'self'"
        else:
            response.headers['Content-Security-Policy'] = "frame-ancestors 'self'"
    except Exception:
        pass
    response.headers['Cache-Control'] = 'no-store'
    return response

@app.route('/assets/<path:filename>')
def vite_assets(filename):
    """Serve Vite built assets if present (dist/assets)."""
    dist_assets = BASE_DIR / 'frontend' / 'dist' / 'assets'
    if dist_assets.exists():
        return send_from_directory(str(dist_assets), filename)
    return ("Not found", 404)

@app.route("/aiword")
def aiword_index():
    _ensure_aiword_access()
    """Serve AiWord app (built with Vite) under /aiword with asset path rewrite and pre‑bootstrap.
    Injects a small script to prefill localStorage with BibTeX from this app's DB and LM Studio config.
    """
    dist = BASE_DIR / 'AiWord' / 'dist'
    idx = dist / 'index.html'
    if not idx.exists():
        return ("AiWord/dist/index.html not found. Please build AiWord or include dist.", 404)
    try:
        html = idx.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return send_file(str(idx))

    # Переписываем абсолютные ссылки на ресурсы в /aiword/assets
    html = html.replace('src="/assets/', 'src="/aiword/assets/')
    html = html.replace('href="/assets/', 'href="/aiword/assets/')
    # Переписываем vite.svg, если он упоминается
    html = html.replace('href="/vite.svg', 'href="/aiword/vite.svg')

    # Готовим bootstrap-скрипт для предварительной загрузки BibTeX и настроек ЛМ
    lm_base = os.getenv('LMSTUDIO_API_BASE', 'http://localhost:1234/v1')
    lm_model = os.getenv('LMSTUDIO_MODEL', 'google/gemma-3n-e4b')
    bootstrap = f"""
    <script>
    (function(){{
      try {{
        // Синхронизируем настройки LM Studio с Agregator
        localStorage.setItem('llm-writer-base-url', {lm_base!r});
        localStorage.setItem('llm-writer-model', {lm_model!r});
      }} catch(e){{}}
      // Загружаем библиографию из базы Agregator
      fetch('/api/aiword/bibtex').then(r=>r.ok?r.text():Promise.resolve('')).then(txt=>{{
        if (!txt) return;
        try {{
          var saved = localStorage.getItem('llm-writer-autosave');
          var data = saved ? JSON.parse(saved) : null;
          if (!data || !data.project || !data.content) {{
            data = {{
              project: {{
                title: 'Новая статья',
                language: 'ru',
                style_guide: 'Пиши чётко и структурированно. Без воды. Сохраняй академический тон, «выводы» — коротко и по делу. Используй Markdown: заголовки, списки, таблицы. Формулы как $...$ или ```math``` при необходимости.',
                persona: 'Ты — научный редактор и соавтор. Помогаешь с планом, структурой, стилем, и аргументацией.'
              }},
              content: '# Заголовок\n\nВставьте/пишите текст здесь...',
              bibText: ''
            }};
          }}
          data.bibText = txt;
          localStorage.setItem('llm-writer-autosave', JSON.stringify(data));
        }} catch(e){{}}
      }}).catch(function(){{}});
    }})();
    </script>
    """
    # Вставляем bootstrap перед первым тегом module-скрипта, если он есть
    marker = '<script type="module"'
    if marker in html:
        html = html.replace(marker, bootstrap + "\n" + marker, 1)
    else:
        html = html.replace('</head>', bootstrap + '\n</head>')
    return html

@app.route('/aiword/assets/<path:filename>')
def aiword_assets(filename):
    _ensure_aiword_access()
    dist_assets = BASE_DIR / 'AiWord' / 'dist' / 'assets'
    if dist_assets.exists():
        return send_from_directory(str(dist_assets), filename)
    return ("Not found", 404)

@app.route('/aiword/vite.svg')
def aiword_vite_svg():
    _ensure_aiword_access()
    p = BASE_DIR / 'AiWord' / 'dist' / 'vite.svg'
    if p.exists():
        return send_file(str(p))
    return ("Not found", 404)


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    smart = str(request.args.get('smart', '')).lower() in ('1','true','yes','on')
    material_type = request.args.get("type", "").strip()
    tag_filters = request.args.getlist("tag")

    # Дополнительные необязательные фильтры (обратная совместимость)
    year_from = (request.args.get("year_from") or "").strip()
    year_to = (request.args.get("year_to") or "").strip()
    size_min = (request.args.get("size_min") or "").strip()
    size_max = (request.args.get("size_max") or "").strip()
    collection_filter = _parse_collection_param(request.args.get('collection_id'))

    query = File.query.join(Collection, File.collection_id == Collection.id)
    query = _apply_file_access_filter(query)
    try:
        query = query.filter(Collection.searchable == True)
    except Exception:
        pass
    if collection_filter is not None:
        query = query.filter(File.collection_id == collection_filter)
    if material_type:
        query = query.filter(File.material_type == material_type)
    if q and not smart:
        query = _apply_text_search_filter(query, q)
    if year_from:
        query = query.filter(File.year >= year_from)
    if year_to:
        query = query.filter(File.year <= year_to)
    if size_min:
        try:
            query = query.filter(File.size >= int(size_min))
        except Exception:
            pass
    if size_max:
        try:
            query = query.filter(File.size <= int(size_max))
        except Exception:
            pass
    for tf in tag_filters:
        if "=" in tf:
            k, v = tf.split("=", 1)
            t = aliased(Tag)
            query = query.join(t, t.file_id == File.id).filter(and_(t.key == k, t.value.ilike(f"%{v}%")))
    query = query.distinct()

    if q and smart:
        candidates = _search_candidate_ids(q)
        if candidates:
            query = query.filter(File.id.in_(candidates))
        elif candidates == []:
            return jsonify([])

    rows = query.order_by(File.mtime.desc().nullslast()).limit(200).all()
    if q and smart:
        qlem = list(_expand_synonyms(_lemmas(q)))
        def _match(f: File) -> bool:
            lab = set(_lemmas(_row_text_for_search(f)))
            return any(l in lab for l in qlem)
        rows = [r for r in rows if _match(r)]
    return jsonify([{
        "id": r.id,
        "title": r.title,
        "author": r.author,
        "year": r.year,
        "material_type": r.material_type,
        "path": r.path,
        "rel_path": r.rel_path,
        "text_excerpt": r.text_excerpt,
        "abstract": getattr(r, 'abstract', None),
        "tags": [{"key": t.key, "value": t.value} for t in r.tags]
    } for r in rows])

@app.route("/api/search_v2")
def api_search_v2():
    """Расширенный поиск с пагинацией и доп. фильтрами.
    Query: q, type, tag=key=val (multi), year_from, year_to, size_min, size_max, limit, offset
    Returns: { items: [...], total: n }
    """
    q = request.args.get("q", "").strip()
    smart = str(request.args.get('smart', '')).lower() in ('1','true','yes','on')
    material_type = request.args.get("type", "").strip()
    tag_filters = request.args.getlist("tag")
    year_from = (request.args.get("year_from") or "").strip()
    year_to = (request.args.get("year_to") or "").strip()
    size_min = (request.args.get("size_min") or "").strip()
    size_max = (request.args.get("size_max") or "").strip()
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 200)
    except Exception:
        limit = 50
    try:
        offset = max(int(request.args.get("offset", 0)), 0)
    except Exception:
        offset = 0

    base = File.query.join(Collection, File.collection_id == Collection.id)
    base = _apply_file_access_filter(base)
    collection_filter = _parse_collection_param(request.args.get('collection_id'))
    if collection_filter is not None:
        base = base.filter(File.collection_id == collection_filter)
    try:
        base = base.filter(Collection.searchable == True)
    except Exception:
        pass
    if material_type:
        base = base.filter(File.material_type == material_type)
    if q and not smart:
        base = _apply_text_search_filter(base, q)
    if year_from:
        base = base.filter(File.year >= year_from)
    if year_to:
        base = base.filter(File.year <= year_to)
    if size_min:
        try:
            base = base.filter(File.size >= int(size_min))
        except Exception:
            pass
    if size_max:
        try:
            base = base.filter(File.size <= int(size_max))
        except Exception:
            pass
    qx = base
    for tf in tag_filters:
        if "=" in tf:
            k, v = tf.split("=", 1)
            if k == 'author':
                qx = qx.filter(File.author.ilike(f"%{v}%"))
            else:
                t = aliased(Tag)
                qx = qx.join(t, t.file_id == File.id).filter(and_(t.key == k, t.value.ilike(f"%{v}%")))
    qx = qx.distinct()

    if q and smart:
        candidates = _search_candidate_ids(q)
        if candidates:
            qx = qx.filter(File.id.in_(candidates))
        elif candidates == []:
            return jsonify({"items": [], "total": 0})
        # Морфологическая фильтрация на сервере с синонимами
        qlem = list(_expand_synonyms(_lemmas(q)))
        # Ограничиваем выборку разумным числом для подсчёта, упорядоченным по свежести
        cap = 10000
        cand = qx.order_by(File.mtime.desc().nullslast()).limit(cap).all()
        def _match(f: File) -> bool:
            lab = set(_lemmas(_row_text_for_search(f)))
            return any(l in lab for l in qlem)
        matched = [r for r in cand if _match(r)]
        total = len(matched)
        rows = matched[offset:offset+limit]
    else:
        total = qx.count()
        rows = qx.order_by(File.mtime.desc().nullslast()).offset(offset).limit(limit).all()
    items = [{
        "id": r.id,
        "title": r.title,
        "author": r.author,
        "year": r.year,
        "material_type": r.material_type,
        "path": r.path,
        "rel_path": r.rel_path,
        "text_excerpt": r.text_excerpt,
        "abstract": getattr(r, 'abstract', None),
        "tags": [{"key": t.key, "value": t.value} for t in r.tags]
    } for r in rows]
    return jsonify({"items": items, "total": total})


@app.route('/api/voice-search', methods=['POST'])
def api_voice_search():
    """Принимает короткий аудиофрагмент, распознаёт его и возвращает текст."""
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Не авторизовано'}), 401
    if not TRANSCRIBE_ENABLED:
        return jsonify({'ok': False, 'error': 'Распознавание аудио отключено'}), 503
    if request.content_length and request.content_length > 15 * 1024 * 1024:
        return jsonify({'ok': False, 'error': 'Аудио слишком большое'}), 413

    audio = request.files.get('audio')
    if not audio or audio.filename is None:
        return jsonify({'ok': False, 'error': 'Не удалось получить аудио'}), 400

    filename = secure_filename(audio.filename) or 'voice-input.webm'
    suffix = Path(filename).suffix or '.webm'
    src_path: Path | None = None
    wav_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            audio.save(tmp)
            src_path = Path(tmp.name)

        transcript = ''
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                wav_path = Path(wav_tmp.name)
            _convert_to_wav_pcm16(src_path, wav_path)
            transcript = transcribe_audio(wav_path, limit_chars=5000)
        except Exception as conv_exc:
            app.logger.info(f'[voice-search] wav convert failed, fallback to raw: {conv_exc}')
            transcript = transcribe_audio(src_path, limit_chars=5000)

        transcript = (transcript or '').strip()
        if not transcript:
            return jsonify({'ok': True, 'text': '', 'warning': 'Речь не распознана'})

        try:
            _log_user_action(user, 'voice_search', detail=json.dumps({'length': len(transcript)}))
        except Exception:
            pass

        return jsonify({'ok': True, 'text': transcript})
    except Exception as exc:
        app.logger.warning(f'[voice-search] failed: {exc}')
        return jsonify({'ok': False, 'error': 'Не удалось распознать аудио'}), 500
    finally:
        if src_path:
            try:
                src_path.unlink()
            except FileNotFoundError:
                pass
        if wav_path:
            try:
                wav_path.unlink()
            except FileNotFoundError:
                pass

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    """JSON API для чтения/изменения основных настроек UI/сканирования."""
    _require_admin()
    runtime = _rt()
    provider_default = runtime.lm_default_provider or 'openai'
    if request.method == 'GET':
        _ensure_llm_schema_once()
        try:
            cols = Collection.query.order_by(Collection.name.asc()).all()
            counts = {}
            try:
                rows = db.session.query(File.collection_id, func.count(File.id)).group_by(File.collection_id).all()
                counts = {cid: int(cnt) for (cid, cnt) in rows}
            except Exception:
                counts = {}
            collections = [{
                'id': c.id,
                'name': c.name,
                'slug': c.slug,
                'searchable': bool(c.searchable),
                'graphable': bool(c.graphable),
                'is_private': bool(getattr(c, 'is_private', False)),
                'count': int(counts.get(c.id, 0)),
            } for c in cols]
        except Exception:
            collections = []
        try:
            llms = LlmEndpoint.query.order_by(LlmEndpoint.created_at.desc()).all()
            llm_items = [{
                'id': ep.id,
                'name': ep.name,
                'base_url': ep.base_url,
                'model': ep.model,
                'weight': float(ep.weight or 0.0),
                'purpose': ep.purpose,
                'purposes': _llm_parse_purposes(ep.purpose),
                'provider': (ep.provider or provider_default),
            } for ep in llms]
        except Exception:
            llm_items = []
        try:
            ai_entries = AiWordAccess.query.join(User, AiWordAccess.user_id == User.id).all()
            ai_users = [{
                'user_id': entry.user_id,
                'username': entry.user.username if entry.user else None,
                'full_name': entry.user.full_name if entry.user else None,
            } for entry in ai_entries]
        except Exception:
            ai_users = []
        return jsonify({
            'scan_root': str(runtime.scan_root),
            'extract_text': bool(runtime.extract_text),
            'lm_base': runtime.lmstudio_api_base,
            'lm_model': runtime.lmstudio_model,
            'lm_key': runtime.lmstudio_api_key,
            'lm_provider': provider_default,
            'rag_embedding_backend': runtime.rag_embedding_backend,
            'rag_embedding_model': runtime.rag_embedding_model,
            'rag_embedding_dim': int(runtime.rag_embedding_dim),
            'rag_embedding_batch': int(runtime.rag_embedding_batch_size),
            'rag_embedding_device': runtime.rag_embedding_device,
            'rag_embedding_endpoint': runtime.rag_embedding_endpoint or runtime.lmstudio_api_base,
            'rag_embedding_api_key': runtime.rag_embedding_api_key or runtime.lmstudio_api_key,
            'transcribe_enabled': bool(runtime.transcribe_enabled),
            'transcribe_backend': runtime.transcribe_backend,
            'transcribe_model': runtime.transcribe_model_path,
            'transcribe_language': runtime.transcribe_language,
            'summarize_audio': bool(runtime.summarize_audio),
            'audio_keywords_llm': bool(runtime.audio_keywords_llm),
            'vision_images': bool(runtime.images_vision_enabled),
            'kw_to_tags': bool(runtime.keywords_to_tags_enabled),
            'type_detect_flow': runtime.type_detect_flow,
            'type_llm_override': bool(runtime.type_llm_override),
            'import_subdir': runtime.import_subdir,
            'move_on_rename': bool(runtime.move_on_rename),
            'collections_in_dirs': bool(runtime.collections_in_separate_dirs),
            'collection_type_subdirs': bool(runtime.collection_type_subdirs),
            'type_dirs': runtime.type_dirs,
            'ocr_langs': runtime.ocr_langs_cfg,
            'pdf_ocr_pages': int(runtime.pdf_ocr_pages_cfg),
            'prompts': runtime.prompts,
            'prompt_defaults': dict(DEFAULT_PROMPTS),
            'ai_rerank_llm': bool(runtime.ai_rerank_llm),
            'ocr_first_page_dissertation': bool(runtime.always_ocr_first_page_dissertation),
            'llm_cache_enabled': bool(runtime.llm_cache_enabled),
            'llm_cache_ttl_seconds': int(runtime.llm_cache_ttl_seconds),
            'llm_cache_max_items': int(runtime.llm_cache_max_items),
            'llm_cache_only_mode': bool(runtime.llm_cache_only_mode),
            'search_cache_enabled': bool(runtime.search_cache_enabled),
            'search_cache_ttl_seconds': int(runtime.search_cache_ttl_seconds),
            'search_cache_max_items': int(runtime.search_cache_max_items),
            'lm_max_input_chars': int(runtime.lm_max_input_chars),
            'lm_max_output_tokens': int(runtime.lm_max_output_tokens),
            'azure_openai_api_version': runtime.azure_openai_api_version,
            'collections': collections,
            'llm_endpoints': llm_items,
            'llm_purposes': LLM_PURPOSES,
            'llm_providers': LLM_PROVIDER_OPTIONS,
            'aiword_users': ai_users,
            'default_use_llm': bool(runtime.default_use_llm),
            'default_prune': bool(runtime.default_prune),
            'material_types': runtime.material_types,
        })

    data = request.json or {}
    before = runtime_settings_store.snapshot()
    update_payload = {}

    def _set(key: str, value):
        update_payload[key] = value

    if 'scan_root' in data:
        value = data.get('scan_root') or str(runtime.scan_root)
        _set('SCAN_ROOT', str(value))
    if 'extract_text' in data:
        _set('EXTRACT_TEXT', data.get('extract_text'))
    if 'lm_base' in data:
        _set('LMSTUDIO_API_BASE', data.get('lm_base'))
    if 'lm_model' in data:
        _set('LMSTUDIO_MODEL', data.get('lm_model'))
    if 'lm_key' in data:
        _set('LMSTUDIO_API_KEY', data.get('lm_key'))
    if 'lm_provider' in data:
        candidate = (data.get('lm_provider') or '').strip().lower()
        if candidate not in LLM_PROVIDER_CHOICES:
            candidate = runtime.lm_default_provider if runtime.lm_default_provider in LLM_PROVIDER_CHOICES else 'openai'
        _set('LM_DEFAULT_PROVIDER', candidate)
    if 'rag_embedding_backend' in data:
        backend = (data.get('rag_embedding_backend') or '').strip().lower()
        _set('RAG_EMBEDDING_BACKEND', backend or runtime.rag_embedding_backend)
    if 'rag_embedding_model' in data:
        _set('RAG_EMBEDDING_MODEL', data.get('rag_embedding_model'))
    if 'rag_embedding_dim' in data:
        try:
            _set('RAG_EMBEDDING_DIM', int(data.get('rag_embedding_dim')))
        except Exception:
            pass
    if 'rag_embedding_batch' in data:
        try:
            _set('RAG_EMBEDDING_BATCH', int(data.get('rag_embedding_batch')))
        except Exception:
            pass
    if 'rag_embedding_device' in data:
        _set('RAG_EMBEDDING_DEVICE', (data.get('rag_embedding_device') or '').strip())
    if 'rag_embedding_endpoint' in data:
        _set('RAG_EMBEDDING_ENDPOINT', (data.get('rag_embedding_endpoint') or '').strip())
    if 'rag_embedding_api_key' in data:
        _set('RAG_EMBEDDING_API_KEY', data.get('rag_embedding_api_key') or '')
    if 'transcribe_enabled' in data:
        _set('TRANSCRIBE_ENABLED', data.get('transcribe_enabled'))
    if 'transcribe_backend' in data:
        _set('TRANSCRIBE_BACKEND', data.get('transcribe_backend'))
    if 'transcribe_model' in data:
        _set('TRANSCRIBE_MODEL_PATH', data.get('transcribe_model'))
    if 'transcribe_language' in data:
        _set('TRANSCRIBE_LANGUAGE', data.get('transcribe_language'))
    if 'summarize_audio' in data:
        _set('SUMMARIZE_AUDIO', data.get('summarize_audio'))
    if 'audio_keywords_llm' in data:
        _set('AUDIO_KEYWORDS_LLM', data.get('audio_keywords_llm'))
    if 'vision_images' in data:
        _set('IMAGES_VISION_ENABLED', data.get('vision_images'))
    if 'kw_to_tags' in data:
        _set('KEYWORDS_TO_TAGS_ENABLED', data.get('kw_to_tags'))
    if 'type_detect_flow' in data:
        _set('TYPE_DETECT_FLOW', data.get('type_detect_flow'))
    if 'type_llm_override' in data:
        _set('TYPE_LLM_OVERRIDE', data.get('type_llm_override'))
    if 'import_subdir' in data:
        value = data.get('import_subdir') or runtime.import_subdir
        _set('IMPORT_SUBDIR', value)
    if 'move_on_rename' in data:
        _set('MOVE_ON_RENAME', data.get('move_on_rename'))
    if 'collections_in_dirs' in data:
        _set('COLLECTIONS_IN_SEPARATE_DIRS', data.get('collections_in_dirs'))
    if 'collection_type_subdirs' in data:
        _set('COLLECTION_TYPE_SUBDIRS', data.get('collection_type_subdirs'))
    if 'type_dirs' in data and isinstance(data.get('type_dirs'), dict):
        _set('TYPE_DIRS', data.get('type_dirs'))
    if 'material_types' in data and isinstance(data.get('material_types'), list):
        _set('MATERIAL_TYPES', data.get('material_types'))
    if 'default_use_llm' in data:
        _set('DEFAULT_USE_LLM', data.get('default_use_llm'))
    if 'default_prune' in data:
        _set('DEFAULT_PRUNE', data.get('default_prune'))
    if 'ocr_langs' in data:
        _set('OCR_LANGS_CFG', data.get('ocr_langs'))
    if 'pdf_ocr_pages' in data:
        _set('PDF_OCR_PAGES_CFG', data.get('pdf_ocr_pages'))
    if 'ocr_first_page_dissertation' in data:
        _set('ALWAYS_OCR_FIRST_PAGE_DISSERTATION', data.get('ocr_first_page_dissertation'))
    if 'prompts' in data and isinstance(data.get('prompts'), dict):
        _set('PROMPTS', data.get('prompts'))
    if 'ai_rerank_llm' in data:
        _set('AI_RERANK_LLM', data.get('ai_rerank_llm'))
    if 'llm_cache_enabled' in data:
        _set('LLM_CACHE_ENABLED', data.get('llm_cache_enabled'))
    if 'llm_cache_ttl_seconds' in data:
        _set('LLM_CACHE_TTL_SECONDS', data.get('llm_cache_ttl_seconds'))
    if 'llm_cache_max_items' in data:
        _set('LLM_CACHE_MAX_ITEMS', data.get('llm_cache_max_items'))
    if 'llm_cache_only_mode' in data:
        _set('LLM_CACHE_ONLY_MODE', data.get('llm_cache_only_mode'))
    if 'search_cache_enabled' in data:
        _set('SEARCH_CACHE_ENABLED', data.get('search_cache_enabled'))
    if 'search_cache_ttl_seconds' in data:
        _set('SEARCH_CACHE_TTL_SECONDS', data.get('search_cache_ttl_seconds'))
    if 'search_cache_max_items' in data:
        _set('SEARCH_CACHE_MAX_ITEMS', data.get('search_cache_max_items'))
    if 'lm_max_input_chars' in data:
        _set('LM_MAX_INPUT_CHARS', data.get('lm_max_input_chars'))
    if 'lm_max_output_tokens' in data:
        _set('LM_MAX_OUTPUT_TOKENS', data.get('lm_max_output_tokens'))
    if 'azure_openai_api_version' in data:
        _set('AZURE_OPENAI_API_VERSION', data.get('azure_openai_api_version'))
    if 'rag_rerank_backend' in data:
        _set('RAG_RERANK_BACKEND', data.get('rag_rerank_backend'))
    if 'rag_rerank_model' in data:
        _set('RAG_RERANK_MODEL', data.get('rag_rerank_model'))
    if 'rag_rerank_device' in data:
        _set('RAG_RERANK_DEVICE', data.get('rag_rerank_device'))
    if 'rag_rerank_batch_size' in data:
        _set('RAG_RERANK_BATCH_SIZE', data.get('rag_rerank_batch_size'))
    if 'rag_rerank_max_length' in data:
        _set('RAG_RERANK_MAX_LENGTH', data.get('rag_rerank_max_length'))
    if 'rag_rerank_max_chars' in data:
        _set('RAG_RERANK_MAX_CHARS', data.get('rag_rerank_max_chars'))

    if update_payload:
        runtime_settings_store.apply_updates(update_payload)
        _refresh_runtime_globals()
        _ensure_rag_embedding_defaults()
        _reset_rag_rerank_cache()
    after = runtime_settings_store.snapshot()
    runtime = _rt()

    if any(before.get(key) != after.get(key) for key in ('LMSTUDIO_API_BASE', 'LMSTUDIO_MODEL', 'LMSTUDIO_API_KEY', 'LM_DEFAULT_PROVIDER')):
        _invalidate_llm_cache()

    configure_llm_cache(
        enabled=runtime.llm_cache_enabled,
        max_items=runtime.llm_cache_max_items,
        ttl_seconds=runtime.llm_cache_ttl_seconds,
    )
    configure_search_cache(
        enabled=runtime.search_cache_enabled,
        max_items=runtime.search_cache_max_items,
        ttl_seconds=runtime.search_cache_ttl_seconds,
    )
    runtime.apply_to_flask_config(app)

    collections_payload = data.get('collections')
    if isinstance(collections_payload, list):
        try:
            for entry in collections_payload:
                try:
                    cid = int(entry.get('id'))
                except Exception:
                    continue
                col = Collection.query.get(cid)
                if not col:
                    continue
                if 'searchable' in entry:
                    col.searchable = bool(entry['searchable'])
                if 'graphable' in entry:
                    col.graphable = bool(entry['graphable'])
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.warning('Не удалось обновить коллекции в настройках: %s', exc)

    aiword_payload = data.get('aiword_users')
    if aiword_payload is not None:
        try:
            AiWordAccess.query.delete()
            db.session.commit()
            added = set()
            current_user = _load_current_user()
            grant_id = current_user.id if current_user else None
            users_list = aiword_payload if isinstance(aiword_payload, list) else []
            for item in users_list:
                try:
                    uid = int(item)
                except Exception:
                    continue
                if uid in added:
                    continue
                u = User.query.get(uid)
                if not u:
                    continue
                db.session.add(AiWordAccess(user_id=uid, granted_by=grant_id))
                added.add(uid)
            db.session.commit()
            try:
                _log_user_action('aiword_access_update', 'aiword', None, detail=json.dumps(sorted(list(added))))
            except Exception:
                pass
        except Exception:
            db.session.rollback()

    _save_runtime_settings_to_disk()
    return jsonify({'ok': True})


@app.route('/api/material-types')
def api_material_types():
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Не авторизовано'}), 401
    runtime = _rt()
    items = getattr(runtime, 'material_types', None) or []
    payload = copy.deepcopy(items)
    labels: Dict[str, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get('key') or '').strip()
        if not key:
            continue
        label = str(entry.get('label') or key).strip() or key
        labels[key] = label
        for alias in entry.get('aliases') or []:
            alias_key = str(alias or '').strip()
            if alias_key and alias_key not in labels:
                labels[alias_key] = label
    return jsonify({'ok': True, 'items': payload, 'labels': labels})


@app.route('/api/facet-config', methods=['GET', 'POST'])
def api_facet_config():
    admin = _require_admin()

    def _payload() -> dict:
        return {
            'ok': True,
            'config': _facet_config_state(),
            'options': facet_service.key_options()
        }

    if request.method == 'GET':
        return jsonify(_payload())

    data = request.get_json(silent=True) or {}
    search_payload = data.get('search') if isinstance(data.get('search'), dict) else {}
    graph_payload = data.get('graph') if isinstance(data.get('graph'), dict) else {}

    def _parse_keys(value) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            cleaned = [str(v or '').strip() for v in value if str(v or '').strip()]
            return cleaned
        return []

    runtime = _rt()
    include_types = bool(search_payload.get('include_types', runtime.search_facet_include_types))
    search_keys = _parse_keys(search_payload.get('tag_keys'))
    graph_keys = _parse_keys(graph_payload.get('tag_keys'))

    runtime_settings_store.apply_updates({
        'SEARCH_FACET_INCLUDE_TYPES': include_types,
        'SEARCH_FACET_TAG_KEYS': search_keys,
        'GRAPH_FACET_TAG_KEYS': graph_keys,
    })
    _refresh_runtime_globals()
    runtime = _rt()
    runtime.apply_to_flask_config(current_app)
    _save_runtime_settings_to_disk()
    try:
        detail = json.dumps({
            'search': {'include_types': include_types, 'tag_keys': search_keys},
            'graph': {'tag_keys': graph_keys}
        }, ensure_ascii=False)
        _log_user_action(admin, 'facet_config_update', 'facet_config', None, detail)
    except Exception:
        pass
    facet_service.invalidate('facet_config_update')
    return jsonify(_payload())


@app.route('/api/facets')
def api_facets():
    """Фасеты для текущих фильтров: типы и теги (с учётом выбранных фильтров)."""
    q = request.args.get('q', '').strip()
    material_type = request.args.get('type', '').strip()
    tag_filters = request.args.getlist('tag')
    context = (request.args.get('context') or 'search').strip().lower()
    if context not in ('search', 'graph'):
        context = 'search'
    cfg_key = 'GRAPH_FACET_TAG_KEYS' if context == 'graph' else 'SEARCH_FACET_TAG_KEYS'
    allowed_cfg = current_app.config.get(cfg_key)
    if isinstance(allowed_cfg, (list, tuple)):
        allowed_keys_list = [str(v or '').strip() for v in allowed_cfg if str(v or '').strip()]
    elif allowed_cfg is None:
        allowed_keys_list = None
    else:
        allowed_keys_list = []
    include_types = bool(current_app.config.get('SEARCH_FACET_INCLUDE_TYPES', True)) if context == 'search' else False

    year_from = (request.args.get('year_from') or '').strip()
    year_to = (request.args.get('year_to') or '').strip()
    size_min = (request.args.get('size_min') or '').strip()
    size_max = (request.args.get('size_max') or '').strip()
    collection_filter = _parse_collection_param(request.args.get('collection_id'))
    allowed_scope = _current_allowed_collections()

    request_args_tuple = tuple(sorted((key, tuple(request.args.getlist(key))) for key in request.args))
    params = FacetQueryParams(
        query=q,
        material_type=material_type,
        context=context,
        include_types=include_types,
        tag_filters=tag_filters,
        collection_filter=collection_filter,
        allowed_scope=allowed_scope,
        allowed_keys_list=allowed_keys_list,
        year_from=year_from,
        year_to=year_to,
        size_min=size_min,
        size_max=size_max,
        sources={'tags': True},
        request_args=request_args_tuple,
    )

    payload = facet_service.get_facets(
        params,
        search_candidate_fn=_search_candidate_ids,
        like_filter_fn=_apply_like_filter,
    )
    return jsonify(payload)

    base_query = _apply_file_access_filter(File.query)
    try:
        base_query = base_query.join(Collection, File.collection_id == Collection.id).filter(Collection.searchable == True)
    except Exception:
        pass
    if collection_filter is not None:
        base_query = base_query.filter(File.collection_id == collection_filter)
    if material_type:
        base_query = base_query.filter(File.material_type == material_type)
    if q:
        like = f'%{q}%'
        tag_like = exists().where(and_(
            Tag.file_id == File.id,
            or_(Tag.value.ilike(like), Tag.key.ilike(like))
        ))
        filters = [
            File.title.ilike(like),
            File.author.ilike(like),
            File.keywords.ilike(like),
            File.filename.ilike(like),
            File.text_excerpt.ilike(like),
        ]
        if hasattr(File, 'abstract'):
            filters.append(File.abstract.ilike(like))
        filters.append(tag_like)
        base_query = base_query.filter(or_(*filters))
    if year_from:
        base_query = base_query.filter(File.year >= year_from)
    if year_to:
        base_query = base_query.filter(File.year <= year_to)
    if size_min:
        try:
            base_query = base_query.filter(File.size >= int(size_min))
        except Exception:
            pass
    if size_max:
        try:
            base_query = base_query.filter(File.size <= int(size_max))
        except Exception:
            pass

    types_facet: list[list] = []
    if include_types:
        types = db.session.query(File.material_type, func.count(File.id))
        types = types.filter(File.id.in_(base_query.with_entities(File.id)))
        types = types.group_by(File.material_type).all()
        types_facet = [[mt, cnt] for (mt, cnt) in types]

    base_ids_subq = base_query.with_entities(File.id).subquery()
    base_keys = [row[0] for row in db.session.query(Tag.key).filter(Tag.file_id.in_(base_ids_subq)).distinct().all()]
    selected: dict[str, list[str]] = {}
    for tf in tag_filters:
        if '=' in tf:
            k, v = tf.split('=', 1)
            selected.setdefault(k, []).append(v)
            if k not in base_keys:
                base_keys.append(k)

    if allowed_keys_set is not None:
        base_keys = [k for k in base_keys if k in allowed_keys_set or k in selected]
    for key in selected:
        if key not in base_keys:
            base_keys.append(key)

    tag_facets: dict[str, list[list]] = {}
    for key in base_keys:
        if allowed_keys_set is not None and key not in allowed_keys_set and key not in selected:
            continue
        qk = base_query
        for tf in tag_filters:
            if '=' not in tf:
                continue
            k, v = tf.split('=', 1)
            if k == key:
                continue
            tk = aliased(Tag)
            qk = qk.join(tk, tk.file_id == File.id).filter(and_(tk.key == k, tk.value.ilike(f'%{v}%')))
        ids_subq = qk.with_entities(File.id).distinct().subquery()
        rows = db.session.query(Tag.value, func.count(Tag.id)) \
            .filter(and_(Tag.file_id.in_(ids_subq), Tag.key == key)) \
            .group_by(Tag.value) \
            .order_by(func.count(Tag.id).desc()) \
            .all()
        tag_facets[key] = [[val, cnt] for (val, cnt) in rows]
        if key in selected:
            present_vals = {val for (val, _c) in tag_facets[key]}
            for v in selected[key]:
                if v not in present_vals:
                    tag_facets[key].append([v, 0])

    return jsonify({
        'types': types_facet,
        'tag_facets': tag_facets,
        'include_types': include_types,
        'allowed_keys': allowed_keys_list,
        'context': context,
    })

@app.route("/settings", methods=["GET"])
def settings_redirect():
    """Совместимость: перенаправить на новый интерфейс настроек."""
    return redirect('/app/settings')

@app.route('/settings/collections', methods=['POST'])
def settings_collections():
    # Обновляем флаги у существующих коллекций и при необходимости добавляем новую
    try:
        cols = Collection.query.all()
        # обновляем флаги
        for c in cols:
            s_key = f'search_{c.id}'
            g_key = f'graph_{c.id}'
            c.searchable = (request.form.get(s_key) == 'on')
            c.graphable = (request.form.get(g_key) == 'on')
        new_name = (request.form.get('new_name') or '').strip()
        if new_name:
            slug = _slugify(new_name)
            exists = Collection.query.filter((Collection.slug==slug) | (Collection.name==new_name)).first()
            if not exists:
                db.session.add(Collection(name=new_name, slug=slug, searchable=True, graphable=True))
        db.session.commit()
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return json_error(f'Ошибка обновления коллекций: {e}', 500)

@app.route('/admin/backup-db', methods=['POST'])
@require_admin
def backup_db():
    # Создаём и отправляем резервную копию SQLite с отметкой времени
    try:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        src = BASE_DIR / 'catalogue.db'
        if not src.exists():
            return json_error('Файл базы не найден', 404)
        bdir = BASE_DIR / 'backups'
        bdir.mkdir(exist_ok=True)
        dst = bdir / f'catalogue_{ts}.db'
        import shutil
        shutil.copy2(src, dst)
        return send_from_directory(bdir, dst.name, as_attachment=True)
    except Exception as e:
        return json_error(f'Ошибка резервного копирования: {e}', 500)

@app.route('/admin/clear-db', methods=['POST'])
@require_admin
def clear_db():
    # Опасно: удалить все записи из основных таблиц
    try:
        Tag.query.delete()
        ChangeLog.query.delete()
        File.query.delete()
        db.session.commit()
        # очищаем статические кэши (текстовые фрагменты и миниатюры)
        try:
            static_dir = Path(app.static_folder)
            txt_cache = static_dir / 'cache' / 'text_excerpts'
            if txt_cache.exists():
                for fp in txt_cache.glob('*.txt'):
                    try: fp.unlink()
                    except Exception: pass
            thumbs = static_dir / 'thumbnails'
            if thumbs.exists():
                for fp in thumbs.glob('*.png'):
                    try: fp.unlink()
                    except Exception: pass
        except Exception:
            pass
        return jsonify({"status": "ok"})
    except Exception as e:
        db.session.rollback()
        return json_error(f'Ошибка очистки: {e}', 500)

@app.route('/admin/import-db', methods=['POST'])
@require_admin
def import_db():
    """Replace current SQLite database with uploaded file.
    Creates an automatic timestamped backup of the current DB before replacing.
    """
    file = request.files.get('dbfile')
    if not file or not file.filename:
        return json_error('Файл базы не выбран', 400)
    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.db'):
        return json_error('Ожидался файл .db (SQLite)', 400)

    try:
        head = file.stream.read(16)
        if head[:15] != b'SQLite format 3':
            file.stream.seek(0)
            return json_error('Файл не похож на SQLite базу', 400)
    except Exception:
        pass
    finally:
        try:
            file.stream.seek(0)
        except Exception:
            pass

    tmp = None
    try:
        dst = BASE_DIR / 'catalogue.db'
        if dst.exists():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            bdir = BASE_DIR / 'backups'
            bdir.mkdir(exist_ok=True)
            backup_path = bdir / f'catalogue_before_import_{ts}.db'
            import shutil
            shutil.copy2(dst, backup_path)

        tmp = BASE_DIR / f'.upload_import_tmp_{os.getpid()}_{int(time.time())}.db'
        file.save(tmp)

        # Проверка схемы
        try:
            con = sqlite3.connect(str(tmp))
            cur = con.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {r[0] for r in cur.fetchall()}
            required = {'files', 'tags', 'tag_schemas', 'changelog'}
            missing = required - tables
            con.close()
            if missing:
                try:
                    tmp.unlink()
                except Exception:
                    pass
                return json_error('В файле базы нет требуемых таблиц: ' + ', '.join(sorted(missing)), 400)
        except Exception as e:
            if tmp and tmp.exists():
                tmp.unlink()
            return json_error(f'Не удалось проверить схему базы: {e}', 500)

        # Убеждаемся, что SQLAlchemy освобождает файловые дескрипторы
        try:
            db.session.close()
            db.session.remove()
        except Exception:
            pass
        try:
            db.engine.dispose()
        except Exception:
            pass

        import shutil
        shutil.move(str(tmp), str(dst))
        tmp = None

        try:
            static_dir = Path(app.static_folder)
            txt_cache = static_dir / 'cache' / 'text_excerpts'
            if txt_cache.exists():
                for fp in txt_cache.glob('*.txt'):
                    try: fp.unlink()
                    except Exception: pass
            thumbs = static_dir / 'thumbnails'
            if thumbs.exists():
                for fp in thumbs.glob('*.png'):
                    try: fp.unlink()
                    except Exception: pass
        except Exception:
            pass

        return jsonify({"status": "ok"})
    except Exception as e:
        if tmp and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return json_error(f'Ошибка импорта базы: {e}', 500)


def _task_to_dict(task: TaskRecord) -> dict:
    payload_json = {}
    if task.payload:
        try:
            payload_json = json.loads(task.payload)
        except Exception:
            payload_json = {}
    return {
        'id': task.id,
        'name': task.name,
        'status': task.status,
        'progress': float(task.progress or 0.0),
        'payload': task.payload,
        'payload_json': payload_json,
        'created_at': task.created_at.isoformat() if task.created_at else None,
        'started_at': task.started_at.isoformat() if task.started_at else None,
        'finished_at': task.finished_at.isoformat() if task.finished_at else None,
        'error': task.error,
    }


def _reset_inflight_tasks() -> int:
    """Mark tasks that were left in non-final statuses as failed on startup."""
    try:
        stuck = TaskRecord.query.filter(TaskRecord.status.in_(TASK_STUCK_STATUSES)).all()
    except Exception as exc:
        app.logger.warning(f"[tasks] failed to load stuck tasks: {exc}")
        return 0
    if not stuck:
        return 0
    now = datetime.utcnow()
    updated = 0
    for task in stuck:
        try:
            task.status = 'error'
            task.error = 'Прервано: перезапуск приложения'
            if not task.finished_at:
                task.finished_at = now
            if not task.started_at:
                task.started_at = task.created_at or now
            if task.progress is None:
                task.progress = 0.0
            updated += 1
        except Exception:
            app.logger.debug('[tasks] failed to mark task #%s as interrupted', getattr(task, 'id', '?'))
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(f"[tasks] failed to commit stuck task reset: {exc}")
        return 0
    if updated:
        app.logger.info(f"[tasks] reset {updated} stuck task(s) on startup")
    return updated


def _cleanup_old_tasks() -> None:
    cutoff = datetime.utcnow() - TASK_RETENTION_WINDOW
    try:
        deleted = TaskRecord.query.filter(
            TaskRecord.status.in_(TASK_FINAL_STATUSES),
            TaskRecord.created_at.isnot(None),
            TaskRecord.created_at < cutoff
        ).delete(synchronize_session=False)
        db.session.commit()
        if deleted:
            app.logger.info(f"[tasks] removed {deleted} tasks older than {cutoff.isoformat()} (final statuses)")
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(f"[tasks] cleanup failed: {exc}")


@admin_bp.route('/tasks', methods=['GET', 'DELETE'])
@require_admin
def api_admin_tasks():
    if request.method == 'DELETE':
        payload = request.get_json(silent=True) or {}
        status_key = str(request.args.get('status') or payload.get('status') or 'final').strip().lower()
        before_raw = str(request.args.get('before') or payload.get('before') or '').strip()
        status_map: dict[str, tuple[str, ...] | None] = {
            'final': TASK_FINAL_STATUSES,
            'done': TASK_FINAL_STATUSES,
            'completed': ('completed',),
            'complete': ('completed',),
            'errors': ('error',),
            'error': ('error',),
            'failed': ('error',),
            'cancelled': ('cancelled',),
            'canceled': ('cancelled',),
            'all': None,
            'any': None,
        }
        statuses = status_map.get(status_key or 'final')
        if statuses is None and status_key not in ('all', 'any'):
            return jsonify({'ok': False, 'error': 'unsupported_status'}), 400
        cutoff: datetime | None = None
        if before_raw:
            try:
                if len(before_raw) <= 10:
                    cutoff = datetime.strptime(before_raw, '%Y-%m-%d') + timedelta(days=1)
                else:
                    cutoff = datetime.fromisoformat(before_raw)
            except Exception:
                return jsonify({'ok': False, 'error': 'invalid_datetime'}), 400
        query = TaskRecord.query
        if statuses:
            query = query.filter(TaskRecord.status.in_(statuses))
        elif status_key not in ('all', 'any'):
            query = query.filter(TaskRecord.status.in_(TASK_FINAL_STATUSES))
        if cutoff is not None:
            query = query.filter(TaskRecord.created_at.isnot(None)).filter(TaskRecord.created_at < cutoff)
        try:
            deleted = query.delete(synchronize_session=False)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            app.logger.warning(f"[tasks] bulk delete failed: {exc}")
            return jsonify({'ok': False, 'error': 'delete_failed'}), 500
        actor = _load_current_user()
        detail_payload = {'deleted': deleted, 'status': status_key or 'final'}
        if cutoff is not None:
            detail_payload['before'] = cutoff.isoformat()
        try:
            _log_user_action(actor, 'task_cleanup', 'task', None, detail=json.dumps(detail_payload))
        except Exception:
            pass
        app.logger.info(f"[tasks] bulk delete {deleted} records (status={status_key or 'final'}, before={cutoff.isoformat() if cutoff else '—'})")
        return jsonify({'ok': True, 'deleted': deleted})
    try:
        limit = int(request.args.get('limit', '50'))
    except Exception:
        limit = 50
    limit = max(1, min(limit, 200))
    _cleanup_old_tasks()
    cutoff = datetime.utcnow() - TASK_RETENTION_WINDOW
    tasks = TaskRecord.query.order_by(TaskRecord.created_at.desc()) \
        .filter(
            or_(
                TaskRecord.status.notin_(TASK_FINAL_STATUSES),
                TaskRecord.status.is_(None),
                TaskRecord.created_at.is_(None),
                TaskRecord.created_at >= cutoff
            )
        ).limit(limit).all()
    payload = [_task_to_dict(t) for t in tasks]
    if SCAN_PROGRESS.get('running'):
        progress = 0.0
        total = float(SCAN_PROGRESS.get('total') or 0)
        processed = float(SCAN_PROGRESS.get('processed') or 0)
        if total > 0:
            progress = min(1.0, processed / total)
        active_id = SCAN_TASK_ID or 0
        if not any(t.get('id') == active_id for t in payload):
            payload.insert(0, {
                'id': active_id,
                'name': 'scan',
                'status': 'running',
                'progress': progress,
                'payload': json.dumps({
                    'stage': SCAN_PROGRESS.get('stage'),
                    'current': SCAN_PROGRESS.get('current'),
                }),
                'created_at': None,
                'started_at': None,
                'finished_at': None,
                'error': SCAN_PROGRESS.get('error'),
            })
    return jsonify({'ok': True, 'tasks': payload})


@admin_bp.route('/tasks/<int:task_id>', methods=['PATCH', 'DELETE'])
@require_admin
def api_admin_task_detail(task_id: int):
    task = TaskRecord.query.get_or_404(task_id)
    if request.method == 'DELETE':
        db.session.delete(task)
        db.session.commit()
        return jsonify({'ok': True})
    data = request.get_json(silent=True) or {}
    status = str(data.get('status') or '').strip().lower()
    if status == 'cancel':
        if task.name == 'scan' and SCAN_PROGRESS.get('running') and SCAN_TASK_ID == task.id:
            scan_cancel()
            return jsonify({'ok': True, 'task': _task_to_dict(task)})
        task.status = 'cancelled'
        db.session.commit()
        return jsonify({'ok': True, 'task': _task_to_dict(task)})
    if status:
        task.status = status
    progress = data.get('progress')
    if progress is not None:
        try:
            task.progress = float(progress)
        except Exception:
            pass
    db.session.commit()
    return jsonify({'ok': True, 'task': _task_to_dict(task)})


@admin_bp.route('/system-logs', methods=['GET', 'DELETE'])
@require_admin
def api_admin_system_logs():
    if request.method == 'DELETE':
        payload = request.get_json(silent=True) or {}
        name = str(request.args.get('name') or payload.get('name') or LOG_FILE_PATH.name).strip()
        path = _resolve_log_name(name)
        if not path:
            return jsonify({'ok': False, 'error': 'log_not_found'}), 404
        try:
            with path.open('w', encoding='utf-8') as fh:
                fh.truncate(0)
        except Exception as exc:
            return jsonify({'ok': False, 'error': str(exc)}), 500
        handler = _get_rotating_log_handler()
        if handler and Path(getattr(handler, 'baseFilename', '')).resolve() == path.resolve():
            handler.acquire()
            try:
                handler.flush()
                stream = getattr(handler, 'stream', None)
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass
                handler.stream = handler._open()
            finally:
                handler.release()
        actor = _load_current_user()
        try:
            detail = json.dumps({'name': path.name, 'action': 'truncate'})
        except Exception:
            detail = None
        _log_user_action(actor, 'system_log_clear', 'log', None, detail=detail)
        app.logger.info(f"[logs] {getattr(actor, 'username', 'admin')} truncated log {path.name}")
        return jsonify({'ok': True, 'files': _list_system_log_files()})

    try:
        limit = int(request.args.get('limit', '200'))
    except Exception:
        limit = 200
    limit = max(10, min(limit, 2000))
    name = str(request.args.get('name') or '').strip()
    path = _resolve_log_name(name) or LOG_FILE_PATH
    if not path.exists():
        return jsonify({'ok': False, 'error': 'log_not_found'}), 404
    available_files = _list_system_log_files()
    selected = next((entry for entry in available_files if entry.get('name') == path.name), None)
    if not selected:
        try:
            stat = path.stat()
            selected = {
                'name': path.name,
                'size': stat.st_size,
                'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'rotated': path.name != LOG_FILE_PATH.name,
            }
        except Exception:
            selected = {'name': path.name, 'rotated': path.name != LOG_FILE_PATH.name}
    lines = _tail_log_file(path, limit)
    return jsonify({'ok': True, 'file': selected, 'lines': lines, 'available': available_files, 'limit': limit})


@admin_bp.route('/system-logs/rotate', methods=['POST'])
@require_admin
def api_admin_system_logs_rotate():
    handler = _get_rotating_log_handler()
    if not handler:
        return jsonify({'ok': False, 'error': 'rotation_unavailable'}), 400
    handler.acquire()
    try:
        handler.flush()
        handler.doRollover()
        if handler.stream is None:
            handler.stream = handler._open()
    except Exception as exc:
        handler.release()
        app.logger.warning(f"[logs] manual rotation failed: {exc}")
        return jsonify({'ok': False, 'error': str(exc)}), 500
    handler.release()
    actor = _load_current_user()
    try:
        detail = json.dumps({'name': LOG_FILE_PATH.name, 'action': 'rotate'})
    except Exception:
        detail = None
    _log_user_action(actor, 'system_log_rotate', 'log', None, detail=detail)
    app.logger.info(f"[logs] {getattr(actor, 'username', 'admin')} rotated system log")
    return jsonify({'ok': True, 'files': _list_system_log_files()})


@admin_bp.route('/system-logs/download', methods=['GET'])
@require_admin
def api_admin_system_logs_download():
    name = str(request.args.get('name') or LOG_FILE_PATH.name).strip()
    path = _resolve_log_name(name)
    if not path:
        abort(404)
    return send_file(path, mimetype='text/plain', as_attachment=True, download_name=path.name, conditional=True)


@admin_bp.route('/actions', methods=['GET', 'DELETE'])
@require_admin
def api_admin_actions():
    if request.method == 'DELETE':
        payload = request.get_json(silent=True) or {}
        before_raw = str(request.args.get('before') or payload.get('before') or '').strip()
        if not before_raw:
            return jsonify({'ok': False, 'error': 'parameter "before" is required'}), 400
        try:
            if len(before_raw) <= 10:
                before_dt = datetime.strptime(before_raw, '%Y-%m-%d') + timedelta(days=1)
            else:
                before_dt = datetime.fromisoformat(before_raw)
        except Exception:
            return jsonify({'ok': False, 'error': 'invalid datetime format'}), 400
        try:
            deleted = UserActionLog.query.filter(UserActionLog.created_at < before_dt).delete(synchronize_session=False)
            db.session.commit()
            app.logger.info(f"[actions] deleted {deleted} log records older than {before_dt.isoformat()}")
            return jsonify({'ok': True, 'deleted': deleted})
        except Exception as exc:
            db.session.rollback()
            app.logger.warning(f"[actions] failed to delete logs: {exc}")
            return jsonify({'ok': False, 'error': 'delete_failed'}), 500

    try:
        limit = int(request.args.get('limit', '100'))
    except Exception:
        limit = 100
    limit = max(1, min(limit, 500))
    q = UserActionLog.query.order_by(UserActionLog.created_at.desc())
    user_id = request.args.get('user_id')
    if user_id:
        try:
            q = q.filter(UserActionLog.user_id == int(user_id))
        except Exception:
            pass
    action = request.args.get('action')
    if action:
        q = q.filter(UserActionLog.action == action)
    logs = q.limit(limit).all()
    return jsonify({'ok': True, 'actions': [
        {
            'id': log.id,
            'user_id': log.user_id,
            'username': log.user.username if getattr(log, 'user', None) else None,
            'full_name': log.user.full_name if getattr(log, 'user', None) else None,
            'action': log.action,
            'entity': log.entity,
            'entity_id': log.entity_id,
            'detail': log.detail,
            'created_at': log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]})


@admin_bp.route('/llm-endpoints', methods=['GET', 'POST'])
@require_admin
def api_admin_llm_endpoints():
    _ensure_llm_schema_once()
    if request.method == 'GET':
        eps = LlmEndpoint.query.order_by(LlmEndpoint.created_at.desc()).all()
        return jsonify({'ok': True, 'items': [
            {
                'id': ep.id,
                'name': ep.name,
                'base_url': ep.base_url,
                'model': ep.model,
                'weight': float(ep.weight or 0.0),
                'purpose': ep.purpose,
                'purposes': _llm_parse_purposes(ep.purpose),
                'provider': (ep.provider or _rt().lm_default_provider),
                'created_at': ep.created_at.isoformat() if ep.created_at else None,
            }
            for ep in eps
        ], 'purposes_catalog': LLM_PURPOSES, 'providers_catalog': LLM_PROVIDER_OPTIONS})
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    base_url = (data.get('base_url') or '').strip()
    model = (data.get('model') or '').strip()
    provider = (data.get('provider') or _rt().lm_default_provider or 'openai').strip().lower()
    if provider not in LLM_PROVIDER_CHOICES:
        return jsonify({'ok': False, 'error': 'некорректный провайдер LLM'}), 400
    if not name or not base_url or not model:
        return jsonify({'ok': False, 'error': 'name, base_url и model обязательны'}), 400
    normalized_purpose = _llm_normalize_purposes(data.get('purposes') if 'purposes' in data else data.get('purpose'))
    ep = LlmEndpoint(
        name=name,
        base_url=base_url,
        model=model,
        api_key=(data.get('api_key') or '').strip() or None,
        weight=float(data.get('weight') or 1.0),
        purpose=normalized_purpose,
        provider=provider,
    )
    db.session.add(ep)
    db.session.commit()
    _invalidate_llm_cache()
    _log_user_action(_load_current_user(), 'llm_add', 'llm_endpoint', ep.id, detail=json.dumps({'name': name, 'purpose': ep.purpose}))
    return jsonify({'ok': True, 'item': {
        'id': ep.id,
        'name': ep.name,
        'base_url': ep.base_url,
        'model': ep.model,
        'weight': float(ep.weight or 0.0),
        'purpose': ep.purpose,
        'purposes': _llm_parse_purposes(ep.purpose),
        'provider': ep.provider,
        'created_at': ep.created_at.isoformat() if ep.created_at else None,
    }}), 201


@admin_bp.route('/llm-endpoints/<int:endpoint_id>', methods=['PATCH', 'DELETE'])
@require_admin
def api_admin_llm_endpoint_detail(endpoint_id: int):
    _ensure_llm_schema_once()
    ep = LlmEndpoint.query.get_or_404(endpoint_id)
    if request.method == 'DELETE':
        db.session.delete(ep)
        db.session.commit()
        _invalidate_llm_cache()
        _log_user_action(_load_current_user(), 'llm_delete', 'llm_endpoint', endpoint_id)
        return jsonify({'ok': True})
    data = request.get_json(silent=True) or {}
    updated = False
    if 'name' in data:
        ep.name = str(data['name']).strip() or ep.name
        updated = True
    if 'base_url' in data:
        ep.base_url = str(data['base_url']).strip() or ep.base_url
        updated = True
    if 'model' in data:
        ep.model = str(data['model']).strip() or ep.model
        updated = True
    if 'api_key' in data:
        ep.api_key = str(data['api_key']).strip() or None
        updated = True
    if 'weight' in data:
        try:
            ep.weight = float(data['weight'])
            updated = True
        except Exception:
            pass
    if 'purposes' in data:
        ep.purpose = _llm_normalize_purposes(data.get('purposes'))
        updated = True
    elif 'purpose' in data:
        ep.purpose = _llm_normalize_purposes(data.get('purpose'))
        updated = True
    if 'provider' in data:
        prov = str(data['provider']).strip().lower()
        if prov in LLM_PROVIDER_CHOICES:
            ep.provider = prov
            updated = True
        else:
            return jsonify({'ok': False, 'error': 'некорректный провайдер LLM'}), 400
    if updated:
        db.session.commit()
        _invalidate_llm_cache()
        _log_user_action(_load_current_user(), 'llm_update', 'llm_endpoint', ep.id)
    return jsonify({'ok': True, 'item': {
        'id': ep.id,
        'name': ep.name,
        'base_url': ep.base_url,
        'model': ep.model,
        'weight': float(ep.weight or 0.0),
        'purpose': ep.purpose,
        'purposes': _llm_parse_purposes(ep.purpose),
        'provider': ep.provider,
        'created_at': ep.created_at.isoformat() if ep.created_at else None,
    }})


# ------------------- CLI helpers -------------------

# ------------------- Diagnostics -------------------
@app.route('/diagnostics/transcribe')
def diag_transcribe():
    """Диагностика транскрибации одного файла.
    Параметры:
      - path: абсолютный или относительный к SCAN_ROOT путь
      - file_id: альтернатива path, взять файл из БД по id
      - limit: ограничение символов транскрипта (по умолчанию 1000)
    Возвращает JSON: состояние бэкенда/модели, сведения о файле и образец транскрипта.
    """
    res = {
        "transcribe_enabled": bool(TRANSCRIBE_ENABLED),
        "backend": (TRANSCRIBE_BACKEND or '').lower(),
        "language": TRANSCRIBE_LANGUAGE,
        "model_path": TRANSCRIBE_MODEL_PATH,
        "model_exists": False,
        "ffmpeg": _ffmpeg_available(),
        "ffprobe": shutil.which('ffprobe') is not None,
        "file": {},
        "transcript": {
            "ok": False,
            "len": 0,
            "sample": "",
        },
        "warnings": [],
        "faster_whisper": {},
        "applied": {},
    }
    try:
        mp = TRANSCRIBE_MODEL_PATH or ''
        try:
            res["model_exists"] = bool(mp) and Path(mp).exists()
        except Exception:
            res["model_exists"] = False

        # Детали разрешения модели faster-whisper (опционально)
        try:
            fw = {
                "hf_available": bool(hf_snapshot_download),
                "cache_dir": str(FW_CACHE_DIR),
                "model_ref": (request.args.get('model') or TRANSCRIBE_MODEL_PATH or '').strip(),
                "repo": None,
                "target_dir": None,
                "target_exists": None,
                "downloaded": False,
            }
            mref = fw["model_ref"]
            if mref:
                # Если локальный каталог существует, считаем путь найденным
                p = Path(mref).expanduser()
                if p.exists() and p.is_dir():
                    fw["target_dir"] = str(p)
                    fw["target_exists"] = True
                else:
                    repo = _fw_alias_to_repo(mref) or (mref if '/' in mref else None)
                    fw["repo"] = repo
                    if repo:
                        safe_name = repo.replace('/', '__')
                        target_dir = FW_CACHE_DIR / safe_name
                        fw["target_dir"] = str(target_dir)
                        fw["target_exists"] = target_dir.exists() and any(target_dir.iterdir())
                        # Необязательная загрузка, даже если бэкенд сейчас не faster-whisper
                        if (request.args.get('download') or '').lower() in ('1','true','yes','on'):
                            if hf_snapshot_download is None:
                                res["warnings"].append("Пакет huggingface_hub не установлен — автозагрузка невозможна")
                            else:
                                FW_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                                hf_snapshot_download(repo_id=repo, local_dir=str(target_dir), local_dir_use_symlinks=False, revision=None)
                                fw["downloaded"] = True
                                fw["target_exists"] = target_dir.exists() and any(target_dir.iterdir())
                    else:
                        res["warnings"].append("Неизвестная ссылка на модель faster-whisper (не каталог, не алиас и не repo id)")
            res["faster_whisper"] = fw
        except Exception as e:
            res["warnings"].append(f"fw_resolve:{e}")

        # Переопределения (backend/lang/vad/model) из query-параметров
        eff_backend = (request.args.get('backend') or '').strip().lower() or None
        eff_lang = (request.args.get('lang') or '').strip() or None
        vad_param = (request.args.get('vad') or '').strip().lower()
        if vad_param in ('1','true','yes','on'): eff_vad = True
        elif vad_param in ('0','false','no','off'): eff_vad = False
        else: eff_vad = None
        model_override = (request.args.get('model') or '').strip() or None
        res["applied"] = {"backend": eff_backend, "lang": eff_lang, "vad": eff_vad, "model": model_override}

        # Нормализация backend: поддерживаем только faster-whisper
        if eff_backend and eff_backend != 'faster-whisper':
            res["warnings"].append("Бэкенд не поддерживается (Vosk удалён); используются настройки faster-whisper")
            eff_backend = None

        # Разрешение пути к файлу
        p = None
        q_path = (request.args.get('path') or '').strip()
        q_id = (request.args.get('file_id') or '').strip()
        if q_path:
            pp = Path(q_path)
            if not pp.is_absolute():
                root = _scan_root_path()
                # Пробуем типовые варианты, чтобы избежать дублирования корневой папки в пути
                candidates = [
                    root / q_path,
                    root.parent / q_path,
                ]
                picked = None
                for cand in candidates:
                    try:
                        if cand.exists():
                            picked = cand
                            break
                    except Exception:
                        pass
                pp = picked or (root / q_path)
            p = pp
        elif q_id:
            try:
                f = File.query.get(int(q_id))
                if f: p = Path(f.path)
            except Exception:
                p = None
        # Разрешить диагностику только модели (без файла)
        if not p:
            return jsonify(res)

        # метаданные файла
        file_info = {
            "requested": q_path or q_id,
            "path": str(p) if p else None,
            "exists": bool(p and p.exists()),
            "ext": (p.suffix.lower() if p else None),
            "size": (p.stat().st_size if p and p.exists() else None),
        }
        if p and p.exists():
            try:
                file_info["duration_seconds"] = _ffprobe_duration_seconds(p)
                file_info["duration_str"] = audio_duration_hhmmss(p)
            except Exception:
                file_info["duration_seconds"] = None
                file_info["duration_str"] = None
        res["file"] = file_info

        # Выполнить транскрибацию при наличии файла и разрешённых настройках (для аудио)
        if not TRANSCRIBE_ENABLED and eff_backend is None:
            res["warnings"].append("TRANSCRIBE_ENABLED is off")
            return jsonify(res)
        if not (p and p.exists() and p.is_file()):
            return jsonify(res)

        limit = 0
        try:
            limit = int(request.args.get('limit', '1000'))
        except Exception:
            limit = 1000
        limit = max(200, min(20000, limit or 1000))

        try:
            if p.suffix.lower() in AUDIO_EXTS:
                tx = transcribe_audio(p, limit_chars=limit,
                                      backend_override=eff_backend,
                                      model_path_override=model_override,
                                      lang_override=eff_lang,
                                      vad_override=eff_vad)
                sample = (tx or '')[:200]
                res["transcript"].update({"ok": bool(tx), "len": len(tx or ''), "sample": sample})
            elif p.suffix.lower() in IMAGE_EXTS and IMAGES_VISION_ENABLED:
                vis = call_lmstudio_vision(p, p.name)
                desc = (vis.get('description') or '') if isinstance(vis, dict) else ''
                sample = desc[:200]
                res["transcript"].update({"ok": bool(sample), "len": len(desc), "sample": sample})
            else:
                res["warnings"].append("Файл не аудио/изображение или распознавание изображений выключено")
        except Exception as e:
            res["warnings"].append(f"transcribe_error:{e}")
        return jsonify(res)
    except Exception as e:
        res["warnings"].append(f"unexpected:{e}")
    return jsonify(res), 500

@admin_bp.route('/ai-search/metrics', methods=['GET', 'DELETE'])
@require_admin
def api_admin_ai_metrics():
    if request.method == 'DELETE':
        try:
            deleted = AiSearchMetric.query.delete(synchronize_session=False)
            db.session.commit()
            return jsonify({'ok': True, 'deleted': deleted or 0})
        except Exception as exc:
            db.session.rollback()
            app.logger.warning(f"[ai-metrics] cleanup failed: {exc}")
            return jsonify({'ok': False, 'error': 'Не удалось очистить метрики'}), 500

    try:
        limit = int(request.args.get('limit', '100'))
    except Exception:
        limit = 100
    limit = max(1, min(limit, 500))
    rows = AiSearchMetric.query.order_by(AiSearchMetric.created_at.desc()).limit(limit).all()
    payload = []
    for row in rows:
        user_obj = row.user
        payload.append({
            'id': row.id,
            'query_hash': row.query_hash,
            'user_id': row.user_id,
             'user_username': getattr(user_obj, 'username', None),
             'user_full_name': getattr(user_obj, 'full_name', None),
            'total_ms': row.total_ms,
            'keywords_ms': row.keywords_ms,
            'candidate_ms': row.candidate_ms,
            'deep_ms': row.deep_ms,
            'llm_answer_ms': row.llm_answer_ms,
            'llm_snippet_ms': row.llm_snippet_ms,
            'created_at': row.created_at.isoformat() if row.created_at else None,
            'meta': row.meta,
        })
    summary = {}
    if rows:
        fields = ['total_ms', 'keywords_ms', 'candidate_ms', 'deep_ms', 'llm_answer_ms', 'llm_snippet_ms']
        for field in fields:
            vals = [getattr(r, field) for r in rows if getattr(r, field) is not None]
            if vals:
                summary[field] = sum(vals) / len(vals)
    return jsonify({'ok': True, 'items': payload, 'summary': summary})

# ------------------- Statistics & Visualization -------------------
from collections import Counter, defaultdict

@app.route("/api/stats")
def api_stats():
    # Агрегация по авторам, годам, типам материалов
    try:
        files = File.query.join(Collection, File.collection_id == Collection.id).filter(Collection.searchable == True).all()
    except Exception:
        files = File.query.all()
    authors = Counter()
    years = Counter()
    types = Counter()
    exts = Counter()
    size_buckets = Counter()
    months = Counter()
    # новые распределения
    weekdays = Counter()  # 0..6 (Mon..Sun)
    hours = Counter()     # 0..23
    # дополнительные агрегации
    kw = Counter()
    tag_keys = Counter()
    tag_values = Counter()
    total_files = 0
    total_size = 0
    # средний размер по типам
    size_sum_by_type = Counter()
    size_cnt_by_type = Counter()
    # заполненность ключевых полей
    meta_presence = Counter()
    meta_missing = Counter()
    # дополнительные показатели
    tags_total = 0
    tagless_files = 0
    now = datetime.utcnow()
    recent_7_cut = now - timedelta(days=7)
    recent_30_cut = now - timedelta(days=30)
    recent_counts = {'7d': 0, '30d': 0}
    def bucket_size(sz):
        if sz is None or sz <= 0:
            return "неизв."
        mb = sz / (1024*1024)
        if mb < 1: return "< 1 МБ"
        if mb < 10: return "1–10 МБ"
        if mb < 50: return "10–50 МБ"
        if mb < 100: return "50–100 МБ"
        return "> 100 МБ"
    for f in files:
        total_files += 1
        try:
            total_size += int(f.size or 0)
        except Exception:
            pass
        if f.author:
            authors[f.author] += 1
        if f.year:
            years[f.year] += 1
        if f.material_type:
            types[f.material_type] += 1
        if f.ext:
            exts[f.ext.lower().lstrip('.')] += 1
        size_buckets[bucket_size(f.size or 0)] += 1
        if f.mtime:
            try:
                d = datetime.fromtimestamp(f.mtime)
                months[d.strftime('%Y-%m')] += 1
                weekdays[d.weekday()] += 1  # Понедельник=0
                hours[d.hour] += 1
                if d >= recent_30_cut:
                    recent_counts['30d'] += 1
                if d >= recent_7_cut:
                    recent_counts['7d'] += 1
            except Exception:
                pass
        # ключевые слова
        if f.keywords:
            for part in re.split(r"[\n,;]+", f.keywords):
                w = (part or '').strip()
                if w:
                    kw[w] += 1
        # ключи тегов
        tag_count = 0
        try:
            for t in f.tags:
                tag_count += 1
                if t.key:
                    tag_keys[t.key] += 1
                if t.value:
                    tag_values[str(t.value).strip()] += 1
        except Exception:
            tag_count = 0
        tags_total += tag_count
        # средний размер по типам
        if f.material_type and (f.size or 0) > 0:
            size_sum_by_type[f.material_type] += int(f.size or 0)
            size_cnt_by_type[f.material_type] += 1
        # заполненность полей
        if f.title:
            meta_presence['Название'] += 1
        else:
            meta_missing['Название'] += 1
        if f.author:
            meta_presence['Автор'] += 1
        else:
            meta_missing['Автор'] += 1
        if f.year:
            meta_presence['Год'] += 1
        else:
            meta_missing['Год'] += 1
        if f.keywords:
            meta_presence['Ключевые слова'] += 1
        else:
            meta_missing['Ключевые слова'] += 1
        if tag_count > 0:
            meta_presence['Теги'] += 1
        else:
            meta_missing['Теги'] += 1
            tagless_files += 1
    # подготовка выходных структур
    # средний размер по типам (в МБ, округляем до десятых)
    avg_size_type = []
    for mt in size_sum_by_type.keys():
        cnt = max(1, size_cnt_by_type[mt])
        avg_mb = (size_sum_by_type[mt] / cnt) / (1024*1024)
        avg_size_type.append((mt, round(avg_mb, 1)))
    avg_size_type.sort(key=lambda x: x[1], reverse=True)
    # недели: упорядочим Пн..Вс
    weekday_names = ['Пн','Вт','Ср','Чт','Пт','Сб','Вс']
    weekdays_list = []
    for i in range(7):
        weekdays_list.append((weekday_names[i], int(weekdays.get(i, 0))))
    # часы 0..23
    hours_list = [(str(h), int(hours.get(h, 0))) for h in range(24)]
    # коллекции: количество файлов по коллекциям (searchable)
    collections_counts = []
    try:
        q = db.session.query(Collection.name, func.count(File.id)) \
            .join(File, File.collection_id == Collection.id) \
            .filter(Collection.searchable == True) \
            .group_by(Collection.id) \
            .order_by(func.count(File.id).desc())
        collections_counts = [(name, int(cnt)) for (name, cnt) in q.all()]
    except Exception:
        pass
    # общий размер по коллекциям (только searchable)
    collections_total_size = []
    try:
        rows = db.session.query(Collection.name, func.coalesce(func.sum(File.size), 0)) \
            .join(File, File.collection_id == Collection.id) \
            .filter(Collection.searchable == True) \
            .group_by(Collection.id) \
            .order_by(func.coalesce(func.sum(File.size), 0).desc()) \
            .all()
        collections_total_size = [(name, int(total or 0)) for (name, total) in rows]
    except Exception:
        pass

    # самые крупные файлы (только searchable)
    largest_files = []
    try:
        rows = db.session.query(File.filename, File.size) \
            .join(Collection, File.collection_id == Collection.id) \
            .filter(Collection.searchable == True) \
            .order_by(File.size.desc().nullslast()) \
            .limit(15).all()
        largest_files = [(fn or '(без имени)', int(sz or 0)) for (fn, sz) in rows]
    except Exception:
        pass

    tags_summary = {
        'avg_per_file': round(tags_total / total_files, 2) if total_files else 0.0,
        'with_tags': max(total_files - tagless_files, 0),
        'without_tags': tagless_files,
        'total_tags': tags_total,
    }

    return jsonify({
        "authors": authors.most_common(20),
        "authors_cloud": authors.most_common(100),
        "years": sorted(years.items()),
        "types": types.most_common(),
        "exts": exts.most_common(),
        "sizes": sorted(size_buckets.items(), key=lambda x: ["неизв.","< 1 МБ","1–10 МБ","10–50 МБ","50–100 МБ","> 100 МБ"].index(x[0]) if x[0] in ["неизв.","< 1 МБ","1–10 МБ","10–50 МБ","50–100 МБ","> 100 МБ"] else 999),
        "months": sorted(months.items()),
        "top_keywords": kw.most_common(30),
        "tag_keys": tag_keys.most_common(30),
        "tag_values_cloud": tag_values.most_common(150),
        "weekdays": weekdays_list,
        "hours": hours_list,
        "avg_size_type": avg_size_type,
        "meta_presence": sorted(meta_presence.items(), key=lambda x: x[0]),
        "meta_missing": sorted(meta_missing.items(), key=lambda x: x[0]),
        "total_files": total_files,
        "total_size_bytes": total_size,
        "collections_counts": collections_counts,
        "collections_total_size": collections_total_size,
        "largest_files": largest_files,
        "recent_counts": {
            "7d": int(recent_counts['7d']),
            "30d": int(recent_counts['30d']),
        },
        "tags_summary": tags_summary,
    })

@app.route('/api/stats/tag-values')
def api_stats_tag_values():
    """Возвращает топ значений для заданного ключа тега.
    Query: key, limit=30
    Учитывает только коллекции с searchable=true.
    """
    key = (request.args.get('key') or '').strip()
    if not key:
        return jsonify({"items": []})
    try:
        try:
            lim = min(max(int(request.args.get('limit', '30')), 1), 200)
        except Exception:
            lim = 30
        rows = db.session.query(Tag.value, func.count(Tag.id)) \
            .join(File, File.id == Tag.file_id) \
            .join(Collection, File.collection_id == Collection.id) \
            .filter(Collection.searchable == True) \
            .filter(Tag.key == key) \
            .group_by(Tag.value) \
            .order_by(func.count(Tag.id).desc()) \
            .limit(lim).all()
        return jsonify({"items": [[v or '', int(c or 0)] for (v, c) in rows]})
    except Exception as e:
        return jsonify({"items": [], "error": str(e)}), 500

@app.route("/stats")
def stats_redirect():
    return redirect('/app/stats')


@app.route('/graph')
def graph_redirect():
    return redirect('/app/graph')


def _read_text_file_with_fallback(path: Path) -> str:
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_text_for_rag(path: Path, *, limit_chars: int = 120_000) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext == "pdf":
        return extract_text_pdf(path, limit_chars=limit_chars, force_ocr_first_page=False)
    if ext == "docx":
        return extract_text_docx(path, limit_chars=limit_chars)
    if ext == "rtf":
        return extract_text_rtf(path, limit_chars=limit_chars)
    if ext == "epub":
        return extract_text_epub(path, limit_chars=limit_chars)
    if ext == "djvu":
        return extract_text_djvu(path, limit_chars=limit_chars)
    if ext in AUDIO_EXTS:
        return transcribe_audio(path, limit_chars=limit_chars)
    if ext in {"txt", "md", "markdown"}:
        return _read_text_file_with_fallback(path)
    try:
        return _read_text_file_with_fallback(path)
    except Exception:
        return ""


def _resolve_candidate_paths(file_obj: File) -> List[Path]:
    candidates: List[Path] = []
    if file_obj.path:
        candidates.append(Path(file_obj.path))
    if file_obj.rel_path:
        rel = Path(file_obj.rel_path)
        candidates.append(rel)
        try:
            scan_root = runtime_settings_store.current.scan_root
            candidates.append(scan_root / rel)
        except Exception:
            pass
        candidates.append(Path("library") / rel)
    seen: set[str] = set()
    resolved: List[Path] = []
    for candidate in candidates:
        try:
            normalized = candidate.expanduser()
        except Exception:
            normalized = candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(normalized)
    return resolved


def _collect_text_for_rag(file_obj: File, *, limit_chars: int = 120_000) -> tuple[str, Optional[Path]]:
    """Возвращает текст для RAG и путь к исходному файлу (если удалось определить)."""
    if not file_obj:
        return "", None
    last_error: Exception | None = None
    for candidate in _resolve_candidate_paths(file_obj):
        try:
            if not candidate.exists() or not candidate.is_file():
                continue
            text = _extract_text_for_rag(candidate, limit_chars=limit_chars)
            if text and text.strip():
                return text, candidate
        except Exception as exc:
            last_error = exc
    fallback = (file_obj.text_excerpt or "").strip()
    if fallback:
        return fallback, None
    if last_error:
        raise RuntimeError(f"Не удалось извлечь текст: {last_error}") from last_error
    return "", None


def _embed_missing_chunks_for_document(
    document_id: int,
    backend: EmbeddingBackend,
    *,
    batch_size: int = 32,
    commit: bool = True,
) -> int:
    """Создаёт эмбеддинги для чанков документа, у которых они отсутствуют."""
    if not document_id:
        return 0
    model_name = getattr(backend, "model_name", "unknown")
    model_version = getattr(backend, "model_version", "unknown")
    join_condition = and_(
        RagChunkEmbedding.chunk_id == RagDocumentChunk.id,
        RagChunkEmbedding.model_name == model_name,
        RagChunkEmbedding.model_version == model_version,
    )
    pending_chunks: List[RagDocumentChunk] = (
        db.session.query(RagDocumentChunk)
        .outerjoin(RagChunkEmbedding, join_condition)
        .filter(RagDocumentChunk.document_id == document_id, RagChunkEmbedding.id.is_(None))
        .order_by(RagDocumentChunk.id.asc())
        .all()
    )
    if not pending_chunks:
        return 0
    total_created = 0
    batch_size = max(1, int(batch_size or 1))
    for start in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[start : start + batch_size]
        texts = [chunk.content or "" for chunk in batch]
        vectors = backend.embed_many(texts)
        if len(vectors) != len(batch):
            raise RuntimeError("Размера ответа backend эмбеддингов недостаточно для сохранения.")
        for chunk, vector in zip(batch, vectors):
            vector_bytes = vector_to_bytes(vector)
            checksum = hashlib.sha256(vector_bytes).hexdigest()
            embedding = RagChunkEmbedding(
                chunk_id=chunk.id,
                model_name=model_name,
                model_version=model_version,
                dim=len(vector),
                vector=vector_bytes,
                vector_checksum=checksum,
            )
            db.session.add(embedding)
            total_created += 1
        if commit:
            db.session.commit()
        else:
            db.session.flush()
    return total_created


def _run_rag_collection_job(task_id: int, collection_id: int, options: dict[str, object]) -> None:
    """Фоновая задача построения RAG-индекса для коллекции."""
    try:
        db.session.rollback()
    except Exception:
        db.session.remove()

    task = TaskRecord.query.get(task_id)
    if not task:
        return

    def _update_task(status: Optional[str] = None, progress: Optional[float] = None, error: Optional[str] = None, final: bool = False, payload_override: Optional[dict] = None) -> None:
        payload = payload_override or summary
        try:
            task_ref = TaskRecord.query.get(task_id)
            if not task_ref:
                return
            if status:
                task_ref.status = status
            if progress is not None:
                task_ref.progress = max(0.0, min(1.0, float(progress)))
            task_ref.payload = json.dumps(payload, ensure_ascii=False)
            if status == 'running' and task_ref.started_at is None:
                task_ref.started_at = datetime.utcnow()
            if status in ('completed', 'error', 'cancelled'):
                task_ref.finished_at = datetime.utcnow()
            if error:
                task_ref.error = error
            db.session.commit()
        except Exception:
            db.session.rollback()

    user_id = options.get("user_id")
    try:
        runtime = runtime_settings_store.current
    except Exception:
        runtime = None

    default_backend = (getattr(runtime, "rag_embedding_backend", "auto") or "auto").strip().lower()
    default_model = getattr(runtime, "rag_embedding_model", None) or "intfloat/multilingual-e5-large"
    default_dim = max(8, int(getattr(runtime, "rag_embedding_dim", 384) or 384))
    default_batch = max(1, int(getattr(runtime, "rag_embedding_batch_size", 32) or 32))
    default_device = getattr(runtime, "rag_embedding_device", None)
    default_endpoint = getattr(runtime, "rag_embedding_endpoint", None) or getattr(runtime, "lmstudio_api_base", "")
    default_api_key = getattr(runtime, "rag_embedding_api_key", None) or getattr(runtime, "lmstudio_api_key", "")

    def _log_result(action: str, detail_payload: dict[str, object]) -> None:
        try:
            actor = User.query.get(int(user_id)) if user_id else None
        except Exception:
            actor = None
        try:
            _log_user_action(
                actor,
                action,
                "collection",
                collection_id,
                detail=json.dumps(detail_payload, ensure_ascii=False),
            )
        except Exception:
            pass

    try:
        collection = Collection.query.get(collection_id)
        if not collection:
            raise RuntimeError("Коллекция не найдена.")
    except Exception as exc:
        summary = {
            "collection_id": collection_id,
            "collection_name": None,
            "total_files": 0,
            "ingested": 0,
            "skipped": 0,
            "embedded": 0,
            "failures": [
                {"reason": "collection_missing", "message": str(exc)},
            ],
            "status": "error",
        }
        _update_task(status='error', progress=0.0, error=str(exc), final=True, payload_override=summary)
        _log_result(
            "collection_rag_reindex_failed",
            {"reason": "collection_missing", "message": str(exc)},
        )
        return

    chunk_config = ChunkConfig(
        max_tokens=max(16, int(options.get("chunk_max_tokens", 700))),
        overlap=max(0, int(options.get("chunk_overlap", 120))),
        min_tokens=max(1, int(options.get("chunk_min_tokens", 80))),
    )
    skip_if_unchanged = bool(options.get("skip_if_unchanged", True))
    limit_chars = max(1000, int(options.get("limit_chars", 120_000)))
    embedding_backend_name = (
        str(options.get("embedding_backend") or default_backend or "auto").strip().lower() or default_backend or "auto"
    )
    embedding_model_name = str(options.get("embedding_model_name") or default_model).strip() or default_model
    embedding_dim = max(8, int(options.get("embedding_dim", default_dim)))
    embedding_batch = max(1, int(options.get("embedding_batch_size", default_batch)))
    embedding_normalize = bool(options.get("embedding_normalize", True))
    embedding_device = options.get("embedding_device")
    if not embedding_device and default_device:
        embedding_device = default_device
    embedding_endpoint = str(options.get("embedding_endpoint") or default_endpoint or "").strip()
    embedding_api_key = str(options.get("embedding_api_key") or default_api_key or "")

    try:
        files = (
            File.query.filter(File.collection_id == collection_id)
            .order_by(File.id.asc())
            .all()
        )
    except Exception as exc:
        summary = {
            "collection_id": collection_id,
            "collection_name": collection.name,
            "total_files": 0,
            "ingested": 0,
            "skipped": 0,
            "embedded": 0,
            "failures": [
                {"reason": "files_query_failed", "message": str(exc)},
            ],
            "status": "error",
        }
        _update_task(status='error', progress=0.0, error=str(exc), final=True, payload_override=summary)
        _log_result(
            "collection_rag_reindex_failed",
            {"reason": "files_query_failed", "message": str(exc)},
        )
        return

    indexer = RagIndexer(chunk_config=chunk_config, normalizer_version=str(options.get("normalizer_version") or "v1"))

    try:
        backend = load_embedding_backend(
            embedding_backend_name,
            model_name=embedding_model_name,
            dim=embedding_dim,
            normalize=embedding_normalize,
            batch_size=embedding_batch,
            device=embedding_device,
            base_url=embedding_endpoint or None,
            api_key=embedding_api_key or None,
        )
    except Exception as exc:
        summary = {
            "collection_id": collection_id,
            "collection_name": collection.name,
            "total_files": len(files),
            "ingested": 0,
            "skipped": 0,
            "embedded": 0,
            "failures": [
                {"reason": "backend_init_failed", "message": str(exc)},
            ],
            "status": "error",
        }
        _update_task(status='error', progress=0.0, error=str(exc), final=True, payload_override=summary)
        _log_result(
            "collection_rag_reindex_failed",
            {"reason": "backend_init_failed", "message": str(exc)},
        )
        return

    summary: dict[str, object] = {
        "collection_id": collection_id,
        "collection_name": collection.name,
        "total_files": len(files),
        "ingested": 0,
        "skipped": 0,
        "embedded": 0,
        "failures": [],
        "status": "running",
        "embedding_backend": {
            "name": getattr(backend, "name", embedding_backend_name),
            "model_name": getattr(backend, "model_name", embedding_model_name),
            "model_version": getattr(backend, "model_version", None),
            "dim": getattr(backend, "dim", embedding_dim),
            "endpoint": embedding_endpoint,
        },
        "options": {
            "skip_if_unchanged": skip_if_unchanged,
            "limit_chars": limit_chars,
            "chunk_max_tokens": chunk_config.max_tokens,
            "chunk_overlap": chunk_config.overlap,
            "chunk_min_tokens": chunk_config.min_tokens,
            "embedding_backend": embedding_backend_name,
            "embedding_model_name": embedding_model_name,
            "embedding_endpoint": embedding_endpoint,
        },
    }
    _update_task(status='running', progress=0.0)

    total_files = len(files) or 1
    failure_entries: list[dict[str, object]] = []

    for idx, file_obj in enumerate(files, start=1):
        ingest_result: dict[str, object] | None = None
        try:
            text, source_path = _collect_text_for_rag(file_obj, limit_chars=limit_chars)
            if not text.strip():
                raise RuntimeError("Пустой текст после извлечения.")
            metadata = {
                "source": "api.collection_rag_reindex",
                "source_path": str(source_path) if source_path else None,
                "file_sha1": file_obj.sha1,
                "collection_id": collection_id,
                "file_id": file_obj.id,
            }
            ingest_result = indexer.ingest_document(
                file_obj,
                text,
                metadata=metadata,
                skip_if_unchanged=skip_if_unchanged,
                commit=True,
            )
        except Exception as exc:
            db.session.rollback()
            failure_entries.append(
                {"file_id": file_obj.id, "reason": "ingest_failed", "message": str(exc)}
            )
            continue

        if ingest_result is None:
            continue

        if ingest_result.get("skipped"):
            summary["skipped"] = int(summary.get("skipped", 0)) + 1
        else:
            summary["ingested"] = int(summary.get("ingested", 0)) + 1

        document_id = int(ingest_result.get("document_id") or 0)
        if document_id:
            try:
                created = _embed_missing_chunks_for_document(
                    document_id,
                    backend,
                    batch_size=embedding_batch,
                    commit=True,
                )
                summary["embedded"] = int(summary.get("embedded", 0)) + created
            except Exception as exc:
                db.session.rollback()
                failure_entries.append(
                    {"file_id": file_obj.id, "reason": "embed_failed", "message": str(exc)}
                )

        progress = idx / total_files
        summary["failures"] = failure_entries[:20]
        _update_task(status='running', progress=progress)

    summary["status"] = "completed"
    try:
        backend.close()
    except Exception:
        pass

    summary["failures"] = failure_entries[:20]
    _update_task(status='completed', progress=1.0)

    _log_result(
        "collection_rag_reindex_done",
        {
            "ingested": summary.get("ingested"),
            "skipped": summary.get("skipped"),
            "embedded": summary.get("embedded"),
            "failures": len(failure_entries),
            "collection_name": collection.name,
            "backend": embedding_backend_name,
            "model": embedding_model_name,
        },
    )


def _build_translation_hint(query_lang: Optional[str], section_lang: Optional[str]) -> str:
    q = (query_lang or "").lower()
    s = (section_lang or "").lower()
    if not q or not s or q == s or s == "mixed" or s == "unknown":
        return ""
    if q == "ru":
        return f"Фрагмент на {s}, ответ оставь на русском, цитаты — на оригинале."
    if s == "ru":
        return f"Fragment language ru; respond in {q or 'user language'}, keep quotes на русском."
    return f"Фрагмент на {s}, ответ оставь на {q or 'языке запроса'}, цитаты — на оригинале."


def _select_rag_embedding_variant(document_ids: Sequence[int]) -> Optional[tuple[str, Optional[str], int]]:
    if not document_ids:
        return None
    variant = (
        db.session.query(
            RagChunkEmbedding.model_name,
            RagChunkEmbedding.model_version,
            RagChunkEmbedding.dim,
        )
        .join(RagDocumentChunk, RagDocumentChunk.id == RagChunkEmbedding.chunk_id)
        .filter(RagDocumentChunk.document_id.in_(list(document_ids)))
        .order_by(RagChunkEmbedding.id.desc())
        .first()
    )
    if variant:
        return variant
    fallback_variant = (
        db.session.query(
            RagChunkEmbedding.model_name,
            RagChunkEmbedding.model_version,
            RagChunkEmbedding.dim,
        )
        .order_by(RagChunkEmbedding.id.desc())
        .first()
    )
    return fallback_variant


def _serialize_context_section(section: ContextSection) -> Dict[str, Any]:
    return {
        "doc_id": section.doc_id,
        "chunk_id": section.chunk_id,
        "title": section.title,
        "language": section.language,
        "translation_hint": section.translation_hint,
        "score_dense": section.score_dense,
        "score_sparse": section.score_sparse,
        "combined_score": section.combined_score,
        "reasoning_hint": section.reasoning_hint,
        "token_estimate": section.token_estimate,
        "preview": section.preview,
        "content": section.content,
        "url": section.url,
        "extra": section.extra,
    }


def _summarize_search_hits(hit_records: Sequence[Dict[str, Any]]) -> str:
    """Сводит типы совпадений классического поиска в компактную строку."""
    if not hit_records:
        return ""
    mapping = {
        "title": "название",
        "author": "автор",
        "keywords": "ключевые слова",
        "excerpt": "выдержка",
        "abstract": "аннотация",
    }
    summary: List[str] = []
    seen: set[str] = set()
    for hit in hit_records:
        hit_type = str((hit or {}).get("type") or "").lower()
        if not hit_type:
            continue
        if hit_type == "tag":
            tag_key = str((hit or {}).get("key") or "").strip()
            label = f"тег {tag_key}" if tag_key else "тег"
        else:
            label = mapping.get(hit_type, hit_type)
        if not label or label in seen:
            continue
        seen.add(label)
        summary.append(label)
        if len(summary) >= 5:
            break
    return ", ".join(summary)


def _collect_file_metadata_lines(file_obj: Optional["File"], *, max_tags: int = 3) -> List[str]:
    """Возвращает краткое описание файла для использования в контексте LLM."""
    if not file_obj:
        return []
    parts: List[str] = []
    if getattr(file_obj, "author", None):
        parts.append(f"Автор: {file_obj.author}")
    if getattr(file_obj, "year", None):
        parts.append(f"Год: {file_obj.year}")
    if getattr(file_obj, "material_type", None):
        parts.append(f"Тип: {file_obj.material_type}")
    if getattr(file_obj, "keywords", None):
        kw = str(file_obj.keywords).strip()
        if kw:
            parts.append(f"Ключевые слова: {kw}")
    tag_values: List[str] = []
    try:
        for tag in (getattr(file_obj, "tags", None) or []):
            key = str(getattr(tag, "key", "") or "").strip()
            value = str(getattr(tag, "value", "") or "").strip()
            if not key or not value:
                continue
            tag_values.append(f"{key}:{value}")
    except Exception:
        tag_values = []
    if tag_values:
        deduped: List[str] = []
        seen_tags: set[str] = set()
        for item in tag_values:
            if item in seen_tags:
                continue
            seen_tags.add(item)
            deduped.append(item)
        if deduped:
            parts.append(f"Теги: {', '.join(deduped[:max_tags])}")
    return parts


def _build_result_metadata_summary(
    file_obj: Optional["File"],
    result_entry: Optional[Dict[str, Any]],
) -> List[str]:
    """Комбинирует метаданные файла и сведения о совпадениях классического поиска."""
    summary = _collect_file_metadata_lines(file_obj)
    if not result_entry:
        return summary
    hits_summary = _summarize_search_hits(result_entry.get("hits") or [])
    if hits_summary:
        summary.append(f"Совпадения: {hits_summary}")
    matched_terms = result_entry.get("matched_terms") or []
    if matched_terms:
        terms_preview = ", ".join(str(term) for term in matched_terms[:5])
        if terms_preview:
            summary.append(f"Термины: {terms_preview}")
    sources = result_entry.get("snippet_sources") or []
    if sources:
        summary.append(f"Источники сниппетов: {', '.join(sources[:3])}")
    return summary


def _compose_snippet_based_answer(
    query: str,
    entries: Sequence[Dict[str, Any]],
    *,
    primary_terms: Sequence[str] | None = None,
    anchor_terms: Sequence[str] | None = None,
    term_idf: Optional[Mapping[str, float]] = None,
    max_items: int = 5,
    snippet_limit: int = 360,
) -> tuple[str, int, List[Dict[str, Any]]]:
    """Формирует безопасный ответ по сниппетам без генерации LLM."""
    noise_terms = {
        "кто",
        "что",
        "какой",
        "какая",
        "какие",
        "каков",
        "какова",
        "каково",
        "такой",
        "такая",
        "такие",
        "такое",
        "является",
        "являлся",
        "являлась",
        "являются",
        "есть",
        "это",
        "может",
        "нужно",
        "надо",
        "биография",
        "биографии",
    }
    usable_terms_raw = [
        term.lower()
        for term in (primary_terms or [])
        if term and len(term) >= 3 and term.lower() not in noise_terms
    ]
    usable_terms: List[str] = []
    if term_idf:
        for term in usable_terms_raw:
            if term_idf.get(term, 1.0) >= KEYWORD_IDF_MIN:
                usable_terms.append(term)
    else:
        usable_terms = usable_terms_raw[:]
    if not usable_terms:
        usable_terms = [term for term in usable_terms_raw if len(term) >= 5]
    anchor_terms_lower = [term.lower() for term in (anchor_terms or []) if term]
    def _anchor_in_text(lower_text: str) -> bool:
        if not anchor_terms_lower:
            return True
        for anchor in anchor_terms_lower:
            if anchor and anchor in lower_text:
                return True
            if len(anchor) >= 5:
                stem = anchor[:-2]
                if stem and stem in lower_text:
                    return True
            if len(anchor) >= 6:
                stem = anchor[:-3]
                if stem and stem in lower_text:
                    return True
        return False
    facts: List[str] = []
    sources: List[str] = []
    used_count = 0
    seen_snippets: set[str] = set()
    context_items: List[Dict[str, Any]] = []
    for idx, res in enumerate(entries, start=1):
        snippet_candidates: List[str] = []
        snippet_candidates.extend(res.get("snippets") or [])
        selected_snippets: List[str] = []
        for candidate in snippet_candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text)
            if not normalized:
                continue
            lower = normalized.lower()
            if usable_terms and not any(term in lower for term in usable_terms):
                continue
            if not _anchor_in_text(lower):
                continue
            if lower in seen_snippets:
                continue
            seen_snippets.add(lower)
            trimmed = normalized[:max(snippet_limit, 360)].rstrip()
            selected_snippets.append(trimmed)
            if len(selected_snippets) >= 3:
                break
        if not selected_snippets:
            if not anchor_terms_lower and snippet_candidates:
                fallback_candidate = re.sub(r"\s+", " ", str(snippet_candidates[0] or "")).strip()
                if fallback_candidate:
                    selected_snippets.append(fallback_candidate[:max(snippet_limit, 360)].rstrip())
            else:
                continue
        if not selected_snippets:
            continue
        primary_snippet = selected_snippets[0][:snippet_limit].rstrip(". ")
        order = len(context_items) + 1
        facts.append(f"- {primary_snippet} [{order}]")
        title = (res.get("title") or res.get("rel_path") or f"file-{res.get('file_id')}") or ""
        title = str(title).strip()
        sources.append(f"- [{order}] {title}")
        used_count += 1
        context_items.append(
            {
                "order": order,
                "snippet": primary_snippet,
                "snippets": selected_snippets,
                "title": title,
                "file_id": int(res.get("file_id") or 0),
                "meta": res,
            }
        )
        if used_count >= max_items:
            break
    if not facts:
        return "", 0, []
    lines = ["Факты:", *facts, "Источники:", *sources]
    return "\n".join(lines), used_count, context_items


_SNIPPET_REF_RE = re.compile(r"\[(\d+)\]")


def _validate_snippet_llm_answer(answer: str, allowed_ids: Sequence[int]) -> Tuple[bool, List[str]]:
    text = (answer or "").strip()
    issues: List[str] = []
    if not text:
        issues.append("пустой ответ")
        return False, issues
    cited = sorted({int(match.group(1)) for match in _SNIPPET_REF_RE.finditer(text)})
    if not cited:
        issues.append("нет ссылок [n]")
    allowed_set = set(int(x) for x in allowed_ids)
    invalid = [cid for cid in cited if cid not in allowed_set]
    if invalid:
        issues.append(f"неизвестные ссылки {invalid}")
    return (not issues), issues


def _generate_snippet_llm_summary(
    query: str,
    context_items: Sequence[Dict[str, Any]],
    *,
    progress: Optional["_ProgressLogger"] = None,
    temperature: float = 0.1,
    max_tokens: int = 320,
) -> str:
    if not context_items:
        return ""
    allowed_ids = [item["order"] for item in context_items]
    context_lines: List[str] = []
    for item in context_items:
        title = item.get("title") or f"Источник {item.get('order')}"
        snippets = item.get("snippets") or [item.get("snippet") or ""]
        file_id = item.get("file_id")
        meta = item.get("meta") or {}
        hits = ", ".join(
            sorted({hit.get("term") for hit in (meta.get("hits") or []) if hit.get("term")})
        )
        header = f"[{item['order']}] Источник: {title}"
        if file_id:
            header += f" (file_id={file_id})"
        if hits:
            header += f" | ключи: {hits}"
        body_lines = []
        for snip in snippets:
            text = str(snip or "").strip()
            if not text:
                continue
            body_lines.append(f"- {text}")
        if not body_lines:
            body_lines.append("- (фрагмент отсутствует)")
        block = "\n".join([header] + body_lines)
        context_lines.append(block)
    system = (
        "Ты отвечаешь на вопросы поиска. Используй ТОЛЬКО предоставленные сниппеты, "
        "каждый помечен номером в квадратных скобках. Любой факт должен ссылаться на соответствующий номер. "
        "Если фрагменты содержат факты — перечисли их. Фразу «Недостаточно информации» используй "
        "только если ни один сниппет не даёт полезных сведений. Не придумывай новых фактов."
    )
    user = (
        f"Вопрос: {query}\n"
        "Доступные фрагменты:\n"
        + "\n".join(context_lines)
        + "\n\n"
        "Сформулируй краткий ответ на русском языке. Каждый факт сопровождай ссылкой [n]. "
        "Если информации недостаточно, ответь: 'Недостаточно информации в найденных источниках.'"
    )
    try:
        answer = call_lmstudio_compose(system, user, temperature=temperature, max_tokens=max_tokens) or ""
    except Exception:
        return ""
    answer = answer.strip()
    if not answer:
        return ""
    answer, translated = _ensure_russian_text(answer, label='snippet-answer')
    valid, issues = _validate_snippet_llm_answer(answer, allowed_ids)
    if not valid:
        if progress:
            progress.add(f"LLM ответ по сниппетам отклонён: {', '.join(issues)}")
        return ""
    if progress:
        note = "LLM ответ по сниппетам готов"
        if translated:
            note += ", выполнен перевод"
        progress.add(note)
    return answer


def _prepare_rag_context(
    query: str,
    results: Sequence[Dict[str, Any]],
    *,
    language_filters: Optional[Sequence[str]],
    max_chunks: int,
    progress: Optional["_ProgressLogger"] = None,
) -> tuple[Optional[Dict[str, Any]], list[str]]:
    notes: list[str] = []
    file_ids = [int(res.get('file_id')) for res in results if res.get('file_id')]
    if not file_ids:
        reason = "RAG: нет файлов с идентификаторами"
        if progress:
            progress.add(reason)
        notes.append(reason)
        return None, notes
    result_lookup: Dict[int, Dict[str, Any]] = {
        int(res.get("file_id")): res
        for res in results
        if res.get("file_id")
    }
    documents = (
        db.session.query(RagDocument)
        .filter(RagDocument.file_id.in_(file_ids), RagDocument.is_ready_for_rag == True)
        .all()
    )
    doc_map = {doc.id: doc for doc in documents}
    doc_ids = list(doc_map.keys())
    if not doc_ids:
        if progress:
            progress.add("RAG: нет документов, готовых для RAG-индекса")
        notes.append("Нет документов, подготовленных для RAG-индексации.")
        return None, notes
    variant = _select_rag_embedding_variant(doc_ids)
    if not variant:
        if progress:
            progress.add("RAG: эмбеддинги для документов не найдены")
        notes.append("Не найдены эмбеддинги для документов, используем fallback.")
        return None, notes
    model_name, model_version, dim = variant
    backend_choice = 'auto'
    version_token = (model_version or '').strip().lower()
    if 'lm-studio' in version_token or version_token in {'lmstudio', 'lm_studio'}:
        backend_choice = 'lm-studio'
    elif version_token in {'openai', 'openai-api'}:
        backend_choice = 'openai'
    try:
        backend = load_embedding_backend(
            backend_choice,
            model_name=model_name,
            dim=dim or 384,
        )
    except Exception as exc:
        if progress:
            progress.add(f"RAG: не удалось загрузить backend '{model_name}', fallback hash ({exc})")
        notes.append(f"Backend {model_name} недоступен ({exc}); используем hash fallback.")
        try:
            backend = load_embedding_backend(
                'hash',
                model_name=model_name,
                dim=dim or 384,
            )
        except Exception:
            backend = None
    if backend is None:
        if progress:
            progress.add("RAG: не удалось инициализировать backend эмбеддингов")
        notes.append("Не удалось инициализировать backend эмбеддингов.")
        return None, notes
    backend_name_label = ""
    try:
        vectors = backend.embed_many([query]) or []
        if vectors:
            query_vector = vectors[0]
        else:
            query_vector = [0.0] * (dim or 384)
        backend_name_label = getattr(backend, "name", getattr(backend, "model_name", "embedding-backend"))
    except Exception as exc:
        if progress:
            progress.add(f"RAG: не удалось вычислить эмбеддинг запроса ({exc})")
        notes.append(f"Не удалось вычислить эмбеддинг запроса ({exc}).")
        query_vector = [0.0] * (dim or 384)
    finally:
        try:
            backend.close()
        except Exception:
            pass
    if backend_name_label:
        notes.append(f"Эмбеддинги: backend {backend_name_label}")
    language_filters = [lang.lower() for lang in (language_filters or [])] or None
    vector_retriever = VectorRetriever(
        model_name=model_name,
        model_version=model_version,
        max_candidates=min(500, max_chunks * 40),
    )
    keyword_retriever = KeywordRetriever(
        limit=300,
        search_service=search_service,
        expand_terms_fn=_expand_synonyms,
        lemma_fn=_lemma,
    )
    rag_reranker = _get_rag_reranker()
    selector = ContextSelector(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        dense_weight=1.0,
        sparse_weight=0.6,
        doc_penalty=0.2,
        max_per_document=2,
        dense_top_k=max_chunks * 3,
        sparse_limit=300,
        max_total_tokens=_lm_safe_context_tokens(),
        rerank_fn=rag_reranker,
    )
    contexts = selector.select(
        query,
        query_vector,
        top_k=max_chunks,
        languages=language_filters,
        allowed_document_ids=doc_ids,
    )
    if rag_reranker:
        cfg = getattr(rag_reranker, "_config", None)
        if cfg and getattr(cfg, "model_name", None):
            notes.append(f"RAG rerank: cross-encoder {cfg.model_name}")
        else:
            notes.append("RAG rerank: cross-encoder активен")
    if not contexts:
        if progress:
            progress.add("RAG: контекст не подобран")
        notes.append("Контекст RAG не подобран; используется классический режим.")
        return None, notes
    query_lang = detect_language(query)
    sections: List[ContextSection] = []
    allowed_refs: set[Tuple[int, int]] = set()
    chunk_ids: List[int] = []
    for idx, cand in enumerate(contexts, start=1):
        doc = cand.document or doc_map.get(cand.chunk.document_id)
        if doc is None:
            continue
        file_obj = getattr(doc, "file", None)
        title = (getattr(file_obj, "title", None) or getattr(file_obj, "filename", None) or getattr(file_obj, "rel_path", None) or f"Документ {doc.id}").strip()
        section_lang = cand.chunk.lang_primary or doc.lang_primary or ""
        translation_hint = _build_translation_hint(query_lang, section_lang)
        res_entry = None
        if file_obj is not None:
            res_entry = result_lookup.get(getattr(file_obj, "id", 0))
        preview_text = cand.preview or ""
        if not preview_text and res_entry:
            snippets = list(res_entry.get("snippets") or [])
            if res_entry.get("llm_snippet"):
                snippets.append(res_entry["llm_snippet"])
            if snippets:
                snippet_text = str(snippets[0]).strip()
                if snippet_text:
                    preview_text = snippet_text[:200]
        extra_fields: Dict[str, Any] = {
            "section_path": cand.chunk.section_path or "",
            "keywords": cand.chunk.keywords_top or "",
        }
        if file_obj is not None:
            if getattr(file_obj, "author", None):
                extra_fields["Автор"] = file_obj.author
            if getattr(file_obj, "year", None):
                extra_fields["Год"] = file_obj.year
            if getattr(file_obj, "material_type", None):
                extra_fields["Тип материала"] = file_obj.material_type
            if getattr(file_obj, "keywords", None):
                extra_fields["Ключевые слова файла"] = file_obj.keywords
            tag_lines = _collect_file_metadata_lines(file_obj)
            tag_strings = [line for line in tag_lines if line.startswith("Теги: ")]
            if tag_strings:
                extra_fields["Теги"] = tag_strings[0].replace("Теги: ", "", 1)
        if cand.matched_terms:
            extra_fields["Совпавшие термины"] = ", ".join(cand.matched_terms[:5])
        if res_entry:
            hits_summary = _summarize_search_hits(res_entry.get("hits") or [])
            if hits_summary:
                extra_fields["Совпадения поиска"] = hits_summary
            matched_terms = res_entry.get("matched_terms") or []
            if matched_terms:
                extra_fields.setdefault(
                    "Термины поиска",
                    ", ".join(str(term) for term in matched_terms[:5]),
                )
            snippet_sources = res_entry.get("snippet_sources") or []
            if snippet_sources:
                extra_fields["Источники сниппетов"] = ", ".join(snippet_sources[:3])
        llm_snippet = res_entry.get("llm_snippet") if res_entry else None
        section = ContextSection(
            doc_id=doc.id,
            chunk_id=cand.chunk.id,
            title=title,
            language=section_lang,
            token_estimate=cand.token_estimate,
            score_dense=cand.dense_score,
            score_sparse=cand.sparse_score,
            combined_score=cand.combined_score,
            reasoning_hint=cand.reasoning_hint,
            preview=preview_text,
            content=cand.chunk.content or "",
            url=getattr(file_obj, "rel_path", None),
            extra={
                **extra_fields,
                **({"LLM сниппет": llm_snippet} if llm_snippet else {}),
            },
            translation_hint=translation_hint,
        )
        sections.append(section)
        allowed_refs.add((doc.id, cand.chunk.id))
        chunk_ids.append(cand.chunk.id)
        if progress:
            progress.add(
                f"RAG контекст [{idx}/{len(contexts)}]: doc={doc.id} chunk={cand.chunk.id} combined={cand.combined_score:.3f} hint={cand.reasoning_hint or '—'}"
            )
    if not sections:
        notes.append("Контекст RAG не сформирован (нет подходящих секций).")
        return None, notes
    return {
        "sections": sections,
        "allowed_refs": allowed_refs,
        "query_lang": query_lang,
        "chunk_ids": chunk_ids,
        "model_name": model_name,
        "model_version": model_version,
        "notes": notes,
    }, notes


def _store_rag_session(
    *,
    query: str,
    query_lang: Optional[str],
    chunk_ids: Sequence[int],
    system_prompt: str,
    user_prompt: str,
    answer: str,
    validation: "ValidationResult",
    model_name: Optional[str],
    params: Optional[Dict[str, Any]],
) -> Optional[int]:
    try:
        record = RagSession(
            query=query,
            query_lang=(query_lang or "")[:16],
            chunk_ids=",".join(str(cid) for cid in sorted(set(chunk_ids))),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=answer,
            validation=json.dumps(validation.as_dict(), ensure_ascii=False),
            model_name=model_name,
            params=json.dumps(params or {}, ensure_ascii=False),
        )
        db.session.add(record)
        db.session.commit()
        return record.id
    except Exception as exc:
        db.session.rollback()
        app.logger.warning("Не удалось сохранить RAG-сессию: %s", exc)
        return None


def _estimate_section_tokens(section: ContextSection) -> int:
    estimate = getattr(section, "token_estimate", 0) or 0
    if estimate > 0:
        return int(estimate)
    text = (section.content or "").strip()
    if not text:
        return 120
    words = len(text.split())
    return max(80, int(words * 1.6) + 80)


def _total_section_tokens(sections: Sequence[ContextSection]) -> int:
    total = 0
    for section in sections:
        total += _estimate_section_tokens(section)
    return total


def _generate_rag_answer(
    query: str,
    bundle: Dict[str, Any],
    *,
    temperature: float,
    max_tokens: int,
    progress: Optional["_ProgressLogger"] = None,
) -> Tuple[str, "ValidationResult", Optional[int], str, str]:
    sections: List[ContextSection] = list(bundle.get("sections") or [])
    safe_limit = _lm_safe_context_tokens()
    trimmed_sections = sections[:]
    trimmed = False
    while len(trimmed_sections) > 1 and _total_section_tokens(trimmed_sections) > safe_limit:
        trimmed_sections = trimmed_sections[:-1]
        trimmed = True
    if trimmed and trimmed_sections:
        notes_list = bundle.setdefault("notes", [])
        note = f"Контекст RAG сокращён до {len(trimmed_sections)} секций (ограничение окна модели)."
        notes_list.append(note)
        if progress:
            progress.add(f"RAG: контекст сокращён до {len(trimmed_sections)} секций (ограничение окна модели)")
        allowed_refs_trimmed = {(sec.doc_id, sec.chunk_id) for sec in trimmed_sections}
        bundle["sections"] = trimmed_sections
        bundle["allowed_refs"] = allowed_refs_trimmed
        bundle["chunk_ids"] = [sec.chunk_id for sec in trimmed_sections]
        sections = trimmed_sections
    else:
        sections = bundle["sections"]
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, sections)
    if progress:
        progress.add(f"RAG ответ: используем {len(sections)} чанков контекста")
    try:
        answer = call_lmstudio_compose(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        ).strip()
    except Exception as exc:
        if progress:
            progress.add(f"RAG ответ: ошибка генерации ({exc})")
        answer = ""
    if not answer:
        answer = fallback_answer()
        if progress:
            progress.add("RAG ответ: fallback без источников")
    validation = validate_answer(answer, sorted(bundle["allowed_refs"]))
    if validation.hallucination_warning and progress:
        progress.add("RAG валидация: обнаружены потенциальные проблемы с цитатами")
    if validation.missing_citations and progress:
        progress.add("RAG валидация: некоторые факты без ссылок")
    if validation.unknown_citations and progress:
        progress.add(f"RAG валидация: неизвестные ссылки {validation.unknown_citations}")
    if validation.extra_citations and progress:
        progress.add(f"RAG валидация: лишние ссылки {validation.extra_citations}")
    session_id = _store_rag_session(
        query=query,
        query_lang=bundle.get("query_lang"),
        chunk_ids=bundle.get("chunk_ids") or [],
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        answer=answer,
        validation=validation,
        model_name=bundle.get("model_name"),
        params={
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model_version": bundle.get("model_version"),
            "lm_model": getattr(_rt(), "lmstudio_model", None),
        },
    )
    if session_id and progress:
        progress.add(f"RAG сессия сохранена (id={session_id})")
    return answer, validation, session_id, system_prompt, user_prompt


@app.cli.command("init-db")
def init_db():
    db.create_all()
    # Заполняем или обновляем TagSchema известными ключами (идемпотентно)
    seeds = [
            ("dissertation", "научный руководитель", "ФИО научного руководителя"),
            ("dissertation", "специальность", "Код/направление ВАК"),
            ("dissertation", "организация", "Базовая организация / вуз"),
            ("dissertation", "степень", "Кандидат / Доктор"),
            ("textbook", "дисциплина", "Учебная дисциплина"),
            ("textbook", "издательство", "Издательство"),
            ("article", "журнал", "Журнал / сборник"),
            ("article", "номер", "Номер выпуска"),
            ("article", "страницы", "Диапазон страниц"),
            ("article", "doi", "Digital Object Identifier"),
            ("monograph", "издательство", "Издательство"),
            ("any", "isbn", "Международный стандартный номер книги"),
        ]
    # Дополняем дополнительными ключами для новой таксономии тегов
    seeds += [
            ("any", "lang", "Язык текста (ru/en/...)"),
            ("any", "ext", "Расширение файла (без точки)"),
            ("any", "pages", "Число страниц (для PDF)"),
            ("any", "doi", "Digital Object Identifier"),
            ("any", "udk", "Универсальный десятичный классификатор (УДК)"),
            ("any", "bbk", "Библиотечно-библиографическая классификация (ББК)"),
            ("article", "journal", "Название журнала / сборника"),
            ("article", "volume_issue", "Том/номер журнала"),
            ("article", "pages", "Страницы в выпуске"),
            ("standard", "standard", "Код стандарта (ГОСТ/ISO/IEC/СТО/СП/СанПиН/ТУ)"),
            ("standard", "status", "Статус стандарта (active/replaced)"),
            ("proceedings", "conference", "Название конференции/симпозиума"),
            ("report", "doc_kind", "Вид документа (ТЗ/Пояснительная записка и т.п.)"),
            ("report", "organization", "Организация-издатель/разработчик"),
            ("patent", "patent_no", "Номер патента"),
            ("patent", "ipc", "Класс международной патентной классификации (IPC/МПК)"),
            ("presentation", "slides", "Признак презентации (слайды)"),
        ]
    added = 0
    for mt, k, d in seeds:
        exists = TagSchema.query.filter_by(material_type=mt, key=k).first()
        if not exists:
            db.session.add(TagSchema(material_type=mt, key=k, description=d))
            added += 1
    db.session.commit()
    print(f"DB initialized. Added {added} tag schema rows (existing preserved).")


@app.cli.command("rag-ingest-file")
@click.argument("file_id", type=int)
@click.option("--text-path", type=click.Path(exists=True, dir_okay=False), default=None, help="Путь к готовому тексту (если нужно переопределить извлечение).")
@click.option("--normalizer-version", default="v1", show_default=True, help="Версия нормализатора текста.")
@click.option("--max-tokens", type=int, default=700, show_default=True, help="Максимум токенов в чанке до overlap.")
@click.option("--overlap", type=int, default=120, show_default=True, help="Число токенов overlap между чанками.")
@click.option("--min-tokens", type=int, default=80, show_default=True, help="Минимум токенов в чанке, при меньшем объёме чанки сливаются.")
@click.option("--skip-if-unchanged/--force", default=True, show_default=True, help="Пропускать индексацию, если текст не изменился.")
@click.option("--commit/--no-commit", default=True, show_default=True, help="Коммитить изменения в базе данных.")
def rag_ingest_file_cli(
    file_id: int,
    text_path: Optional[str],
    normalizer_version: str,
    max_tokens: int,
    overlap: int,
    min_tokens: int,
    skip_if_unchanged: bool,
    commit: bool,
) -> None:
    """Индексирует файл в таблицах RAG (документ, чанки, версии)."""
    file_obj = File.query.filter_by(id=file_id).first()
    if not file_obj:
        raise click.ClickException(f"Файл с id={file_id} не найден.")

    cfg = ChunkConfig(
        max_tokens=max(16, max_tokens),
        overlap=max(0, min(overlap, max_tokens - 1)),
        min_tokens=max(1, min_tokens),
    )
    raw_text: Optional[str] = None
    source_path: Optional[Path] = None

    if text_path:
        source_path = Path(text_path)
        raw_text = _read_text_file_with_fallback(source_path)
    else:
        for candidate in _resolve_candidate_paths(file_obj):
            if not candidate.exists() or not candidate.is_file():
                continue
            extracted = _extract_text_for_rag(candidate)
            if extracted:
                source_path = candidate
                raw_text = extracted
                break
    if not raw_text:
        raw_text = file_obj.text_excerpt or ""
    if not raw_text.strip():
        raise click.ClickException("Не удалось получить текст для индексации (файл пустой или недоступен).")

    metadata = {
        "source": "cli.rag_ingest_file",
        "source_path": str(source_path) if source_path else None,
        "file_sha1": file_obj.sha1,
        "manual_text_path": bool(text_path),
    }

    indexer = RagIndexer(chunk_config=cfg, normalizer_version=normalizer_version)
    try:
        result = indexer.ingest_document(
            file_obj,
            raw_text,
            metadata=metadata,
            skip_if_unchanged=skip_if_unchanged,
            commit=commit,
        )
    except Exception as exc:
        app.logger.exception("RAG ingest failed for file_id=%s", file_id)
        raise click.ClickException(str(exc))

    if result.get("skipped"):
        click.echo(
            f"RAG индекс для файла {file_id} пропущен: {result.get('reason', 'unchanged')}. "
            f"Версия={result.get('version')}"
        )
    else:
        click.echo(
            f"RAG индекс готов: файл {file_id}, версия {result.get('version')} (id={result.get('version_id')}), "
            f"чанков={result.get('chunks')}, язык={result.get('language')}, commit={'yes' if commit else 'no'}"
        )


@app.cli.command("rag-embed-chunks")
@click.option("--backend", type=click.Choice(["auto", "hash", "sentence-transformers", "lm-studio", "openai"]), default="auto", show_default=True)
@click.option("--model-name", default="intfloat/multilingual-e5-large", show_default=True, help="Название модели/конфигурации.")
@click.option("--model-version", default=None, help="Версия модели; по умолчанию берётся из backend.")
@click.option("--dim", type=int, default=384, show_default=True, help="Размерность эмбеддингов (для hash backend).")
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--limit", type=int, default=256, show_default=True, help="Максимум чанков для обработки за один запуск.")
@click.option("--min-chunk-id", type=int, default=None, help="Обрабатывать чанки с id >= указанного.")
@click.option("--normalize/--no-normalize", default=True, show_default=True, help="Нормализовать векторы (L2).")
@click.option("--device", default=None, help="Устройство для sentence-transformers (cpu/cuda:0/... ).")
@click.option("--endpoint", default=None, help="URL OpenAI/LM Studio embeddings API (если отличается от настроек).")
@click.option("--api-key", default=None, help="API ключ для embeddings API (если требуется).")
@click.option("--timeout", type=float, default=None, help="Таймаут запросов к embeddings API, секунды.")
@click.option("--commit/--no-commit", default=True, show_default=True, help="Сохранять изменения в базе данных.")
def rag_embed_chunks_cli(
    backend: str,
    model_name: str,
    model_version: Optional[str],
    dim: int,
    batch_size: int,
    limit: int,
    min_chunk_id: Optional[int],
    normalize: bool,
    device: Optional[str],
    endpoint: Optional[str],
    api_key: Optional[str],
    timeout: Optional[float],
    commit: bool,
) -> None:
    """Генерирует эмбеддинги для чанков RAG и сохраняет их в БД."""
    try:
        runtime = runtime_settings_store.current
    except Exception:
        runtime = None
    resolved_endpoint = endpoint or (getattr(runtime, "rag_embedding_endpoint", None) if runtime else None) or (
        getattr(runtime, "lmstudio_api_base", None) if runtime else None
    )
    resolved_api_key = api_key or (getattr(runtime, "rag_embedding_api_key", None) if runtime else None) or (
        getattr(runtime, "lmstudio_api_key", None) if runtime else None
    )
    engine = load_embedding_backend(
        backend,
        model_name=model_name,
        dim=dim,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
        base_url=resolved_endpoint,
        api_key=resolved_api_key,
        timeout=timeout,
    )
    resolved_model_name = getattr(engine, "model_name", model_name)
    resolved_model_version = model_version or getattr(engine, "model_version", "unknown")
    try:
        join_condition = and_(
            RagChunkEmbedding.chunk_id == RagDocumentChunk.id,
            RagChunkEmbedding.model_name == resolved_model_name,
            RagChunkEmbedding.model_version == resolved_model_version,
        )
        query = db.session.query(RagDocumentChunk).outerjoin(RagChunkEmbedding, join_condition)
        query = query.filter(RagChunkEmbedding.id.is_(None))
        if min_chunk_id is not None:
            query = query.filter(RagDocumentChunk.id >= min_chunk_id)
        query = query.order_by(RagDocumentChunk.id)
        if limit > 0:
            query = query.limit(limit)
        chunks: List[RagDocumentChunk] = query.all()
        if not chunks:
            click.echo("Нет чанков, требующих построения эмбеддингов.")
            return
        click.echo(f"Будет обработано чанков: {len(chunks)} (backend={getattr(engine, 'name', backend)})")
        processed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content or "" for c in batch]
            vectors = engine.embed_many(texts)
            if len(vectors) != len(batch):
                raise RuntimeError("Размер ответа backend эмбеддингов не совпадает с размером батча.")
            for chunk_obj, vector in zip(batch, vectors):
                vector_bytes = vector_to_bytes(vector)
                vector_checksum = hashlib.sha256(vector_bytes).hexdigest()
                embedding = RagChunkEmbedding(
                    chunk_id=chunk_obj.id,
                    model_name=resolved_model_name,
                    model_version=resolved_model_version,
                    dim=len(vector),
                    vector=vector_bytes,
                    vector_checksum=vector_checksum,
                )
                db.session.add(embedding)
                processed += 1
        if commit:
            db.session.commit()
            click.echo(f"Сохранено эмбеддингов: {processed}")
        else:
            db.session.rollback()
            click.echo(f"[no-commit] Сгенерировано эмбеддингов: {processed}, изменения отменены.")
    finally:
        try:
            engine.close()
        except Exception:
            pass


@app.cli.command("rag-search")
@click.argument("query", type=str)
@click.option("--backend", type=click.Choice(["auto", "hash", "sentence-transformers"]), default="auto", show_default=True)
@click.option("--model-name", default="intfloat/multilingual-e5-large", show_default=True)
@click.option("--model-version", default=None)
@click.option("--dim", type=int, default=384, show_default=True)
@click.option("--normalize/--no-normalize", default=True, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--device", default=None)
@click.option("--top-k", type=int, default=6, show_default=True)
@click.option("--max-candidates", type=int, default=1000, show_default=True)
def rag_search_cli(
    query: str,
    backend: str,
    model_name: str,
    model_version: Optional[str],
    dim: int,
    normalize: bool,
    batch_size: int,
    device: Optional[str],
    top_k: int,
    max_candidates: int,
) -> None:
    """Выполняет dense-поиск по чанкам RAG для текстового запроса."""
    engine = load_embedding_backend(
        backend,
        model_name=model_name,
        dim=dim,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
    )
    try:
        vectors = engine.embed_many([query])
        resolved_model_name = getattr(engine, "model_name", model_name)
        resolved_model_version = model_version or getattr(engine, "model_version", "unknown")
    finally:
        try:
            engine.close()
        except Exception:
            pass
    if not vectors:
        click.echo("Не удалось получить эмбеддинг для запроса.")
        return
    retriever = VectorRetriever(
        model_name=resolved_model_name,
        model_version=resolved_model_version,
        max_candidates=max_candidates,
    )
    results = retriever.search_by_vector(vectors[0], top_k=top_k)
    if not results:
        click.echo("Совпадений не найдено.")
        return
    click.echo(f"Топ {len(results)} результатов:")
    for idx, item in enumerate(results, start=1):
        doc = item.document
        doc_info = f"doc_id={doc.id}" if doc else "doc_id=?"
        click.echo(
            f"{idx}. score={item.score:.4f} chunk_id={item.chunk.id} {doc_info} "
            f"lang={item.lang or '-'} keywords={item.keywords or '-'}"
        )


@app.cli.command("rag-context")
@click.argument("query", type=str)
@click.option("--backend", type=click.Choice(["auto", "hash", "sentence-transformers"]), default="auto", show_default=True)
@click.option("--model-name", default="intfloat/multilingual-e5-large", show_default=True)
@click.option("--model-version", default=None)
@click.option("--dim", type=int, default=384, show_default=True)
@click.option("--normalize/--no-normalize", default=True, show_default=True)
@click.option("--batch-size", type=int, default=32, show_default=True)
@click.option("--device", default=None)
@click.option("--top-k", type=int, default=6, show_default=True)
@click.option("--dense-top-k", type=int, default=12, show_default=True)
@click.option("--sparse-limit", type=int, default=100, show_default=True)
@click.option("--max-candidates", type=int, default=1000, show_default=True)
@click.option("--max-per-document", type=int, default=2, show_default=True)
@click.option("--doc-penalty", type=float, default=0.1, show_default=True)
@click.option("--dense-weight", type=float, default=1.0, show_default=True)
@click.option("--sparse-weight", type=float, default=0.6, show_default=True)
@click.option("--min-dense-score", type=float, default=0.0, show_default=True)
@click.option("--min-sparse-score", type=float, default=0.0, show_default=True)
@click.option("--min-combined-score", type=float, default=0.0, show_default=True)
@click.option("--max-tokens", type=int, default=3500, show_default=True, help="Ограничение по суммарным токенам контекста (0 = без ограничения).")
@click.option("--rerank-mode", type=click.Choice(["none", "combined", "dense", "sparse", "cross-encoder"]), default="none", show_default=True, help="Дополнительный реранж, например по dense-score.")
@click.option("--rerank-model", default=None, help="Название модели для cross-encoder rerank (например cross-encoder/ms-marco-MiniLM-L-6-v2).")
@click.option("--rerank-device", default=None, help="Устройство для cross-encoder rerank (по умолчанию как --device).")
@click.option("--rerank-batch-size", type=int, default=16, show_default=True, help="Batch size для cross-encoder rerank.")
@click.option("--rerank-max-length", type=int, default=512, show_default=True, help="Максимальная длина входа CrossEncoder (токены).")
@click.option("--rerank-max-chars", type=int, default=1200, show_default=True, help="Обрезка текста чанка перед cross-encoder rerank (символы).")
@click.option("--lang", multiple=True, help="Ограничить языки чанков (повторяемая опция).")
def rag_context_cli(
    query: str,
    backend: str,
    model_name: str,
    model_version: Optional[str],
    dim: int,
    normalize: bool,
    batch_size: int,
    device: Optional[str],
    top_k: int,
    dense_top_k: int,
    sparse_limit: int,
    max_candidates: int,
    max_per_document: int,
    doc_penalty: float,
    dense_weight: float,
    sparse_weight: float,
    min_dense_score: float,
    min_sparse_score: float,
    min_combined_score: float,
    max_tokens: int,
    rerank_mode: str,
    rerank_model: Optional[str],
    rerank_device: Optional[str],
    rerank_batch_size: int,
    rerank_max_length: int,
    rerank_max_chars: int,
    lang: Sequence[str],
) -> None:
    """Формирует комбинированный контекст (dense + keyword) для запроса."""
    languages = [item.strip() for item in lang if item.strip()] or None
    engine = load_embedding_backend(
        backend,
        model_name=model_name,
        dim=dim,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
    )
    try:
        vectors = engine.embed_many([query])
        resolved_model_name = getattr(engine, "model_name", model_name)
        resolved_model_version = model_version or getattr(engine, "model_version", "unknown")
    finally:
        try:
            engine.close()
        except Exception:
            pass
    if not vectors:
        click.echo("Не удалось получить эмбеддинг для запроса.")
        return
    vector_retriever = VectorRetriever(
        model_name=resolved_model_name,
        model_version=resolved_model_version,
        max_candidates=max_candidates,
    )
    keyword_retriever = KeywordRetriever(
        limit=sparse_limit,
        search_service=search_service,
        expand_terms_fn=_expand_synonyms,
        lemma_fn=_lemma,
    )
    mode = (rerank_mode or "none").lower()

    def _make_rerank(mode_name: str):
        if mode_name == "dense":
            return lambda _q, items: sorted(items, key=lambda c: (c.dense_score, c.combined_score), reverse=True)
        if mode_name == "sparse":
            return lambda _q, items: sorted(items, key=lambda c: (c.sparse_score, c.combined_score), reverse=True)
        if mode_name == "combined":
            return lambda _q, items: sorted(items, key=lambda c: c.combined_score, reverse=True)
        return None

    rerank_fn = None
    rerank_instance = None
    if mode == "cross-encoder":
        if not rerank_model:
            click.echo("Для cross-encoder режима требуется указать --rerank-model.")
            return
        try:
            from agregator.rag.rerank import CrossEncoderConfig, load_reranker

            rerank_instance = load_reranker(
                "cross-encoder",
                config=CrossEncoderConfig(
                    model_name=rerank_model,
                    device=rerank_device or device,
                    batch_size=rerank_batch_size,
                    max_length=rerank_max_length,
                    truncate_chars=rerank_max_chars,
                ),
            )
            rerank_fn = rerank_instance
        except Exception as exc:
            click.echo(f"Не удалось инициализировать cross-encoder rerank: {exc}")
            return
    else:
        rerank_fn = _make_rerank(mode)

    selector = ContextSelector(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        doc_penalty=doc_penalty,
        max_per_document=max_per_document,
        dense_top_k=dense_top_k,
        sparse_limit=sparse_limit,
        min_dense_score=min_dense_score,
        min_sparse_score=min_sparse_score,
        min_combined_score=min_combined_score,
        max_total_tokens=max_tokens if max_tokens > 0 else None,
        rerank_fn=rerank_fn,
    )
    contexts = selector.select(
        query=query,
        query_vector=vectors[0],
        top_k=top_k,
        languages=languages,
        max_total_tokens=max_tokens if max_tokens > 0 else None,
    )
    if rerank_instance:
        try:
            rerank_instance.close()
        except Exception:
            pass
    if not contexts:
        click.echo("Контекст не подобран.")
        return
    click.echo(f"Отобрано чанков: {len(contexts)}")
    for idx, ctx in enumerate(contexts, start=1):
        doc = ctx.document
        doc_id = doc.id if doc else None
        title = None
        if doc and getattr(doc, "file", None):
            title = doc.file.title or doc.file.filename
        keywords_text = ", ".join(ctx.matched_terms[:5]) if ctx.matched_terms else "-"
        preview = (ctx.chunk.preview or ctx.chunk.content or "")[:200].replace("\n", " ").strip()
        click.echo(
            f"{idx}. doc={doc_id or '-'} chunk={ctx.chunk.id} adj={ctx.adjusted_score:.4f} "
            f"combined={ctx.combined_score:.4f} dense={ctx.dense_score:.4f} sparse={ctx.sparse_score:.4f} "
            f"lang={ctx.chunk.lang_primary or '-'} keywords={keywords_text}"
        )
        if title:
            click.echo(f"   title: {title}")
        if ctx.reasoning_hint:
            click.echo(f"   hint: {ctx.reasoning_hint}")
        if preview:
            click.echo(f"   preview: {preview}")


@app.cli.command("rag-generate")
@click.argument("query", type=str)
@click.option("--chunk-id", "chunk_ids", multiple=True, type=int, required=True, help="ID чанков, которые попадут в контекст.")
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--max-tokens", type=int, default=400, show_default=True)
@click.option("--llm/--no-llm", default=True, show_default=True, help="Вызвать ли LLM, иначе вывести только промпты.")
@click.option("--custom-system", default=None, help="Переопределить системный промпт.")
@click.option("--store/--no-store", default=True, show_default=True, help="Сохранить результат в таблицу rag_sessions.")
def rag_generate_cli(
    query: str,
    chunk_ids: Sequence[int],
    temperature: float,
    max_tokens: int,
    llm: bool,
    custom_system: Optional[str],
    store: bool,
) -> None:
    """Собирает промпт для RAG-ответа и опционально вызывает LLM."""
    if not chunk_ids:
        click.echo("Не указаны chunk-id.")
        raise click.Abort()
    rows = (
        db.session.query(RagDocumentChunk, RagDocument, File)
        .join(RagDocument, RagDocumentChunk.document_id == RagDocument.id)
        .join(File, File.id == RagDocument.file_id)
        .filter(RagDocumentChunk.id.in_(chunk_ids))
        .order_by(RagDocumentChunk.id)
        .all()
    )
    if not rows:
        click.echo("Чанки не найдены.")
        return
    sections: List[ContextSection] = []
    allowed_refs = set()
    query_lang = detect_language(query)
    for chunk, document, file_obj in rows:
        extra_meta = {}
        if chunk.meta:
            try:
                extra_meta = json.loads(chunk.meta)
            except Exception:
                extra_meta = {}
        section_lang = (chunk.lang_primary or document.lang_primary or "").lower()
        raw_token_estimate = extra_meta.get("token_estimate")
        if raw_token_estimate is None:
            raw_token_estimate = chunk.token_count
        try:
            token_estimate = int(raw_token_estimate or 0)
        except (TypeError, ValueError):
            token_estimate = 0
        sections.append(
            ContextSection(
                doc_id=document.id,
                chunk_id=chunk.id,
                title=file_obj.title or file_obj.filename or f"Документ {file_obj.id}",
                language=section_lang,
                token_estimate=token_estimate,
                score_dense=float(extra_meta.get("dense_score") or 0.0),
                score_sparse=float(extra_meta.get("sparse_score") or 0.0),
                combined_score=float(extra_meta.get("combined_score") or 0.0),
                reasoning_hint=str(extra_meta.get("hint") or ""),
                preview=chunk.preview or str(extra_meta.get("preview") or ""),
                content=chunk.content or "",
                url=file_obj.rel_path,
                extra={"section_path": chunk.section_path or ""},
            )
        )
        allowed_refs.add((document.id, chunk.id))

    for section in sections:
        section.translation_hint = _build_translation_hint(query_lang, section.language)

    system_prompt = build_system_prompt(custom_system)
    user_prompt = build_user_prompt(query, sections)
    click.echo("=== System Prompt ===")
    click.echo(system_prompt)
    click.echo("\n=== User Prompt ===")
    click.echo(user_prompt)

    if not llm:
        click.echo("\nLLM вызов отключён (флаг --no-llm).")
        return

    answer = call_lmstudio_compose(
        system_prompt,
        user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    ).strip()
    if not answer:
        answer = fallback_answer()
    click.echo("\n=== LLM Ответ ===")
    click.echo(answer)

    validation = validate_answer(answer, sorted(allowed_refs))
    click.echo("\n=== Валидация ===")
    click.echo(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2))

    if store:
        try:
            session_record = RagSession(
                query=query,
                query_lang=query_lang,
                chunk_ids=",".join(str(cid) for cid in chunk_ids),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                answer=answer,
                validation=json.dumps(validation.as_dict(), ensure_ascii=False),
                model_name=getattr(_rt(), "lmstudio_model", None),
                params=json.dumps({
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "llm": bool(llm),
                }, ensure_ascii=False),
            )
            db.session.add(session_record)
            db.session.commit()
            click.echo(f"\nSession saved: id={session_record.id}")
        except Exception as exc:
            db.session.rollback()
            click.echo(f"\n[warn] Не удалось сохранить сессию: {exc}")


# ------------------- App bootstrap -------------------

from routes import routes
app.register_blueprint(routes)

app.config['invalidate_facets'] = _invalidate_facets_cache
app.config['facet_cache'] = FACET_CACHE
app.config['facet_service'] = facet_service
app.config['search_service'] = search_service

# ------------------- Single-file Refresh API -------------------
@app.route("/api/files/<int:file_id>/refresh", methods=["POST"])
def api_file_refresh(file_id):
    """Re-extract text, refresh tags, and optionally re-run LLM for a single file.
    Respects runtime settings for LLM, audio summary and keywords. Always re-extracts text.
    """
    user = _load_current_user()
    f = File.query.get_or_404(file_id)
    refresh_task: TaskRecord | None = None
    if user:
        try:
            refresh_task = TaskRecord(
                name='refresh',
                status='running',
                progress=0.0,
                payload=json.dumps({'file_id': file_id, 'user_id': user.id})
            )
            refresh_task.started_at = datetime.utcnow()
            db.session.add(refresh_task)
            db.session.commit()
        except Exception:
            db.session.rollback()
            refresh_task = None
    refresh_params = {k: request.args.get(k) for k in ('use_llm', 'kws_audio', 'summarize')}
    log_detail = {'file_id': file_id, 'params': {k: v for k, v in refresh_params.items() if v is not None}}
    try:
        detail_payload = json.dumps(log_detail, ensure_ascii=False)
    except Exception:
        detail_payload = str(log_detail)
    try:
        _log_user_action(user, 'file_refresh_start', 'file', file_id, detail=detail_payload[:2000])
    except Exception:
        pass
    try:
        app.logger.info(
            "[user-action] user=%s action=file_refresh_start file_id=%s params=%s",
            getattr(user, 'username', None) or 'anonymous',
            file_id,
            log_detail.get('params')
        )
    except Exception:
        pass
    log: list[str] = []
    def _log(m: str):
        log.append(m)
    try:
        p = Path(f.path)
        if not p.exists() or not p.is_file():
            return jsonify({"ok": False, "error": "file_not_found", "log": ["file not found on disk"]}), 404
        ext = p.suffix.lower()
        filename = p.stem

        # очищаем кэшированные артефакты для этого файла (миниатюра и текстовый фрагмент)
        try:
            # кэш текстового фрагмента
            cache_dir = Path(app.static_folder) / 'cache' / 'text_excerpts'
            key = (f.sha1 or (f.rel_path or '').replace('/', '_')) + '.txt'
            fp = cache_dir / key
            if fp.exists():
                fp.unlink()
        except Exception:
            pass
        try:
            # миниатюра PDF
            if ext == '.pdf':
                thumb = Path(app.static_folder) / 'thumbnails' / (p.stem + '.png')
                if thumb.exists():
                    thumb.unlink()
        except Exception:
            pass

        # извлекаем текст (всегда при обновлении)
        text_excerpt = ""
        _log(f"extract: start for {p.name}")
        if ext == ".pdf":
            force = _always_ocr_first_page_dissertation() and (
                ((f.material_type or '').lower() in ('dissertation', 'dissertation_abstract'))
                or looks_like_dissertation_filename(filename)
            )
            text_excerpt = extract_text_pdf(p, limit_chars=40000, force_ocr_first_page=force)
        elif ext == ".docx":
            text_excerpt = extract_text_docx(p, limit_chars=40000)
        elif ext == ".rtf":
            text_excerpt = extract_text_rtf(p, limit_chars=40000)
        elif ext == ".epub":
            text_excerpt = extract_text_epub(p, limit_chars=40000)
        elif ext == ".djvu":
            text_excerpt = extract_text_djvu(p, limit_chars=40000)
        elif ext in AUDIO_EXTS:
            text_excerpt = transcribe_audio(p, limit_chars=40000)
        if text_excerpt:
            f.text_excerpt = text_excerpt[:40000]
        _log(f"extract: done chars={len(text_excerpt or '')}")
        # Общие теги: расширение и страницы PDF
        try:
            upsert_tag(f, 'ext', ext.lstrip('.'))
        except Exception:
            pass
        page_count = None
        if ext == '.pdf':
            try:
                with fitz.open(str(p)) as _doc:
                    page_count = len(_doc)
                    upsert_tag(f, 'pages', str(len(_doc)))
            except Exception:
                pass

        # Теги, специфичные для аудио
        if ext in AUDIO_EXTS:
            try:
                upsert_tag(f, 'формат', ext.lstrip('.'))
                upsert_tag(f, 'длительность', audio_duration_hhmmss(p))
            except Exception:
                pass

        # Эвристики по имени файла (заполняем только пустые поля)
        title = author = year = None
        for pat in FILENAME_PATTERNS:
            m = pat.match(filename)
            if m:
                gd = m.groupdict()
                title = gd.get("title")
                author = gd.get("author")
                year = gd.get("year")
                break
        if title and not f.title:
            f.title = title
        if author and not f.author:
            f.author = author
        if year and not f.year:
            f.year = year

        # Определяем тип материала, если не указан; корректируем аудио
        if not f.material_type:
            # Явно проставим тип по расширению
            if ext in IMAGE_EXTS:
                f.material_type = 'image'
            elif ext in AUDIO_EXTS:
                f.material_type = 'audio'
            else:
                f.material_type = guess_material_type(ext, text_excerpt, filename)
        if ext in AUDIO_EXTS and (f.material_type or '') == 'document':
            f.material_type = 'audio'
        if ext in IMAGE_EXTS and (f.material_type or '') == 'document':
            f.material_type = 'image'

        journal_entries = _extract_journal_toc_entries(text_excerpt or '') if text_excerpt else []
        f.material_type = _journal_safety_check(f.material_type, text_excerpt, page_count, journal_entries)

        # Теги, зависящие от типа
        try:
            ttags = extract_tags_for_type(f.material_type or '', text_excerpt or '', filename)
            if ttags:
                db.session.flush()
                for k, v in ttags.items():
                    values = v if isinstance(v, (list, tuple, set)) else [v]
                    for single in values:
                        if single is None:
                            continue
                        upsert_tag(f, k, single)
            _log(f"type-tags: {len(ttags or {})}")
        except Exception:
            _log("type-tags: error")
        # Извлекаем встроенные ключевые слова («Ключевые слова:»), если есть в тексте
        try:
            import re as _re
            kw_m = _re.search(r"(?:Ключевые\s+слова|Keywords)\s*[:\-]\s*(.+)", (text_excerpt or ''), flags=_re.IGNORECASE)
            if kw_m:
                f.keywords = kw_m.group(1).strip()[:2000]
                if KEYWORDS_TO_TAGS_ENABLED:
                    db.session.flush(); _upsert_keyword_tags(f)
                _log("inline-keywords: ok")
        except Exception:
            _log("inline-keywords: error")
        # Дополнительные расширенные теги
        try:
            rtags = extract_richer_tags(f.material_type or '', text_excerpt or '', filename)
            if rtags:
                db.session.flush()
                for k, v in rtags.items():
                    values = v if isinstance(v, (list, tuple, set)) else [v]
                    for single in values:
                        if single is None:
                            continue
                        upsert_tag(f, k, single)
            _log(f"richer-tags: {len(rtags or {})}")
        except Exception:
            _log("richer-tags: error")

        # Необязательное обогащение через LLM по настройкам или флагам
        use_llm = (request.args.get('use_llm') in ('1','true','yes','on')) if ('use_llm' in request.args) else DEFAULT_USE_LLM
        do_summarize = (request.args.get('summarize') in ('1','true','yes','on')) if ('summarize' in request.args) else SUMMARIZE_AUDIO
        kws_audio_on = (request.args.get('kws_audio') in ('1','true','yes','on')) if ('kws_audio' in request.args) else AUDIO_KEYWORDS_LLM

        if use_llm and (text_excerpt or ext in {'.txt', '.md'}):
            llm_text = text_excerpt or ""
            if not llm_text and ext in {".txt", ".md"}:
                try:
                    llm_text = p.read_text(encoding="utf-8", errors="ignore")[:15000]
                except Exception:
                    pass
            _log(f"llm: metadata start (len={len(llm_text)})")
            meta = call_lmstudio_for_metadata(llm_text, p.name)
            if meta:
                mt_meta = normalize_material_type(meta.get("material_type"))
                # Не позволяем LLM менять тип для явных форматов
                if ext in AUDIO_EXTS:
                    f.material_type = 'audio'
                elif ext in IMAGE_EXTS:
                    f.material_type = 'image'
                elif mt_meta:
                    f.material_type = mt_meta or f.material_type
                _t = (meta.get("title") or "").strip()
                if _t:
                    f.title = _t
                _a = _normalize_author(meta.get("author"))
                if _a:
                    f.author = _a
                _y = _normalize_year(meta.get("year"))
                if _y:
                    f.year = _y
                _adv = meta.get("advisor")
                if _adv is not None:
                    _adv_s = str(_adv).strip()
                    if _adv_s:
                        f.advisor = _adv_s
                kws = meta.get("keywords") or []
                if isinstance(kws, list) and kws:
                    f.keywords = ", ".join([str(x) for x in kws][:50])
                    if KEYWORDS_TO_TAGS_ENABLED:
                        db.session.flush()
                        _upsert_keyword_tags(f)
                if meta.get("novelty"):
                    db.session.flush()
                    upsert_tag(f, "научная новизна", str(meta.get("novelty")))
                for key in ("literature", "organizations", "classification"):
                    val = meta.get(key)
                    if isinstance(val, list) and val:
                        db.session.flush()
                        upsert_tag(f, key, "; ".join([str(x) for x in val]))
            else:
                _log("llm: metadata empty")

        # Резюме и ключевые слова для аудио
        if ext in AUDIO_EXTS:
            if kws_audio_on and (f.text_excerpt or '') and not (f.keywords or '').strip():
                try:
                    kws = call_lmstudio_keywords(f.text_excerpt, p.name)
                    if kws:
                        f.keywords = ", ".join(kws)
                        db.session.flush()
                        _upsert_keyword_tags(f)
                    _log(f"llm: audio keywords {len(kws or [])}")
                except Exception:
                    _log("llm: audio keywords error")
        # Анализ изображений (vision)
        if ext in IMAGE_EXTS and IMAGES_VISION_ENABLED:
            try:
                vis = call_lmstudio_vision(p, p.name)
                if isinstance(vis, dict):
                    desc = (vis.get('description') or '')
                    if desc:
                        f.abstract = desc[:8000]
                    kws = vis.get('keywords') or []
                    if isinstance(kws, list) and kws:
                        f.keywords = ", ".join([str(x) for x in kws][:50])
                        if KEYWORDS_TO_TAGS_ENABLED:
                            db.session.flush()
                            _upsert_keyword_tags(f)
                _log("vision: ok")
            except Exception:
                _log("vision: error")
            if do_summarize and (f.text_excerpt or ''):
                try:
                    summ = call_lmstudio_summarize(f.text_excerpt, p.name)
                    if summ:
                        f.abstract = summ[:2000]
                    _log("llm: summarize ok")
                except Exception:
                    _log("llm: summarize error")

        db.session.flush()
        # Базовые теги из полей
        if f.material_type:
            upsert_tag(f, "тип", f.material_type)
        if f.author:
            upsert_tag(f, "автор", f.author)
        if f.year:
            upsert_tag(f, "год", str(f.year))
        if refresh_task:
            try:
                refresh_task.status = 'completed'
                refresh_task.progress = 1.0
                refresh_task.finished_at = datetime.utcnow()
                refresh_task.error = None
                db.session.add(refresh_task)
            except Exception:
                logger.warning('Не удалось обновить статус задачи refresh #%s', refresh_task.id if refresh_task else None)
        db.session.commit()

        data = file_to_dict(f)
        data["abstract"] = f.abstract
        data["text_excerpt"] = f.text_excerpt
        return jsonify({"ok": True, "file": data, "log": log})
    except Exception as e:
        db.session.rollback()
        log.append(f"exception: {e}")
        if refresh_task:
            try:
                task_obj = TaskRecord.query.get(refresh_task.id)
                if task_obj:
                    task_obj.status = 'error'
                    task_obj.error = str(e)
                    task_obj.finished_at = datetime.utcnow()
                    db.session.commit()
            except Exception:
                db.session.rollback()
        return jsonify({"ok": False, "error": str(e), "log": log}), 500

# ------------------- Filename Suggestion & Rename -------------------
def _safe_filename_component(s: str, max_len: int = 80) -> str:
    s = (s or '').strip()
    # Заменим небезопасные символы
    s = re.sub(r"[<>:\\/\\|?*\r\n\t\0]", " ", s)
    # Уберём кавычки
    s = s.replace('"', ' ').replace("'", ' ')
    # Превратим последовательности пробелов/точек в один пробел
    s = re.sub(r"[ ]+", " ", s)
    s = s.strip().strip('.')
    # Замена пробелов на подчёркивания
    s = s.replace(' ', '_')
    # Схлопнем повторные знаки подчёркивания
    s = re.sub(r"_+", "_", s)
    # Ограничение длины
    if len(s) > max_len:
        s = s[:max_len]
    return s or "file"

def _extract_lastname(full: str) -> str:
    a = (full or '').strip()
    if not a:
        return ''
    # несколько авторов — берём первого
    a = re.split(r"[,&;/]|\band\b", a, flags=re.I)[0]
    # удалим инициалы вида "И.И." и одиночные
    a = re.sub(r"\b[А-ЯA-Z]\.[А-ЯA-Z]\.?", "", a)
    a = re.sub(r"\b[А-ЯA-Z]\.\b", "", a)
    parts = [p for p in re.split(r"\s+", a) if p]
    if not parts:
        return ''
    # Русские ФИО часто в формате: Фамилия Имя Отчество — берём 1-ю
    cyr = all(re.match(r"^[А-Яа-я\-]+$", p) for p in parts)
    if cyr and len(parts) >= 2:
        # если 3 слова и последнее похоже на отчество (окончания -вич/-вна), используем первое
        if len(parts) >= 3 and re.search(r"(вич|вна|вны|ызы)$", parts[-1], flags=re.I):
            return parts[0]
        # две части: обычно Фамилия Имя — первое
        return parts[0]
    # Иначе — лучшее эвристическое: последнее слово
    return parts[-1]

def _mt_abbr(mt: str) -> str:
    m = (mt or '').lower()
    return {
        'dissertation': 'ДИС', 'dissertation_abstract': 'АВТ',
        'article': 'СТ', 'textbook': 'УЧ', 'monograph': 'МОНО',
        'report': 'ОТЧ', 'patent': 'ПАТ', 'presentation': 'ПРЕЗ',
        'proceedings': 'ТЕЗ', 'standard': 'СТД', 'note': 'ЗАМ',
        'document': 'ДОК', 'audio': 'АУД', 'image': 'ИЗО'
    }.get(m, 'ДОК')

def _degree_abbr(file_obj: File) -> str:
    # по тегу 'степень'
    deg = ''
    try:
        for t in file_obj.tags:
            if (t.key or '').lower() == 'степень':
                deg = (t.value or '').lower()
                break
    except Exception:
        pass
    if 'доктор' in deg:
        return 'ДН'
    if 'кандид' in deg:
        return 'КН'
    return ''

def _build_suggested_basename(file_obj: File) -> str:
    mt = (file_obj.material_type or '').lower()
    abbr = _mt_abbr(mt)
    ctx = {
        'abbr': abbr,
        'degree': _degree_abbr(file_obj),
        'title': (file_obj.title or '').strip(),
        'author_last': _extract_lastname(file_obj.author or ''),
        'year': (file_obj.year or '').strip(),
        'filename': file_obj.filename or '',
    }
    # Используем шаблоны из настроек, если заданы
    pattern = RENAME_PATTERNS.get(mt) or RENAME_PATTERNS.get('default')
    base = None
    if pattern:
        try:
            base = pattern.format(**ctx)
        except Exception:
            base = None
    if not base:
        # Запасной вариант (жёстко заданные правила)
        if mt in ('dissertation','dissertation_abstract'):
            base = f"{abbr}.{(ctx['degree'] + '.') if ctx['degree'] else ''}{ctx['title']}.{ctx['author_last']}"
        elif mt == 'article':
            base = f"СТ.{ctx['title']}.{ctx['author_last'] or ctx['year']}"
        elif mt == 'textbook':
            base = f"УЧ.{ctx['title']}.{ctx['author_last'] or ctx['year']}"
        elif mt == 'monograph':
            base = f"МОНО.{ctx['title']}.{ctx['author_last'] or ctx['year']}"
        elif mt == 'image':
            base = f"ИЗО.{ctx['title'] or ctx['filename']}"
        elif mt == 'audio':
            base = f"АУД.{ctx['title'] or ctx['filename']}"
        else:
            base = f"{abbr}.{ctx['title'] or ctx['filename']}.{ctx['author_last'] or ctx['year']}"

    parts = [p for p in [ _safe_filename_component(x) for x in (base or '').split('.') ] if p]
    name = '.'.join(parts) if parts else _safe_filename_component(file_obj.filename)
    return name[:120]


def _determine_target_dir_for_file(file_obj: File, current_path: Path, material_type: str) -> Path:
    col = getattr(file_obj, 'collection', None)
    if not isinstance(col, Collection):
        col = _get_collection_instance(file_obj.collection_id)
    material_type = (material_type or '').strip().lower()
    fallback_sub = TYPE_DIRS.get(material_type) or TYPE_DIRS.get('other', 'other')
    if COLLECTIONS_IN_SEPARATE_DIRS:
        base_root = _collection_root_dir(col)
        if col and COLLECTION_TYPE_SUBDIRS:
            return base_root / fallback_sub
        return base_root
    if MOVE_ON_RENAME:
            return _scan_root_path() / fallback_sub
    return current_path.parent


def _rename_file_record(file_obj: File, base_name: str | None = None) -> Path:
    p_old = Path(file_obj.path)
    if not p_old.exists():
        raise FileNotFoundError(f"Файл {file_obj.path} не найден")
    base = (base_name or _build_suggested_basename(file_obj)) or file_obj.filename or p_old.stem
    base = _safe_filename_component(base, max_len=120)
    ext = file_obj.ext or p_old.suffix
    mt = (file_obj.material_type or '').strip().lower()
    target_dir = _determine_target_dir_for_file(file_obj, p_old, mt)
    target_dir.mkdir(parents=True, exist_ok=True)
    p_new = target_dir / (base + (ext or ''))
    i = 1
    while p_new.exists() and p_new != p_old:
        p_new = target_dir / (f"{base}_{i}" + (ext or ''))
        i += 1
    renamed = p_new != p_old
    if renamed:
        p_old.rename(p_new)
        if (file_obj.ext or '').lower() == '.pdf':
            try:
                old_thumb = Path(app.static_folder) / 'thumbnails' / (p_old.stem + '.png')
                if old_thumb.exists():
                    old_thumb.unlink()
            except Exception:
                pass
    file_obj.path = str(p_new)
    try:
        file_obj.rel_path = str(p_new.relative_to(_scan_root_path()))
    except Exception:
        file_obj.rel_path = p_new.name
    file_obj.filename = p_new.stem
    try:
        file_obj.mtime = p_new.stat().st_mtime
    except Exception:
        pass
    if renamed:
        try:
            db.session.add(ChangeLog(file_id=file_obj.id, action='rename', field='filename', old_value=p_old.name, new_value=p_new.name))
        except Exception:
            pass
    return p_new

@app.route('/api/files/<int:file_id>/rename-suggest', methods=['GET'])
def api_rename_suggest(file_id):
    f = File.query.get_or_404(file_id)
    try:
        suggested = _build_suggested_basename(f)
        return jsonify({"ok": True, "suggested": suggested, "ext": f.ext or ''})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/api/files/<int:file_id>/rename', methods=['POST'])
def api_rename_apply(file_id):
    f = File.query.get_or_404(file_id)
    data = request.json or {}
    base = (data.get('base') or '').strip()
    if not base:
        base = _build_suggested_basename(f)
    base = _safe_filename_component(base, max_len=120)
    try:
        p_new = _rename_file_record(f, base)
        db.session.commit()
        return jsonify({"ok": True, "new_name": Path(p_new).name, "rel_path": f.rel_path})
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/collections/<int:collection_id>/rename-all', methods=['POST'])
@require_admin
def api_collection_rename_all(collection_id: int):
    col = Collection.query.get_or_404(collection_id)
    files = File.query.filter(File.collection_id == collection_id).all()
    renamed = 0
    errors: list[dict[str, str]] = []
    for f in files:
        try:
            _rename_file_record(f)
            db.session.commit()
            renamed += 1
        except Exception as exc:
            db.session.rollback()
            errors.append({'id': str(f.id), 'error': str(exc)})
    try:
        _log_user_action(_load_current_user(), 'collection_rename_all', 'collection', col.id, detail=json.dumps({'renamed': renamed, 'errors': len(errors)}))
    except Exception:
        pass
    return jsonify({'ok': True, 'renamed': renamed, 'errors': errors})


@app.route('/api/collections/<int:collection_id>/clear', methods=['POST'])
@require_admin
def api_collection_clear(collection_id: int):
    col = Collection.query.get_or_404(collection_id)
    removed_files, errors = _delete_collection_files(collection_id)
    try:
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({'ok': False, 'error': str(exc)}), 500
    actor = _load_current_user()
    try:
        detail = json.dumps({'removed_files': removed_files, 'errors': len(errors)})
    except Exception:
        detail = None
    _log_user_action(actor, 'collection_clear', 'collection', col.id, detail=detail)
    if errors:
        app.logger.warning(f"[collections] clear reported {len(errors)} issues for collection {collection_id}: {errors[:3]}")
    return jsonify({'ok': True, 'removed_files': removed_files, 'errors': errors})


@app.route('/api/collections/<int:collection_id>/rag/reindex', methods=['POST'])
@require_admin
def api_collection_rag_reindex(collection_id: int):
    col = Collection.query.get_or_404(collection_id)
    try:
        file_count = db.session.query(func.count(File.id)).filter(File.collection_id == collection_id).scalar() or 0
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'Не удалось подсчитать файлы: {exc}'}), 500
    if file_count == 0:
        return jsonify({'ok': False, 'error': 'В коллекции нет файлов для RAG', 'error_code': 'collection_empty'}), 400

    data = request.get_json(silent=True) or {}
    runtime = runtime_settings_store.current
    default_backend = (runtime.rag_embedding_backend or 'lm-studio').strip().lower() or 'lm-studio'
    default_model = runtime.rag_embedding_model or 'nomic-ai/nomic-embed-text-v1.5-GGUF'
    default_dim = max(8, int(runtime.rag_embedding_dim or 768))
    default_batch = max(1, int(runtime.rag_embedding_batch_size or 32))
    default_device = runtime.rag_embedding_device
    default_endpoint = runtime.rag_embedding_endpoint or runtime.lmstudio_api_base or ''
    default_api_key = runtime.rag_embedding_api_key or runtime.lmstudio_api_key or ''

    def _int_opt(value: object, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _bool_opt(value: object, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in ('1', 'true', 'yes', 'on')

    chunk_max_tokens = max(16, _int_opt(data.get('max_tokens') or data.get('chunk_max_tokens'), 700))
    chunk_overlap = max(0, _int_opt(data.get('overlap') or data.get('chunk_overlap'), 120))
    chunk_min_tokens = max(1, _int_opt(data.get('min_tokens') or data.get('chunk_min_tokens'), 80))
    limit_chars = max(1000, _int_opt(data.get('limit_chars'), 120_000))
    skip_if_unchanged = _bool_opt(data.get('skip_if_unchanged'), True)
    normalizer_version = str(data.get('normalizer_version') or 'v1').strip() or 'v1'

    embedding_backend_raw = data.get('embedding_backend') or data.get('backend') or default_backend
    embedding_backend = str(embedding_backend_raw).strip().lower() or default_backend
    embedding_model_name = str(data.get('embedding_model_name') or data.get('model_name') or default_model).strip() or default_model
    embedding_dim = max(8, _int_opt(data.get('embedding_dim'), default_dim))
    embedding_batch_size = max(1, _int_opt(data.get('embedding_batch_size') or data.get('batch_size'), default_batch))
    embedding_normalize = _bool_opt(data.get('embedding_normalize'), True)
    device_value = data.get('embedding_device') or data.get('device')
    embedding_device = None
    if isinstance(device_value, str):
        trimmed = device_value.strip()
        embedding_device = trimmed or None
    if embedding_device is None and default_device:
        embedding_device = default_device
    endpoint_value = data.get('embedding_endpoint') or data.get('endpoint')
    if endpoint_value is None:
        endpoint_value = default_endpoint
    embedding_endpoint = str(endpoint_value or "").strip()
    api_key_raw = data.get('embedding_api_key') or data.get('api_key')
    if api_key_raw is None:
        api_key_raw = default_api_key
    embedding_api_key = str(api_key_raw or "")

    user = _load_current_user()

    options = {
        "chunk_max_tokens": chunk_max_tokens,
        "chunk_overlap": chunk_overlap,
        "chunk_min_tokens": chunk_min_tokens,
        "limit_chars": limit_chars,
        "skip_if_unchanged": skip_if_unchanged,
        "normalizer_version": normalizer_version,
        "embedding_backend": embedding_backend,
        "embedding_model_name": embedding_model_name,
        "embedding_dim": embedding_dim,
        "embedding_batch_size": embedding_batch_size,
        "embedding_normalize": embedding_normalize,
        "embedding_device": embedding_device,
        "embedding_endpoint": embedding_endpoint,
        "embedding_api_key": embedding_api_key,
        "user_id": getattr(user, 'id', None),
    }

    payload = {
        "collection_id": collection_id,
        "collection_name": col.name,
        "total_files": int(file_count),
        "status": "queued",
        "options": {
            "chunk_max_tokens": chunk_max_tokens,
            "chunk_overlap": chunk_overlap,
            "chunk_min_tokens": chunk_min_tokens,
            "limit_chars": limit_chars,
            "skip_if_unchanged": skip_if_unchanged,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model_name,
            "embedding_endpoint": embedding_endpoint,
        },
    }

    task = TaskRecord(
        name='rag_collection',
        status='queued',
        payload=json.dumps(payload, ensure_ascii=False),
        progress=0.0,
    )
    try:
        db.session.add(task)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({'ok': False, 'error': f'Не удалось создать задачу: {exc}'}), 500

    try:
        detail = json.dumps({"task_id": task.id, "files": int(file_count)}, ensure_ascii=False)
        _log_user_action(user, 'collection_rag_reindex_enqueued', 'collection', collection_id, detail=detail)
    except Exception:
        pass

    @copy_current_app_context
    def _runner():
        _run_rag_collection_job(task.id, collection_id, options)

    get_task_queue().submit(_runner, description=f"rag-collection-{collection_id}")

    return jsonify({'ok': True, 'task_id': task.id, 'files': int(file_count)})


def _delete_collection_files(col_id: int, remove_fs: bool = True) -> tuple[int, list[str]]:
    files = File.query.filter(File.collection_id == col_id).all()
    removed = 0
    errors: list[str] = []
    static_folder = Path(app.static_folder or (BASE_DIR / 'static'))
    thumbs_dir = static_folder / 'thumbnails'
    cache_dir = static_folder / 'cache' / 'text_excerpts'
    for f in files:
        path = Path(f.path) if f.path else None
        if remove_fs and path and path.exists():
            try:
                path.unlink()
                removed += 1
            except Exception as exc:
                errors.append(f"fs:{path}:{exc}")
        rel_path = Path(f.rel_path) if f.rel_path else None
        if rel_path:
            try:
                thumb = thumbs_dir / (rel_path.stem + '.png')
                if thumb.exists():
                    thumb.unlink()
            except Exception as exc:
                errors.append(f"thumb:{rel_path}:{exc}")
        try:
            key_base = f.sha1 or (f.rel_path or '').replace('/', '_')
            if key_base:
                cache_file = cache_dir / f"{key_base}.txt"
                if cache_file.exists():
                    cache_file.unlink()
        except Exception as exc:
            errors.append(f"cache:{f.id}:{exc}")
        fid = f.id
        db.session.delete(f)
        try:
            _delete_file_from_fts(fid)
        except Exception as exc:
            errors.append(f"fts:{fid}:{exc}")
    return removed, errors


def _remove_collection_directory_path(path: Path | None) -> None:
    if path is None:
        return
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception:
        pass


@app.route('/api/collections/<int:collection_id>', methods=['DELETE'])
@require_admin
def api_delete_collection(collection_id: int):
    col = Collection.query.get_or_404(collection_id)
    collection_root: Path | None = None
    if COLLECTIONS_IN_SEPARATE_DIRS:
        try:
            collection_root = _collection_root_dir(col, ensure=False)
        except Exception:
            collection_root = None
    removed_files, errors = _delete_collection_files(collection_id)
    try:
        CollectionMember.query.filter_by(collection_id=collection_id).delete()
        db.session.delete(col)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        return jsonify({'ok': False, 'error': str(exc)}), 500
    _remove_collection_directory_path(collection_root)
    try:
        _log_user_action(_load_current_user(), 'collection_delete', 'collection', collection_id, detail=json.dumps({'removed_files': removed_files, 'errors': len(errors)}))
    except Exception:
        pass
    return jsonify({'ok': True, 'removed_files': removed_files, 'errors': errors})

# ------------------- Scan with Progress -------------------
import threading, time

SCAN_PROGRESS = {
    "running": False,
    "stage": "idle",
    "total": 0,
    "processed": 0,
    "added": 0,
    "updated": 0,
    "removed": 0,
    "current": "",
    "use_llm": False,
    "error": None,
    "started_at": None,
    "updated_at": None,
    "eta_seconds": None,
    "history": [],
    "task_id": None,
}
SCAN_CANCEL = False
SCAN_TASK_ID: int | None = None

def _resolve_deleted_dir(root: Path) -> Path | None:
    deleted_cfg = app.config.get('DELETED_FOLDER')
    try:
        trash_dir = Path(deleted_cfg).expanduser() if deleted_cfg else Path('_deleted')
    except Exception:
        trash_dir = Path('_deleted')
    if not trash_dir.is_absolute():
        try:
            return (root / trash_dir).resolve()
        except Exception:
            return root / trash_dir
    try:
        return trash_dir.resolve()
    except Exception:
        return trash_dir


def _iter_files_for_scan(root: Path):
    trash_dir = _resolve_deleted_dir(root)
    for path in root.rglob("*"):
        if trash_dir is not None:
            try:
                path.relative_to(trash_dir)
                continue
            except ValueError:
                pass
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        yield path


def _bool_from_form(form, key: str, default: bool) -> bool:
    if form is None:
        return bool(default)
    if key in form:
        val = str(form.get(key) or '').strip().lower()
        return val in ('1', 'true', 'yes', 'on')
    return bool(default)

def _purge_cached_artifacts(records: list[tuple[File, Path]]):
    """Удаляет кэшированные превью и миниатюры для указанных файлов."""
    if not records:
        return
    try:
        static_dir = Path(app.static_folder or 'static').resolve()
    except Exception:
        return
    txt_cache = static_dir / 'cache' / 'text_excerpts'
    thumbs_dir = static_dir / 'thumbnails'
    for file_obj, path in records:
        try:
            if txt_cache.exists():
                key_base = file_obj.sha1 or (file_obj.rel_path or '').replace('/', '_')
                if key_base:
                    fp = txt_cache / f"{key_base}.txt"
                    if fp.exists():
                        fp.unlink()
        except Exception:
            pass
        try:
            if path.suffix.lower() == '.pdf' and thumbs_dir.exists():
                thumb = thumbs_dir / f"{path.stem}.png"
                if thumb.exists():
                    thumb.unlink()
        except Exception:
            pass

MAX_LOG_LINES = 200

def _scan_log(msg: str, level: str = "info"):
    SCAN_PROGRESS.setdefault("history", [])
    entry = {"t": time.time(), "level": level, "msg": str(msg)}
    SCAN_PROGRESS["history"].append(entry)
    if len(SCAN_PROGRESS["history"]) > MAX_LOG_LINES:
        SCAN_PROGRESS["history"] = SCAN_PROGRESS["history"][-MAX_LOG_LINES:]
    SCAN_PROGRESS["updated_at"] = time.time()
    log_line = f"[scan] {msg}"
    try:
        if level == 'error':
            app.logger.error(log_line)
        elif level in ('warn', 'warning'):
            app.logger.warning(log_line)
        else:
            app.logger.info(log_line)
    except Exception:
        pass

def _update_eta():
    try:
        st = SCAN_PROGRESS
        total = int(st.get("total") or 0)
        processed = int(st.get("processed") or 0)
        started = st.get("started_at") or time.time()
        if processed <= 0 or total <= 0:
            st["eta_seconds"] = None
            return
        elapsed = max(0.001, time.time() - float(started))
        rate = processed / elapsed
        remain = max(0, total - processed)
        eta = remain / rate if rate > 0 else None
        st["eta_seconds"] = int(eta) if eta is not None else None
    except Exception:
        st = SCAN_PROGRESS
        st["eta_seconds"] = None

def _run_scan_with_progress(extract_text: bool, use_llm: bool, prune: bool, skip: int = 0, targets: list | None = None):
    global SCAN_PROGRESS, SCAN_CANCEL, SCAN_TASK_ID
    with app.app_context():
        try:
            try:
                db.session.rollback()
            except Exception:
                db.session.remove()
            task = None
            if SCAN_TASK_ID:
                try:
                    task = TaskRecord.query.get(SCAN_TASK_ID)
                    if task:
                        task.status = 'running'
                        task.started_at = datetime.utcnow()
                        task.progress = 0.0
                        db.session.commit()
                except Exception:
                    db.session.rollback()
                    task = None
            SCAN_PROGRESS.update({
                "running": True,
                "stage": "counting",
                "processed": 0,
                "added": 0,
                "updated": 0,
                "removed": 0,
                "error": None,
                "use_llm": bool(use_llm),
                "started_at": time.time(),
                "updated_at": time.time(),
                "eta_seconds": None,
                "history": [],
            })
            _scan_log("Начало сканирования")
            root = _scan_root_path()
            # определяем набор для сканирования: явные цели или полный корневой обход
            if targets:
                try:
                    file_list = [Path(p) for p in targets]
                except Exception:
                    file_list = []
                # фильтруем по разрешённым путям и наличию
                file_list = [p for p in file_list if p.exists() and p.suffix.lower() in ALLOWED_EXTS]
                _scan_log(f"Сканирование только добавленных файлов: {len(file_list)}")
            else:
                file_list = list(_iter_files_for_scan(root))
            total = len(file_list)
            SCAN_PROGRESS["total"] = total
            if task:
                try:
                    task.progress = 0.0
                    task.payload = json.dumps({
                        "total": total,
                        "use_llm": bool(use_llm),
                        "extract_text": bool(extract_text),
                        "prune": bool(prune),
                    })
                    db.session.commit()
                except Exception:
                    db.session.rollback()
            # поддержка продолжения: пропускаем первые N файлов, если задано
            skip = max(0, min(int(skip or 0), total))
            if skip:
                _scan_log(f"Продолжение: пропуск первых {skip} из {total}")
                file_list = file_list[skip:]
                SCAN_PROGRESS["processed"] = skip
            _scan_log(f"Найдено файлов: {len(file_list)}")

            added = updated = 0
            cancelled = False
            for idx, path in enumerate(file_list, start=1):
                if SCAN_CANCEL:
                    _scan_log("Отмена пользователем", level="warn")
                    cancelled = True
                    break
                SCAN_PROGRESS.update({
                    "stage": "processing",
                    "processed": (skip + idx),
                    "current": str(path.name),
                    "updated_at": time.time()
                })
                _update_eta()
                if task and total:
                    try:
                        task.progress = min(1.0, float(skip + idx) / float(total))
                        if idx == 1 or idx % 10 == 0:
                            db.session.commit()
                    except Exception:
                        db.session.rollback()
                if idx == 1 or idx % 10 == 0:
                    _scan_log(f"Обработка: {path.name}")

                ext = path.suffix.lower()
                # по возможности вычисляем относительный путь к SCAN_ROOT
                try:
                    rel_path = str(path.relative_to(root))
                except Exception:
                    rel_path = path.name
                size = path.stat().st_size
                mtime = path.stat().st_mtime
                filename = path.stem
                page_count: int | None = None

                file_obj = File.query.filter_by(path=str(path)).first()
                if not file_obj:
                    sha1 = sha1_of_file(path)
                    file_obj = File(path=str(path), rel_path=rel_path, filename=filename,
                                    ext=ext, size=size, mtime=mtime, sha1=sha1)
                    db.session.add(file_obj)
                    added += 1
                    SCAN_PROGRESS["added"] = added
                else:
                    if (file_obj.size != size) or (file_obj.mtime != mtime):
                        sha1 = sha1_of_file(path)
                        file_obj.sha1 = sha1
                        file_obj.size = size
                        file_obj.mtime = mtime
                        file_obj.filename = filename
                    updated += 1
                    SCAN_PROGRESS["updated"] = updated

                # Извлечение текста (на основе текущей логики)
                text_excerpt = ""
                if extract_text:
                    if ext == ".pdf":
                        text_excerpt = extract_text_pdf(path, limit_chars=40000)
                    elif ext == ".docx":
                        text_excerpt = extract_text_docx(path, limit_chars=40000)
                    elif ext == ".rtf":
                        text_excerpt = extract_text_rtf(path, limit_chars=40000)
                    elif ext == ".epub":
                        text_excerpt = extract_text_epub(path, limit_chars=40000)
                    elif ext == ".djvu":
                        text_excerpt = extract_text_djvu(path, limit_chars=40000)
                    elif ext in AUDIO_EXTS:
                        _scan_log(f"Транскрибация аудио: {path.name}")
                        text_excerpt = transcribe_audio(path, limit_chars=40000)
                    if text_excerpt:
                        file_obj.text_excerpt = text_excerpt[:40000]
                    # Общие теги: расширение и страницы PDF
                    try:
                        upsert_tag(file_obj, 'ext', ext.lstrip('.'))
                    except Exception:
                        pass
                    if ext == '.pdf':
                        try:
                            with fitz.open(str(path)) as _doc:
                                page_count = len(_doc)
                                upsert_tag(file_obj, 'pages', str(len(_doc)))
                        except Exception:
                            pass
                    # Теги, специфичные для аудио
                    if ext in AUDIO_EXTS:
                        try:
                            upsert_tag(file_obj, 'формат', ext.lstrip('.'))
                            upsert_tag(file_obj, 'длительность', audio_duration_hhmmss(path))
                        except Exception:
                            pass
                        # Лёгкие ключевые слова из транскрипта через LLM
                        try:
                            if AUDIO_KEYWORDS_LLM and (file_obj.text_excerpt or '') and not (file_obj.keywords or '').strip():
                                kws = call_lmstudio_keywords(file_obj.text_excerpt, path.name)
                                if kws:
                                    file_obj.keywords = ", ".join(kws)
                        except Exception as _e:
                            _scan_log(f"audio keywords llm failed: {_e}", level="warn")
                    # Теги, специфичные для изображений
                    if ext in IMAGE_EXTS:
                        try:
                            upsert_tag(file_obj, 'формат', ext.lstrip('.'))
                            if PILImage is not None:
                                with PILImage.open(str(path)) as im:
                                    w, h = im.size
                                upsert_tag(file_obj, 'разрешение', f'{w}x{h}')
                                orient = 'портрет' if h >= w else 'альбом'
                                upsert_tag(file_obj, 'ориентация', orient)
                        except Exception:
                            pass
                        # Лёгкие ключевые слова из транскрипта через LLM
                        try:
                            if AUDIO_KEYWORDS_LLM and (file_obj.text_excerpt or '') and not (file_obj.keywords or '').strip():
                                kws = call_lmstudio_keywords(file_obj.text_excerpt, path.name)
                                if kws:
                                    file_obj.keywords = ", ".join(kws)
                        except Exception as _e:
                            _scan_log(f"audio keywords llm failed: {_e}", level="warn")

                # Встроенные ключевые слова в тексте для диссертаций/статей
                try:
                    import re as _re
                    kw_m = _re.search(r"(?:Ключевые\s+слова|Keywords)\s*[:\-]\s*(.+)", (text_excerpt or ''), flags=_re.IGNORECASE)
                    if kw_m:
                        file_obj.keywords = kw_m.group(1).strip()[:2000]
                        if KEYWORDS_TO_TAGS_ENABLED:
                            db.session.flush(); _upsert_keyword_tags(file_obj)
                except Exception:
                    pass

                # Эвристики на основе имени файла
                title, author, year = None, None, None
                for pat in FILENAME_PATTERNS:
                    m = pat.match(filename)
                    if m:
                        gd = m.groupdict()
                        title = gd.get("title") or title
                        author = gd.get("author") or author
                        year = gd.get("year") or year
                        break

                if title and not file_obj.title:
                    file_obj.title = title
                if author and not file_obj.author:
                    file_obj.author = author
                if year and not file_obj.year:
                    file_obj.year = year

                if not file_obj.material_type:
                    cand = _detect_type_pre_llm(ext, text_excerpt, filename)
                    if cand:
                        file_obj.material_type = cand
                if ext in AUDIO_EXTS and (file_obj.material_type or '') == 'document':
                    file_obj.material_type = 'audio'
                if ext in IMAGE_EXTS and (file_obj.material_type or '') == 'document':
                    file_obj.material_type = 'image'

                journal_entries = _extract_journal_toc_entries(text_excerpt or '') if text_excerpt else []
                file_obj.material_type = _journal_safety_check(file_obj.material_type, text_excerpt, page_count, journal_entries)

                # Типо-зависимые теги (до LLM)
                try:
                    ttags = extract_tags_for_type(file_obj.material_type or '', text_excerpt or '', filename)
                    if ttags:
                        db.session.flush()
                        for k, v in ttags.items():
                            values = v if isinstance(v, (list, tuple, set)) else [v]
                            for single in values:
                                if single is None:
                                    continue
                                upsert_tag(file_obj, k, single)
                except Exception as e:
                    _scan_log(f"type tags error: {e}", level="warn")
                # Дополнительные расширенные теги
                try:
                    rtags = extract_richer_tags(file_obj.material_type or '', text_excerpt or '', filename)
                    if rtags:
                        db.session.flush()
                        for k, v in rtags.items():
                            values = v if isinstance(v, (list, tuple, set)) else [v]
                            for single in values:
                                if single is None:
                                    continue
                                upsert_tag(file_obj, k, single)
                except Exception as e:
                    _scan_log(f"richer tags error: {e}", level="warn")

                # Обогащение через LLM (может быть медленным)
                if use_llm and (text_excerpt or ext in {'.txt', '.md'} or ext in IMAGE_EXTS or ext in {'.pdf','.docx','.rtf','.epub','.djvu'}):
                    SCAN_PROGRESS["stage"] = "llm"
                    SCAN_PROGRESS["updated_at"] = time.time()
                    _scan_log(f"LLM-анализ: {path.name}")
                    llm_text = text_excerpt or ""
                    if not llm_text and ext in {".txt", ".md"}:
                        try:
                            llm_text = Path(path).read_text(encoding="utf-8", errors="ignore")[:15000]
                        except Exception:
                            pass
                    # Лёгкое извлечение для распространённых форматов документов, если extract_text выключен
                    if not llm_text and ext in {'.pdf','.docx','.rtf','.epub','.djvu'}:
                        try:
                            if ext == '.pdf':
                                llm_text = extract_text_pdf(
                                    path,
                                    limit_chars=12000,
                                    force_ocr_first_page=(
                                        _always_ocr_first_page_dissertation()
                                        and looks_like_dissertation_filename(filename)
                                    ),
                                )
                            elif ext == '.docx':
                                llm_text = extract_text_docx(path, limit_chars=12000)
                            elif ext == '.rtf':
                                llm_text = extract_text_rtf(path, limit_chars=12000)
                            elif ext == '.epub':
                                llm_text = extract_text_epub(path, limit_chars=12000)
                            elif ext == '.djvu':
                                llm_text = extract_text_djvu(path, limit_chars=12000)
                        except Exception as _e:
                            _scan_log(f"llm lightweight extract error: {_e}", level='warn')
                    meta = call_lmstudio_for_metadata(llm_text, path.name)
                    if meta:
                        mt_meta = normalize_material_type(meta.get("material_type"))
                        # Для очевидных форматов не даём LLM переопределять тип
                        if ext in AUDIO_EXTS:
                            file_obj.material_type = 'audio'
                        elif ext in IMAGE_EXTS:
                            file_obj.material_type = 'image'
                        elif TYPE_LLM_OVERRIDE and mt_meta:
                            file_obj.material_type = mt_meta
                        _t = (meta.get("title") or "").strip()
                        if _t:
                            file_obj.title = _t
                        _a = _normalize_author(meta.get("author"))
                        if _a:
                            file_obj.author = _a
                        _y = _normalize_year(meta.get("year"))
                        if _y:
                            file_obj.year = _y
                        _adv = meta.get("advisor")
                        if _adv is not None:
                            _adv_s = str(_adv).strip()
                            if _adv_s:
                                file_obj.advisor = _adv_s
                        kws = meta.get("keywords") or []
                        if isinstance(kws, list):
                            file_obj.keywords = ", ".join([str(x) for x in kws][:50])
                        if meta.get("novelty"):
                            db.session.flush()
                            upsert_tag(file_obj, "научная новизна", str(meta.get("novelty")))
                        for key in ("literature", "organizations", "classification"):
                            val = meta.get(key)
                            if isinstance(val, list) and val:
                                db.session.flush()
                                upsert_tag(file_obj, key, "; ".join([str(x) for x in val]))
                        file_obj.material_type = _journal_safety_check(file_obj.material_type, llm_text or text_excerpt, page_count)
                    # Резюме для аудио
                    if ext in AUDIO_EXTS and SUMMARIZE_AUDIO and (file_obj.text_excerpt or ''):
                        summ = call_lmstudio_summarize(file_obj.text_excerpt, path.name)
                        if summ:
                            file_obj.abstract = summ[:2000]
                    # Анализ изображений через Vision
                    if ext in IMAGE_EXTS and IMAGES_VISION_ENABLED:
                        try:
                            vis = call_lmstudio_vision(path, path.name)
                            if isinstance(vis, dict):
                                desc = (vis.get('description') or '')
                                if desc:
                                    file_obj.abstract = desc[:8000]
                                kws = vis.get('keywords') or []
                                if isinstance(kws, list) and kws:
                                    file_obj.keywords = ", ".join([str(x) for x in kws][:50])
                        except Exception as e:
                            _scan_log(f"vision error: {e}", level="warn")
                    # Повторный прогон типовых тегов
                    try:
                        ttags = extract_tags_for_type(file_obj.material_type or '', text_excerpt or '', filename)
                        if ttags:
                            db.session.flush()
                            for k, v in ttags.items():
                                values = v if isinstance(v, (list, tuple, set)) else [v]
                                for single in values:
                                    if single is None:
                                        continue
                                    upsert_tag(file_obj, k, single)
                    except Exception as e:
                        _scan_log(f"type tags error(2): {e}", level="warn")
                    # Добавочные расширенные теги после LLM
                    try:
                        rtags = extract_richer_tags(file_obj.material_type or '', text_excerpt or '', filename)
                        if rtags:
                            db.session.flush()
                            for k, v in rtags.items():
                                values = v if isinstance(v, (list, tuple, set)) else [v]
                                for single in values:
                                    if single is None:
                                        continue
                                    upsert_tag(file_obj, k, single)
                    except Exception as e:
                        _scan_log(f"richer tags error(2): {e}", level="warn")

                db.session.flush()
                # Базовые теги
                if file_obj.material_type:
                    upsert_tag(file_obj, "тип", file_obj.material_type)
                if file_obj.author:
                    upsert_tag(file_obj, "автор", file_obj.author)
                if file_obj.year:
                    upsert_tag(file_obj, "год", str(file_obj.year))
                db.session.flush()
                try:
                    _sync_file_to_fts(file_obj)
                except Exception as sync_exc:
                    _scan_log(f"fts sync failed: {sync_exc}", level='warn')
                db.session.commit()

            removed = 0
            if prune and not SCAN_CANCEL:
                SCAN_PROGRESS["stage"] = "prune"
                SCAN_PROGRESS["updated_at"] = time.time()
                _scan_log("Удаление отсутствующих файлов")
                removed = prune_missing_files()
                try:
                    _rebuild_files_fts()
                    _rebuild_tags_fts()
                except Exception as rebuild_exc:
                    _scan_log(f"fts rebuild failed: {rebuild_exc}", level='warn')
                db.session.commit()
            SCAN_PROGRESS.update({"removed": removed, "stage": "done", "running": False, "updated_at": time.time()})
            _scan_log("Сканирование завершено")
            if task:
                try:
                    task.status = 'cancelled' if cancelled else 'completed'
                    task.finished_at = datetime.utcnow()
                    task.progress = 1.0 if not cancelled else task.progress
                    db.session.commit()
                except Exception:
                    db.session.rollback()
        except Exception as e:
            SCAN_PROGRESS.update({"error": str(e), "running": False, "stage": "error", "updated_at": time.time()})
            _scan_log(f"Ошибка: {e}", level="error")
            if task:
                try:
                    task.status = 'error'
                    task.error = str(e)
                    task.finished_at = datetime.utcnow()
                    db.session.commit()
                except Exception:
                    db.session.rollback()
        finally:
            SCAN_CANCEL = False
            SCAN_TASK_ID = None
            SCAN_PROGRESS["task_id"] = None
            SCAN_PROGRESS["scope"] = None
            try:
                db.session.remove()
            except Exception:
                pass

@app.route("/scan/start", methods=["POST"])
@require_admin
def scan_start():
    global SCAN_CANCEL, SCAN_TASK_ID
    if SCAN_PROGRESS.get("running"):
        return jsonify({"status": "busy"}), 409
    extract_text = _bool_from_form(request.form, "extract_text", EXTRACT_TEXT)
    use_llm = _bool_from_form(request.form, "use_llm", DEFAULT_USE_LLM)
    prune = _bool_from_form(request.form, "prune", DEFAULT_PRUNE)
    SCAN_PROGRESS["scope"] = {
        "type": "library",
        "label": "Вся библиотека",
        "extract_text": bool(extract_text),
        "use_llm": bool(use_llm),
        "prune": bool(prune),
    }
    SCAN_CANCEL = False
    skip = 0
    try:
        skip = int(request.form.get('skip', '0') or 0)
    except Exception:
        skip = 0
    # Очищаем статические кэши перед полным сканом: текстовые фрагменты и миниатюры
    try:
        static_dir = Path(app.static_folder)
        txt_cache = static_dir / 'cache' / 'text_excerpts'
        if txt_cache.exists():
            for fp in txt_cache.glob('*.txt'):
                try: fp.unlink()
                except Exception: pass
        thumbs = static_dir / 'thumbnails'
        if thumbs.exists():
            for fp in thumbs.glob('*.png'):
                try: fp.unlink()
                except Exception: pass
    except Exception:
        pass
    payload = {
        "extract_text": extract_text,
        "use_llm": use_llm,
        "prune": prune,
        "skip": skip,
    }
    try:
        task = TaskRecord(name='scan', status='queued', payload=json.dumps(payload), progress=0.0)
        db.session.add(task)
        db.session.commit()
        SCAN_TASK_ID = task.id
        SCAN_PROGRESS["task_id"] = task.id
    except Exception:
        db.session.rollback()
        SCAN_TASK_ID = None
        SCAN_PROGRESS["task_id"] = None
    _log_user_action(_load_current_user(), 'scan_start', 'scan', SCAN_TASK_ID, detail=json.dumps(payload))
    get_task_queue().submit(
        _run_scan_with_progress,
        extract_text,
        use_llm,
        prune,
        skip,
        description="scan-full",
    )
    return jsonify({"status": "started"})


@app.route("/scan/collection/<int:collection_id>", methods=["POST"])
@require_admin
def scan_collection(collection_id: int):
    global SCAN_CANCEL, SCAN_TASK_ID
    if SCAN_PROGRESS.get("running"):
        return jsonify({"status": "busy"}), 409
    collection = Collection.query.get(collection_id)
    if not collection:
        return jsonify({"status": "not_found", "error": "collection_not_found"}), 404
    extract_text = _bool_from_form(request.form, "extract_text", EXTRACT_TEXT)
    use_llm = _bool_from_form(request.form, "use_llm", DEFAULT_USE_LLM)
    prune = _bool_from_form(request.form, "prune", DEFAULT_PRUNE)
    files = File.query.filter(File.collection_id == collection_id).all()
    if not files:
        return jsonify({"status": "empty", "collection_id": collection_id, "files": 0})

    targets: list[str] = []
    records: list[tuple[File, Path]] = []
    missing = 0
    for file_obj in files:
        path_str = (file_obj.path or '').strip()
        path = Path(path_str) if path_str else None
        if not path or not path.exists():
            fallback = None
            if file_obj.rel_path:
                try:
                    fallback = (_scan_root_path() / Path(file_obj.rel_path)).resolve()
                except Exception:
                    fallback = None
            if fallback and fallback.exists():
                path = fallback
            else:
                missing += 1
                continue
        if not path.is_file():
            missing += 1
            continue
        ext = path.suffix.lower()
        if ext not in ALLOWED_EXTS:
            continue
        resolved = path.resolve()
        targets.append(str(resolved))
        records.append((file_obj, resolved))

    if not targets:
        return jsonify({"status": "empty", "collection_id": collection_id, "files": 0, "missing": missing})

    _purge_cached_artifacts(records)
    SCAN_CANCEL = False
    SCAN_PROGRESS["scope"] = {
        "type": "collection",
        "collection_id": collection.id,
        "label": f"Коллекция «{collection.name}»",
        "count": len(targets),
        "extract_text": bool(extract_text),
        "use_llm": bool(use_llm),
        "prune": bool(prune),
    }

    payload = {
        "extract_text": extract_text,
        "use_llm": use_llm,
        "prune": prune,
        "collection_id": collection.id,
        "collection_name": collection.name,
        "targets": len(targets),
        "missing": missing,
    }
    try:
        task = TaskRecord(name='scan', status='queued', payload=json.dumps(payload), progress=0.0)
        db.session.add(task)
        db.session.commit()
        SCAN_TASK_ID = task.id
        SCAN_PROGRESS["task_id"] = task.id
    except Exception:
        db.session.rollback()
        SCAN_TASK_ID = None
        SCAN_PROGRESS["task_id"] = None

    try:
        _log_user_action(_load_current_user(), 'scan_collection_start', 'collection', collection.id, detail=json.dumps(payload))
    except Exception:
        pass

    get_task_queue().submit(
        _run_scan_with_progress,
        extract_text,
        use_llm,
        prune,
        0,
        targets,
        description=f"scan-collection-{collection.id}",
    )
    return jsonify({"status": "started", "files": len(targets), "missing": missing})

@app.route("/scan/status")
def scan_status():
    return jsonify(SCAN_PROGRESS)

@app.route("/scan/cancel", methods=["POST"])
@require_admin
def scan_cancel():
    global SCAN_CANCEL
    SCAN_CANCEL = True
    try:
        if SCAN_TASK_ID:
            task = TaskRecord.query.get(SCAN_TASK_ID)
            if task and task.status not in ('completed','error','cancelled'):
                task.status = 'cancelling'
                db.session.commit()
    except Exception:
        db.session.rollback()
    _log_user_action(_load_current_user(), 'scan_cancel', 'scan', SCAN_TASK_ID)
    return jsonify({"status": "cancelling"})


# ------------------- AI Search (MVP) -------------------

def _normalize_keyword_candidate(raw: str) -> str:
    """Очистить строку, предназначенную для использования в качестве ключевого термина."""
    if not raw:
        return ""
    cleaned = re.sub(r"\s+", " ", str(raw)).strip()
    cleaned = cleaned.strip("\"'«»“”[](){}")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if len(lowered) < 2:
        return ""
    return lowered


def _extract_quoted_phrases(text: str) -> list[str]:
    """Извлечь фразы в кавычках и нормализовать их для использования как термов поиска."""
    if not text:
        return []
    pattern = r'"([^"\n]{2,})"|«([^»\n]{2,})»|“([^”\n]{2,})”|”([^“\n]{2,})“|ʼ([^ʼ\n]{2,})ʼ|\'([^\'\n]{2,})\''
    phrases: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(pattern, str(text)):
        candidate = next((grp for grp in match.groups() if grp), "")
        normalized = _normalize_keyword_candidate(candidate)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        phrases.append(normalized)
    return phrases


def _normalize_language_code(lang: str) -> str | None:
    if lang is None:
        return None
    code = re.sub(r"[^a-zA-Z\-]", "", str(lang)).lower()
    if not code:
        return None
    if "-" in code:
        code = code.split("-", 1)[0]
    if len(code) < 2:
        return None
    return code[:8]


def _guess_query_language(text: str) -> str | None:
    if not text:
        return None
    cyr = sum(1 for ch in text if ('а' <= ch.lower() <= 'я') or ch.lower() == 'ё')
    lat = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
    if cyr == 0 and lat == 0:
        return None
    return 'ru' if cyr >= lat else 'en'


def _ensure_russian_text(text: str, *, label: str = 'text') -> tuple[str, bool]:
    """
    Гарантировать русскую формулировку текста.

    Возвращает (new_text, translated_flag). Перевод применяется, только если в тексте
    отсутствует кириллица, но есть латиница — чтобы не трогать уже русские ответы и смешанные фрагменты.
    """
    original = (text or '').strip()
    if not original:
        return '', False
    cyr = sum(1 for ch in original if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё')
    lat = sum(1 for ch in original if 'a' <= ch.lower() <= 'z')
    if cyr > 0 or lat == 0:
        return original, False
    system = (
        "Ты профессиональный переводчик. Переведи приведённый текст на русский язык, "
        "сохраняя смысл, факты, числовые значения и ссылки. Не добавляй пояснений и комментариев."
    )
    user = (
        "Исходный текст:\n"
        f"{original}\n\n"
        "Переведи этот текст на русский язык, сохранив стиль и точность. "
        "Не добавляй ничего сверх перевода."
    )
    translated = ""
    try:
        translated = call_lmstudio_compose(
            system,
            user,
            temperature=0.0,
            max_tokens=min(max(200, len(original) + 120), _lm_max_output_tokens()),
        )
    except Exception as exc:
        app.logger.debug("Russian translation failed (%s): %s", label, exc)
    translated = (translated or '').strip()
    if not translated:
        return original, False
    trans_cyr = sum(1 for ch in translated if 'а' <= ch.lower() <= 'я' or ch.lower() == 'ё')
    if trans_cyr == 0:
        return original, False
    return translated, True


def _ai_expand_multilingual_terms(query: str, base_terms: Sequence[str], languages: Sequence[str]) -> tuple[list[str], list[tuple[str, int]]]:
    sanitized: list[str] = []
    seen_langs = set()
    for raw_lang in languages:
        normalized = _normalize_language_code(raw_lang)
        if not normalized:
            continue
        if normalized in seen_langs:
            continue
        seen_langs.add(normalized)
        sanitized.append(normalized)
    if not sanitized:
        return [], []
    base_lang = _guess_query_language(query)
    if base_lang:
        sanitized = [lang for lang in sanitized if lang != base_lang]
    if not sanitized:
        return [], []
    sanitized = sanitized[:5]
    base_preview = ", ".join([str(t) for t in list(base_terms)[:6] if t])
    system = (
        "Ты помогаешь поисковой системе расширить запрос. "
        "Для каждого указанного языка подбери 2-6 коротких ключевых фраз (1-3 слова) на этом языке. "
        "Фразы должны быть естественными, без транслитерации и повторов. "
        "Верни JSON-объект без пояснений: {\"en\": [\"term\"...], \"de\": [...]}. Ответ должен быть только JSON."
        " Используй только перечисленные языки в виде ключей."
    )
    user_parts = [
        f"Исходный запрос: {query}",
    ]
    if base_preview:
        user_parts.append(f"Базовые ключевые слова: {base_preview}")
    user_parts.append(f"Языки для расширения: {', '.join(sanitized)}")
    user_parts.append("Не добавляй другие поля и пояснения.")
    raw = call_lmstudio_compose(system, "\n".join(user_parts), temperature=0.15, max_tokens=280)
    if not raw:
        return [], []
    raw = raw.strip()
    if not raw:
        return [], []
    parsed: dict[str, list[str]] | None = None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            parsed = data
    except Exception:
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    parsed = data
            except Exception:
                parsed = None
    if not parsed:
        return [], []
    base_seen = {str(term).strip().lower() for term in base_terms if term}
    extras: list[str] = []
    lang_order: list[str] = []
    lang_counter: dict[str, int] = {}
    for lang, items in parsed.items():
        normalized_lang = _normalize_language_code(lang)
        if not normalized_lang or normalized_lang not in sanitized:
            continue
        if not isinstance(items, list):
            continue
        if normalized_lang not in lang_counter:
            lang_counter[normalized_lang] = 0
            lang_order.append(normalized_lang)
        for item in items:
            normalized_term = _normalize_keyword_candidate(item)
            if not normalized_term:
                continue
            key = normalized_term.lower()
            if key in base_seen:
                continue
            base_seen.add(key)
            extras.append(normalized_term)
            lang_counter[normalized_lang] += 1
    lang_stats = [(lang, lang_counter.get(lang, 0)) for lang in lang_order if lang_counter.get(lang, 0) > 0]
    return extras, lang_stats


def _ai_expand_keywords(
    query: str,
    *,
    multi_lang: bool = False,
    target_langs: Sequence[str] | None = None,
) -> tuple[list[str], list[tuple[str, int]]]:
    q = (query or "").strip()
    if not q:
        return [], []
    mandatory = _extract_quoted_phrases(q)
    sanitized_langs: list[str] = []
    if multi_lang and target_langs:
        seen_langs = set()
        for raw_lang in target_langs:
            normalized = _normalize_language_code(raw_lang)
            if not normalized:
                continue
            if normalized in seen_langs:
                continue
            seen_langs.add(normalized)
            sanitized_langs.append(normalized)
        if sanitized_langs:
            sanitized_langs = sanitized_langs[:5]
    base_lang = _guess_query_language(q)
    cross_langs: list[str] = []
    if base_lang == 'ru':
        cross_langs.append('en')
    elif base_lang == 'en':
        cross_langs.append('ru')
    active_langs: list[str] = []
    for lang in sanitized_langs + cross_langs:
        normalized = _normalize_language_code(lang)
        if not normalized:
            continue
        if normalized == base_lang:
            continue
        if normalized not in active_langs:
            active_langs.append(normalized)
    try:
        ttl_min = int(os.getenv("AI_EXPAND_TTL_MIN", "20") or 20)
    except Exception:
        ttl_min = 20
    cache_suffix = ""
    if active_langs:
        cache_suffix = f"|lang:{','.join(active_langs)}"
    key = _sha256(q + cache_suffix)
    now = _now()
    cached = AI_EXPAND_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_min * 60:
        cached_terms = list(cached[1])
        for phrase in mandatory:
            if phrase and phrase not in cached_terms:
                cached_terms.insert(0, phrase)
        cached_langs = cached[2] if len(cached) > 2 else []
        return cached_terms, cached_langs
    kws = []
    try:
        kws = call_lmstudio_keywords(q, "ai-search") or []
    except Exception:
        kws = []
    if not kws:
        toks = [t.strip() for t in re.split(r"[\s,;]+", q) if t.strip()]
        kws = toks[:12]
    seen = set()
    res: list[str] = []
    for w in kws:
        normalized = _normalize_keyword_candidate(w)
        if not normalized:
            continue
        if normalized in STOP_WORDS:
            continue
        if normalized not in seen:
            seen.add(normalized)
            res.append(normalized)
    lang_details: list[tuple[str, int]] = []
    if active_langs:
        extra_terms, extra_stats = _ai_expand_multilingual_terms(q, res, active_langs)
        for term in extra_terms:
            if term and term not in seen:
                seen.add(term)
                res.append(term)
        lang_details.extend(extra_stats)
    for phrase in reversed(mandatory):
        if phrase and phrase not in seen:
            res.insert(0, phrase)
            seen.add(phrase)
    AI_EXPAND_CACHE[key] = (now, res, lang_details)
    return res, lang_details


def _plan_search_keywords(
    query: str,
    *,
    base_terms: Sequence[str] | None = None,
    ai_terms: Sequence[str] | None = None,
    ttl_minutes: int = 15,
) -> Dict[str, List[str]]:
    """Запрашивает у LLM структурированный набор термов для классического поиска."""
    q = (query or "").strip()
    if not q:
        return {"must": [], "phrases": [], "broad": []}
    plan_key = _sha256(f"kw-plan::{q.lower()}")
    now = _now()
    cached = AI_KEYWORD_PLAN_CACHE.get(plan_key)
    if cached and (now - cached[0]) < ttl_minutes * 60:
        return cached[1]

    base_preview = ", ".join(sorted({t.lower() for t in (base_terms or []) if t})[:8])
    ai_preview = ", ".join(sorted({t.lower() for t in (ai_terms or []) if t})[:8])
    system = (
        "Ты помощник поиска. Получив вопрос, ты возвращаешь JSON с ключами 'must', "
        "'phrases', 'broad'. Не добавляй комментариев. Каждое значение — список строк. "
        "'must' — обязательные термины (точные фамилии, ключевые слова), чистые токены без пунктуации. "
        "'phrases' — короткие фразы (2-4 слова), пригодные для полнотекстового поиска. "
        "'broad' — синонимы и расширения, включая перевод на другой язык. "
        "Если нечего добавить в категорию — верни пустой список."
    )
    user = (
        f"Запрос: {q}\n"
        f"Базовые токены: [{base_preview}]\n"
        f"AI-термины: [{ai_preview}]\n"
        "Ответь строго JSON-объектом, например "
        "{\"must\": [\"термин\"], \"phrases\": [\"фраза\"], \"broad\": [\"синоним\"]}."
    )
    raw_response = ""
    try:
        raw_response = (call_lmstudio_compose(system, user, temperature=0.0, max_tokens=200) or "").strip()
    except Exception:
        raw_response = ""
    data: Dict[str, List[str]] = {"must": [], "phrases": [], "broad": []}
    parsed: Optional[Dict[str, object]] = None
    if raw_response:
        try:
            parsed = json.loads(raw_response)
        except Exception:
            match = re.search(r"\{.*\}", raw_response, flags=re.S)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = None
    if isinstance(parsed, dict):
        def _coerce_list(value: object) -> List[str]:
            if isinstance(value, list):
                return [str(item) for item in value if str(item).strip()]
            if isinstance(value, str):
                return [value]
            return []

        for key in ("must", "phrases", "broad"):
            items = _coerce_list(parsed.get(key))
            normalized: List[str] = []
            for item in items:
                norm = _normalize_keyword_candidate(item)
                if not norm:
                    continue
                if key == "phrases" and " " not in norm:
                    # фразовые термины должны содержать пробел
                    continue
                normalized.append(norm)
            data[key] = normalized
    if not data["must"]:
        data["must"] = [
            _normalize_keyword_candidate(tok)
            for tok in (base_terms or [])
            if _normalize_keyword_candidate(tok)
        ][:4]
    if not data["phrases"]:
        data["phrases"] = [
            _normalize_keyword_candidate(term)
            for term in (ai_terms or [])
            if term and " " in term and _normalize_keyword_candidate(term)
        ][:4]
    if not data["broad"]:
        data["broad"] = [
            _normalize_keyword_candidate(term)
            for term in (ai_terms or [])
            if term and _normalize_keyword_candidate(term)
        ][:6]
    AI_KEYWORD_PLAN_CACHE[plan_key] = (now, data)
    return data


def _read_cached_excerpt_for_file(f: File) -> str:
    try:
        # Предпочитаем фрагмент из базы данных
        if (f.text_excerpt or '').strip():
            return (f.text_excerpt or '')
        cache_dir = Path(app.static_folder) / 'cache' / 'text_excerpts'
        key = (f.sha1 or (f.rel_path or '').replace('/', '_')) + '.txt'
        fp = cache_dir / key
        if fp.exists():
            return fp.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        pass
    return ''


SNIPPET_WINDOW_RADIUS = 220


def _collect_snippets(text: str, terms: list[str], max_snips: int = 2) -> list[str]:
    t = (text or '')
    if not t:
        return []
    tl = t.lower()
    outs: list[tuple[int, str]] = []
    windows = []
    for raw in terms:
        term = (raw or '').strip()
        if not term:
            continue
        ql = term.lower()
        pos = 0
        found_any = False
        for _i in range(3):  # максимум 3 позиционирования на термин
            idx = tl.find(ql, pos)
            if idx < 0:
                break
            found_any = True
            start = max(0, idx - SNIPPET_WINDOW_RADIUS)
            end = min(len(t), idx + len(term) + SNIPPET_WINDOW_RADIUS)
            windows.append((start, end))
            pos = idx + len(term)
        if not found_any and len(ql) >= 3:
            # пытаемся разделить термин на подслова
            for part in re.split(r"[\s\-_/]+", ql):
                if len(part) < 3:
                    continue
                idx = tl.find(part)
                if idx >= 0:
                    start = max(0, idx - SNIPPET_WINDOW_RADIUS)
                    end = min(len(t), idx + len(part) + SNIPPET_WINDOW_RADIUS)
                    windows.append((start, end))
    # объединяем перекрывающиеся окна
    windows.sort()
    merged = []
    for w in windows:
        if not merged or w[0] > merged[-1][1] + 40:
            merged.append(list(w))
        else:
            merged[-1][1] = max(merged[-1][1], w[1])
    for a, b in merged[:max_snips]:
        snip = t[a:b]
        # схлопываем переводы строк для компактности
        snip = re.sub(r"\s+", " ", snip).strip()
        outs.append((a, snip))
    outs.sort(key=lambda x: x[0])
    return [s for _pos, s in outs]


def _split_text_chunks(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    if not text:
        return []
    chunk_chars = max(256, int(chunk_chars or 1024))
    max_chunks = max(1, int(max_chunks or 1))
    limit = min(len(text), chunk_chars * max_chunks)
    trimmed = text[:limit]
    return [trimmed[i:i+chunk_chars] for i in range(0, len(trimmed), chunk_chars)]


def _iter_document_chunks(path: Path, chunk_chars: int = 6000, max_chunks: int = 30):
    if not path or not path.exists() or not path.is_file():
        return
    chunk_chars = max(512, int(chunk_chars or 2000))
    max_chunks = max(1, int(max_chunks or 1))
    ext = path.suffix.lower()
    yielded = 0
    try:
        if ext in {'.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.yaml', '.yml', '.log'}:
            with path.open('r', encoding='utf-8', errors='ignore') as fh:
                while yielded < max_chunks:
                    chunk = fh.read(chunk_chars)
                    if not chunk:
                        break
                    yielded += 1
                    yield chunk
            return
        if ext == '.pdf' and fitz is not None:
            with fitz.open(path) as doc:
                buffer = ''
                for page in doc:
                    buffer += page.get_text('text') or ''
                    while len(buffer) >= chunk_chars and yielded < max_chunks:
                        yield buffer[:chunk_chars]
                        buffer = buffer[chunk_chars:]
                        yielded += 1
                        if yielded >= max_chunks:
                            return
                if buffer and yielded < max_chunks:
                    yield buffer[:chunk_chars]
            return
        if ext == '.docx' and docx is not None:
            document = docx.Document(str(path))
            buffer = ''
            for para in document.paragraphs:
                buffer += (para.text or '') + '\n'
                while len(buffer) >= chunk_chars and yielded < max_chunks:
                    yield buffer[:chunk_chars]
                    buffer = buffer[chunk_chars:]
                    yielded += 1
                    if yielded >= max_chunks:
                        return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return
        if ext == '.rtf' and rtf_to_text is not None:
            text = rtf_to_text(path.read_text(encoding='utf-8', errors='ignore'))
            for chunk in _split_text_chunks(text, chunk_chars, max_chunks):
                yield chunk
            return
        if ext == '.epub' and epub is not None:
            book = epub.read_epub(str(path))
            buffer = ''
            for item in book.get_items():
                if item.get_type() != epub.ITEM_DOCUMENT:
                    continue
                buffer += item.get_content().decode(errors='ignore')
                while len(buffer) >= chunk_chars and yielded < max_chunks:
                    yield buffer[:chunk_chars]
                    buffer = buffer[chunk_chars:]
                    yielded += 1
                    if yielded >= max_chunks:
                        return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return
        if ext == '.djvu' and djvu is not None:
            with djvu.decode.open(str(path)) as d:
                buffer = ''
                for page in d.pages:
                    buffer += page.get_text()
                    while len(buffer) >= chunk_chars and yielded < max_chunks:
                        yield buffer[:chunk_chars]
                        buffer = buffer[chunk_chars:]
                        yielded += 1
                        if yielded >= max_chunks:
                            return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return
    except Exception:
        pass
    text = ''
    try:
        if ext == '.pdf':
            text = extract_text_pdf(path, limit_chars=chunk_chars * max_chunks)
        elif ext == '.docx':
            text = extract_text_docx(path, limit_chars=chunk_chars * max_chunks)
        elif ext == '.rtf':
            text = extract_text_rtf(path, limit_chars=chunk_chars * max_chunks)
        elif ext == '.epub':
            text = extract_text_epub(path, limit_chars=chunk_chars * max_chunks)
        elif ext == '.djvu':
            text = extract_text_djvu(path, limit_chars=chunk_chars * max_chunks)
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        text = ''
    for chunk in _split_text_chunks(text, chunk_chars, max_chunks):
        yield chunk


def _deep_scan_file(file_obj: File, terms: list[str], raw_query: str, *, chunk_chars: int = 5000, max_chunks: int = 40, max_snippets: int = 3):
    if not file_obj or not file_obj.path:
        return {"hit": False, "snippets": [], "matched_terms": set(), "boost": 0.0, "sources": []}
    path = Path(file_obj.path)
    if not path.exists() or not path.is_file():
        return {"hit": False, "snippets": [], "matched_terms": set(), "boost": 0.0, "sources": []}
    term_map = {str(t).lower(): t for t in terms if t}
    raw_lower = (raw_query or '').strip().lower()
    phrase_pat = re.compile(re.escape(raw_lower), flags=re.IGNORECASE) if raw_lower and len(raw_lower) >= 3 else None
    snippets: list[str] = []
    matched: set[str] = set()
    sources_used: set[str] = set()
    # Дополнительные текстовые данные (транскрипты, резюме, подписи) из БД/тегов
    extra_texts: list[tuple[str, str]] = []
    try:
        if (file_obj.text_excerpt or '').strip():
            extra_texts.append(('excerpt', file_obj.text_excerpt))
        if (getattr(file_obj, 'abstract', None) or '').strip():
            extra_texts.append(('abstract', getattr(file_obj, 'abstract')))
        for tag in getattr(file_obj, 'tags', []) or []:
            try:
                key = str(getattr(tag, 'key', '') or '')
                val = str(getattr(tag, 'value', '') or '')
            except Exception:
                continue
            key_l = key.lower()
            if not val or len(val.strip()) < 40:
                continue
            if any(token in key_l for token in ('transcript', 'стенограм', 'caption', 'описан', 'vision', 'summary', 'ocr')):
                extra_texts.append((f'tag:{key}', val))
    except Exception:
        pass
    try:
        for chunk in _iter_document_chunks(path, chunk_chars=chunk_chars, max_chunks=max_chunks):
            if not chunk:
                continue
            chunk_lower = chunk.lower()
            chunk_hits = {orig for low, orig in term_map.items() if low and low in chunk_lower}
            phrase_hit = phrase_pat.search(chunk) if phrase_pat else None
            if not chunk_hits and not phrase_hit:
                continue
            sn = _collect_snippets(chunk, terms, max_snips=1)
            if sn:
                snippets.extend(sn)
            else:
                cleaned = re.sub(r"\s+", " ", chunk).strip()
                if cleaned:
                    snippets.append(cleaned[:300])
            matched.update(chunk_hits)
            sources_used.add('full-text')
            if len(snippets) >= max_snippets:
                break
    except Exception:
        pass
    # При необходимости обогащаем дополнительными текстовыми источниками (транскрипты, подписи, резюме)
    if len(snippets) < max_snippets:
        for label, text in extra_texts:
            if not text:
                continue
            sn = _collect_snippets(text, terms, max_snips=1)
            cleaned = ''
            if sn:
                for item in sn:
                    if item not in snippets:
                        snippets.append(item)
            else:
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned:
                    snippets.append(cleaned[:300])
            if (sn and sn[0]) or cleaned:
                sources_used.add(label)
            if len(snippets) >= max_snippets:
                break
    if not snippets:
        return {"hit": False, "snippets": [], "matched_terms": matched, "boost": 0.0, "sources": sorted(sources_used)}
    boost = 1.5 * min(len(snippets), max_snippets) + 0.4 * len(matched)
    return {
        "hit": True,
        "snippets": snippets[:max_snippets],
        "matched_terms": matched,
        "boost": boost,
        "sources": sorted(sources_used),
    }


# Базовый список русских/английских стоп-слов для исключения неинформативных токенов в ИИ-поиске
STOP_WORDS = set([
    # Русские местоимения, частицы, предлоги, союзы, служебные слова
    'и','или','а','но','же','то','ли','не','ни','да','уж','вот','как','так','что','кто','где','когда','зачем','почему','какой','какая','какие','каков','это','эта','этот','эти','того','тому','этом','этих','тех','там','тут','здесь','бы','либо','пусть','дабы','быть','есть','нет','между','через','после','перед','около','возле','у','к','ко','от','до','для','по','под','над','о','об','обо','при','без','из','из‑за','изза','с','со','же','ну','в','во','на','над','ё','же','уж','еще','ещё','также','тоже','сам','сама','сами','само','свой','своя','свои','своё','мой','моя','мои','моё','твой','твоя','твои','твоё','наш','наша','наши','наше','ваш','ваша','ваши','ваше','тот','та','то','те','эт','прочее','другой','другая','другие','иной','иная','иное','иные',
    # Английские
    'the','a','an','and','or','but','if','then','else','when','where','why','how','what','who','whom','whose','this','that','these','those','is','are','was','were','be','been','being','to','of','in','on','for','with','at','by','from','as','about','into','through','after','over','between','out','against','during','without','before','under','around','among','it','its','we','you','they','he','she','them','his','her','their','our','your','my','me','us','do','does','did','not','no','yes'
])



class _ProgressLogger:
    def __init__(self, emitter=None):
        self.lines: list[str] = []
        self.emitter = emitter

    def add(self, line: str) -> None:
        text = str(line)
        self.lines.append(text)
        try:
            app.logger.info(f"[ai-search] {text}")
        except Exception:
            pass
        if self.emitter:
            try:
                self.emitter(text)
            except Exception:
                pass



def _tokenize_query(q: str) -> list[str]:
    s = (q or '').lower()
    # оставляем буквы, цифры, дефис, подчёркивание; разделяем по остальным символам
    parts = re.split(r"[^\w\-]+", s)
    filtered: list[str] = []
    for p in parts:
        if not p:
            continue
        if len(p) < 2:
            continue
        # отбрасываем токены, состоящие только из цифр
        if p.isdigit():
            continue
        # стоп-слова
        if p in STOP_WORDS:
            continue
        filtered.append(p)
    # удаляем дубликаты, сохраняя порядок
    seen = set()
    out: list[str] = []
    for p in filtered:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:16]


def _idf_for_terms(terms: list[str]) -> dict[str, float]:
    # вычисляем частоты документов по объединению полей файла и тегов
    idf: dict[str, float] = {}
    try:
        allowed = getattr(g, 'allowed_collection_ids', set())
        total_q = db.session.query(func.count(File.id))
        if allowed is not None:
            if not allowed:
                N = 1
            else:
                total_q = total_q.filter(File.collection_id.in_(allowed))
                N = total_q.scalar() or 1
        else:
            N = total_q.scalar() or 1
    except Exception:
        N = 1
    for w in terms:
        like = f"%{w}%"
        try:
            q = db.session.query(func.count(func.distinct(File.id))) \
                .outerjoin(Tag, Tag.file_id == File.id) \
                .filter(or_(
                    File.title.ilike(like),
                    File.author.ilike(like),
                    File.keywords.ilike(like),
                    File.text_excerpt.ilike(like),
                    File.abstract.ilike(like),
                    Tag.value.ilike(like),
                    Tag.key.ilike(like),
                ))
            if allowed is not None:
                if not allowed:
                    df = 0
                    idf[w] = 1.0
                    continue
                q = q.filter(File.collection_id.in_(allowed))
            df = int(q.scalar() or 0)
        except Exception:
            df = 0
        # сглаживание add-one
        val = float((1.0 + (N / (1.0 + df))))
        # логарифмическая шкала, минимум 1.0
        try:
            import math
            val = max(1.0, math.log(val + 1.0))
        except Exception:
            val = 1.0
        idf[w] = val
    return idf


def _ai_search_core(data: dict | None, progress_cb=None) -> dict:
    data = data or {}
    query = (data.get('query') or '').strip()
    if not query:
        raise ValueError("query is required")
    try:
        top_k = int(data.get('top_k') or 10)
    except Exception:
        top_k = 10
    original_top_k = top_k
    top_k = max(1, min(5, top_k))
    deep_search = bool(data.get('deep_search'))
    user = getattr(g, 'current_user', None)
    t_total_start = time.monotonic()
    stage_start = t_total_start
    durations: dict[str, float] = {}
    file_lookup: Dict[int, File] = {}
    try:
        max_candidates = int(data.get('max_candidates') or 15)
    except Exception:
        max_candidates = 15
    max_candidates = max(5, min(max_candidates, 400))
    full_text = bool(data.get('full_text'))
    llm_snippets = bool(data.get('llm_snippets'))
    try:
        chunk_chars = int(data.get('chunk_chars') or 5000)
    except Exception:
        chunk_chars = 5000
    chunk_chars = max(500, min(chunk_chars, 20000))
    try:
        max_chunks = int(data.get('max_chunks') or 40)
    except Exception:
        max_chunks = 40
    max_chunks = max(1, min(max_chunks, 200))
    try:
        max_snippets = int(data.get('max_snippets') or 3)
    except Exception:
        max_snippets = 3
    max_snippets = max(1, min(max_snippets, 10))
    all_languages = bool(data.get('all_languages'))
    try:
        requested_languages = [str(x).strip() for x in (data.get('languages') or []) if str(x).strip()]
    except Exception:
        requested_languages = []
    if requested_languages:
        dedup_langs: list[str] = []
        seen_langs = set()
        for raw_lang in requested_languages:
            key = raw_lang.lower()
            if key in seen_langs:
                continue
            seen_langs.add(key)
            dedup_langs.append(raw_lang)
        requested_languages = dedup_langs
    else:
        requested_languages = []
    rag_enabled = data.get('use_rag')
    if rag_enabled is None:
        rag_enabled = False
    else:
        rag_enabled = bool(rag_enabled)
    progress = _ProgressLogger(progress_cb)
    if not rag_enabled:
        progress.add("RAG: режим отключён настройками запроса")
    query_hash = _query_fingerprint(query, data)
    mandatory_terms = _extract_quoted_phrases(query)
    mandatory_set = set(mandatory_terms)
    progress.add(f"Запрос: {query}")
    progress.add(f"Отпечаток запроса: {query_hash[:12]}…")
    if top_k != original_top_k:
        progress.add(f"Запрошено Top K = {original_top_k}, ограничиваем до {top_k}")
    else:
        progress.add(f"Top K = {top_k}")
    progress.add("Режим: " + ("глубокий" if deep_search else "быстрый"))
    if mandatory_terms:
        preview = ', '.join(f'«{term}»' for term in mandatory_terms[:4])
        if len(mandatory_terms) > 4:
            preview += '…'
        progress.add(f"Обязательные фразы: {preview}")
    else:
        progress.add("Обязательные фразы: нет")
    sources = data.get('sources') or {}
    use_tags = sources.get('tags', True) if isinstance(sources, dict) else True
    use_text = sources.get('text', True) if isinstance(sources, dict) else True
    progress.add(f"Источники: теги {'вкл' if use_tags else 'выкл'}, метаданные {'вкл' if use_text else 'выкл'}")
    progress.add(f"Параметры: кандидатов ≤ {max_candidates}, сниппетов ≤ {max_snippets}")
    progress.add(f"Контент: чанк {chunk_chars} симв., чанков ≤ {max_chunks}, full-text {'вкл' if full_text else 'выкл'}")
    progress.add(f"LLM сниппеты: {'вкл' if llm_snippets else 'выкл'}")
    if all_languages:
        if requested_languages:
            preview_langs = ', '.join(requested_languages[:5])
            if len(requested_languages) > 5:
                preview_langs += '…'
            progress.add(f"Языки: все ({preview_langs})")
        else:
            progress.add("Языки: все (нет языковых тегов)")
    else:
        progress.add("Языки: по умолчанию")
    # Необязательный фильтр по коллекциям (список id)
    if 'collection_id' in data and 'collection_ids' not in data:
        data['collection_ids'] = [data['collection_id']]
    try:
        collection_ids = [int(x) for x in (data.get('collection_ids') or []) if str(x).strip()]
    except Exception:
        collection_ids = []
    allowed = getattr(g, 'allowed_collection_ids', set())
    if allowed is not None:
        if collection_ids:
            collection_ids = [cid for cid in collection_ids if cid in allowed]
        else:
            collection_ids = list(allowed)
    # Необязательный фильтр по типам материалов (список строк)
    material_types = []
    try:
        material_types = [str(x).strip().lower() for x in (data.get('material_types') or []) if str(x).strip()]
    except Exception:
        material_types = []
    # Необязательный диапазон лет
    year_from = (data.get('year_from') or '').strip()
    year_to = (data.get('year_to') or '').strip()
    # Необязательные фильтры тегов: ["key=value", ...]
    tag_filters = []
    try:
        tag_filters = [str(x) for x in (data.get('tag_filters') or []) if '=' in str(x)]
    except Exception:
        tag_filters = []

    # Расширяем и токенизируем
    keywords, lang_details = _ai_expand_keywords(query, multi_lang=all_languages, target_langs=requested_languages)
    base_tokens = _tokenize_query(query)
    keyword_plan = _plan_search_keywords(
        query,
        base_terms=base_tokens,
        ai_terms=keywords,
    )
    extra_tokens: list[str] = []
    ai_phrase_terms: list[str] = []
    if lang_details:
        preview_langs = ', '.join(f"{code}(+{count})" for code, count in lang_details if count > 0)
        if preview_langs:
            progress.add(f"Доп. языки: {preview_langs}")
    plan_must = [
        term for term in keyword_plan.get("must", [])
        if term and term not in mandatory_set
    ]
    plan_phrases = [
        term for term in keyword_plan.get("phrases", []) if term
    ]
    plan_broad = [
        term for term in keyword_plan.get("broad", []) if term
    ]
    plan_must_set = {term for term in plan_must}
    for term in plan_phrases:
        if term not in ai_phrase_terms and term not in mandatory_set:
            ai_phrase_terms.append(term)
    for term in plan_broad:
        extra_tokens.extend(_tokenize_query(term))
    for w in keywords:
        normalized_kw = _normalize_keyword_candidate(w)
        if normalized_kw:
            if ' ' in normalized_kw and normalized_kw not in mandatory_set:
                ai_phrase_terms.append(normalized_kw)
        extra_tokens.extend(_tokenize_query(w))
    if plan_must:
        progress.add("LLM ключи (строгие): " + ", ".join(plan_must[:6]) + ("…" if len(plan_must) > 6 else ""))
    if plan_phrases:
        progress.add("LLM фразы: " + ", ".join(plan_phrases[:6]) + ("…" if len(plan_phrases) > 6 else ""))
    if plan_broad:
        progress.add("LLM расширения: " + ", ".join(plan_broad[:6]) + ("…" if len(plan_broad) > 6 else ""))
    # уникальные термины (токены) с сохранением порядка, отдаём приоритет обязательным и LLM-фразам
    seen = set()
    terms: list[str] = []
    ordered_candidates = mandatory_terms + plan_must + ai_phrase_terms + base_tokens + extra_tokens + plan_broad
    for w in ordered_candidates:
        if not w:
            continue
        if w in seen:
            continue
        seen.add(w)
        terms.append(w)
    original_terms = list(terms)
    filtered_out_terms: list[str] = []
    if not terms and query:
        # запасной вариант: используем исходный запрос как термин
        terms = _tokenize_query(query) or [query.lower()]
    if terms:
        preview = ', '.join(terms[:6])
        if len(terms) > 6:
            preview += '…'
        progress.add(f"Ключевые термины: {preview}")
    else:
        progress.add("Ключевые термины: не выделены")

    # Предварительно считаем IDF по каждому термину
    idf = _idf_for_terms(terms)
    filtered_terms: list[str] = []
    removed_low_idf: list[str] = []
    for w in terms:
        if w in mandatory_set or w in plan_must_set:
            filtered_terms.append(w)
            continue
        if idf.get(w, 1.0) >= KEYWORD_IDF_MIN:
            filtered_terms.append(w)
        else:
            removed_low_idf.append(w)
    if removed_low_idf:
        filtered_out_terms.extend([w for w in removed_low_idf if w not in filtered_out_terms])
        progress.add(f"Фильтр ключевых слов: убрано {len(removed_low_idf)} общеупотребительных терминов")
    if not filtered_terms:
        fallback_terms = [w for w in terms if (w in mandatory_set) or len(w) >= 4]
        if fallback_terms:
            removed = [w for w in terms if w not in fallback_terms]
            if removed:
                filtered_out_terms.extend([w for w in removed if w not in filtered_out_terms])
            progress.add("Фильтр ключевых слов: оставляем только более длинные термины")
            filtered_terms = fallback_terms
    if filtered_terms:
        terms = filtered_terms
    try:
        feedback_rows = AiSearchKeywordFeedback.query.filter_by(query_hash=query_hash).all()
    except Exception:
        feedback_rows = []

    if terms:
        banned_terms = {str(row.keyword).lower() for row in feedback_rows if getattr(row, 'keyword', None) and row.action == 'irrelevant'}
        if banned_terms:
            before_len = len(terms)
            kept_terms: list[str] = []
            removed: list[str] = []
            for w in terms:
                if w in banned_terms and w not in mandatory_set:
                    removed.append(w)
                else:
                    kept_terms.append(w)
            terms = kept_terms
            if removed:
                filtered_out_terms.extend([w for w in removed if w not in filtered_out_terms])
            if len(terms) < before_len:
                progress.add(f"Фильтр обратной связи: игнорируем {before_len - len(terms)} термина по отзывам")
    durations['keywords'] = time.monotonic() - stage_start
    stage_start = time.monotonic()

    # Накапливаем кандидатов с оценками и попаданиями
    scores: dict[int, float] = {}
    hits: dict[int, list[dict]] = {}

    term_hits: dict[int, set[str]] = {}
    def add_score(fid: int, delta: float, hit: dict | None = None, term: str | None = None):
        scores[fid] = scores.get(fid, 0.0) + float(delta)
        if hit:
            hits.setdefault(fid, []).append(hit)
        if term:
            s = term_hits.setdefault(fid, set())
            if term:
                s.add(term)

    # Дополнительные кандидаты через FTS по ключевым терминам (учёт вопросительных слов)
    fts_candidates: List[int] = []
    fts_seed_terms = sorted(terms, key=lambda t: idf.get(t, 0.0), reverse=True)
    if not fts_seed_terms:
        backup_tokens = _tokenize_query(query)
        if backup_tokens:
            fts_seed_terms = sorted(
                backup_tokens,
                key=lambda t: idf.get(t, 0.0),
                reverse=True,
            )
    fts_seed_terms = [t for t in fts_seed_terms if len(t) >= 3][:5]
    fts_seed_query = " ".join(fts_seed_terms) if fts_seed_terms else query.strip()
    if fts_seed_query:
        try:
            fts_candidates = search_service.candidate_ids(fts_seed_query, limit=max(200, max_candidates * 10)) or []
        except Exception:
            fts_candidates = []
    if fts_candidates:
        base_term = fts_seed_terms[0] if fts_seed_terms else None
        if base_term:
            progress.add(f"FTS кандидаты: {len(fts_candidates)} (ядро: {base_term})")
        else:
            progress.add(f"FTS кандидаты: {len(fts_candidates)}")
        fts_cap = min(len(fts_candidates), max_candidates * 10)
        for fid in fts_candidates[:fts_cap]:
            if base_term:
                add_score(
                    fid,
                    AI_SCORE_FTS * idf.get(base_term, 1.0),
                    {"type": "fts", "term": base_term},
                    term=base_term,
                )
            else:
                add_score(
                    fid,
                    AI_SCORE_FTS,
                    {"type": "fts"},
                )
    else:
        progress.add("FTS кандидаты: не обнаружены")

    raw_query_clean = query.strip()
    raw_fts_candidates: List[int] = []
    if raw_query_clean:
        try:
            raw_fts_candidates = search_service.candidate_ids(raw_query_clean, limit=max(200, max_candidates * 10)) or []
        except Exception:
            raw_fts_candidates = []
    if raw_fts_candidates:
        primary_term = next((tok for tok in base_tokens if len(tok) >= 3), None)
        if not primary_term and terms:
            primary_term = next((tok for tok in terms if len(tok) >= 3), None)
        label_term = primary_term or raw_query_clean.lower()
        progress.add(f"FTS оригинальный запрос: {len(raw_fts_candidates)} кандидатов")
        fts_raw_cap = min(len(raw_fts_candidates), max_candidates * 10)
        for fid in raw_fts_candidates[:fts_raw_cap]:
            add_score(
                fid,
                AI_SCORE_FTS_RAW * (idf.get(label_term, 1.0) if isinstance(label_term, str) else 1.0),
                {"type": "fts_raw", "term": label_term},
                term=str(label_term) if isinstance(label_term, str) else None,
            )
    else:
        progress.add("FTS оригинальный запрос: не обнаружены")

    # Совпадения по тегам
    if use_tags:
        for w in terms:
            like = f"%{w}%"
            try:
                q = (
                    db.session.query(Tag.file_id, Tag.key, Tag.value)
                    .join(File, Tag.file_id == File.id)
                    .outerjoin(Collection, File.collection_id == Collection.id)
                    .filter(or_(Collection.searchable == True, Collection.id.is_(None)))
                    .filter(or_(Tag.value.ilike(like), Tag.key.ilike(like)))
                )
                if collection_ids:
                    q = q.filter(File.collection_id.in_(collection_ids))
                if material_types:
                    q = q.filter(func.lower(File.material_type).in_(material_types))
                if year_from:
                    q = q.filter(File.year >= year_from)
                if year_to:
                    q = q.filter(File.year <= year_to)
                rows = q \
                     .limit(4000).all()
                for fid, k, v in rows:
                    add_score(fid, AI_SCORE_TAG * idf.get(w, 1.0), {"type": "tag", "key": k, "value": v, "term": w}, term=w)
            except Exception:
                pass
        progress.add(f"Теги: найдено кандидатов {len(scores)}")
    else:
        progress.add("Теги: пропущено (источник отключён)")

    # Совпадения по полям файла
    if use_text:
        for w in terms:
            like = f"%{w}%"
            try:
                q = (
                    db.session.query(
                        File.id,
                        File.title,
                        File.author,
                        File.keywords,
                        File.text_excerpt,
                        File.abstract,
                    )
                    .outerjoin(Collection, File.collection_id == Collection.id)
                    .filter(or_(Collection.searchable == True, Collection.id.is_(None)))
                    .filter(
                        or_(
                            File.title.ilike(like),
                            File.author.ilike(like),
                            File.keywords.ilike(like),
                            File.text_excerpt.ilike(like),
                            File.abstract.ilike(like),
                        )
                    )
                )
                if collection_ids:
                    q = q.filter(File.collection_id.in_(collection_ids))
                if material_types:
                    q = q.filter(func.lower(File.material_type).in_(material_types))
                if year_from:
                    q = q.filter(File.year >= year_from)
                if year_to:
                    q = q.filter(File.year <= year_to)
                rows = q.limit(4000).all()
                for fid, title, author, kws, excerpt, abstract in rows:
                    if title and re.search(re.escape(w), title, flags=re.I):
                        add_score(fid, AI_SCORE_TITLE * idf.get(w, 1.0), {"type": "title", "term": w}, term=w)
                    if author and re.search(re.escape(w), author, flags=re.I):
                        add_score(fid, AI_SCORE_AUTHOR * idf.get(w, 1.0), {"type": "author", "term": w}, term=w)
                    if kws and re.search(re.escape(w), kws, flags=re.I):
                        add_score(fid, AI_SCORE_KEYWORDS * idf.get(w, 1.0), {"type": "keywords", "term": w}, term=w)
                    if excerpt and re.search(re.escape(w), excerpt, flags=re.I):
                        add_score(fid, AI_SCORE_EXCERPT * idf.get(w, 1.0), {"type": "excerpt", "term": w}, term=w)
                    if abstract and re.search(re.escape(w), abstract, flags=re.I):
                        add_score(fid, AI_SCORE_ABSTRACT * idf.get(w, 1.0), {"type": "abstract", "term": w}, term=w)
            except Exception:
                pass
        progress.add(f"Метаданные: найдено кандидатов {len(scores)}")
    else:
        progress.add("Метаданные: пропущено (источник отключён)")
    if plan_must:
        coverage_lines: List[str] = []
        for key in plan_must[:6]:
            count = sum(1 for matched in term_hits.values() if key in matched)
            coverage_lines.append(f"{key}:{count}")
        progress.add("Покрытие must-термов: " + (", ".join(coverage_lines) or "нет совпадений"))
    if mandatory_set and scores:
        kept_scores: dict[int, float] = {}
        kept_hits: dict[int, list[dict]] = {}
        kept_term_hits: dict[int, set[str]] = {}
        dropped = 0
        for fid, score_val in scores.items():
            matched = term_hits.get(fid, set())
            if mandatory_set.issubset(matched):
                kept_scores[fid] = score_val
                if fid in hits:
                    kept_hits[fid] = hits[fid]
                if matched:
                    kept_term_hits[fid] = matched
            else:
                dropped += 1
        if dropped:
            progress.add(f"Обязательные фразы: исключено {dropped} кандидатов без совпадений")
        scores = kept_scores
        hits = kept_hits
        term_hits = kept_term_hits
    durations['candidates'] = time.monotonic() - stage_start
    stage_start = time.monotonic()

    # Формируем результаты
    file_ids = list(scores.keys())
    results = []
    if file_ids:
        q_files = (
            File.query.outerjoin(Collection, File.collection_id == Collection.id)
            .filter(or_(Collection.searchable == True, Collection.id.is_(None)))
            .filter(File.id.in_(file_ids))
        )
        if collection_ids:
            q_files = q_files.filter(File.collection_id.in_(collection_ids))
        if material_types:
            q_files = q_files.filter(func.lower(File.material_type).in_(material_types))
        if year_from:
            q_files = q_files.filter(File.year >= year_from)
        if year_to:
            q_files = q_files.filter(File.year <= year_to)
        # Применяем фильтры тегов на финальном этапе
        for tf in tag_filters:
            try:
                k, v = tf.split('=', 1)
                k = (k or '').strip(); v = (v or '').strip()
                if not k or not v:
                    continue
                if k.lower() == 'author':
                    q_files = q_files.filter(File.author.ilike(f"%{v}%"))
                else:
                    tflt = aliased(Tag)
                    q_files = q_files.join(tflt, tflt.file_id == File.id).filter(and_(tflt.key == k, tflt.value.ilike(f"%{v}%")))
            except Exception:
                continue
        q_files = q_files.all()
        id2file = {f.id: f for f in q_files}
        file_lookup = id2file
        for fid, sc in scores.items():
            f = id2file.get(fid)
            if not f:
                continue
            # усиление за точную фразу (исходный запрос как фраза)
            phrase_boost = 0.0
            qraw = query.strip()
            if len(qraw) >= 3:
                try:
                    pat = re.escape(qraw)
                    if f.title and re.search(pat, f.title, flags=re.I):
                        phrase_boost += AI_BOOST_PHRASE
                    if f.keywords and re.search(pat, f.keywords, flags=re.I):
                        phrase_boost += AI_BOOST_PHRASE * 0.6
                except Exception:
                    pass
            # усиление за покрытие разных терминов
            n_terms = len(term_hits.get(fid, set()))
            coverage_boost = max(0, n_terms - 1) * AI_BOOST_MULTI
            # сниппеты из кэшированного фрагмента/аннотации
            snips = []
            snippet_sources: list[str] = []
            try:
                text_cache = _read_cached_excerpt_for_file(f)
                snips = _collect_snippets(text_cache, terms, max_snips=max_snippets) if text_cache else []
                if snips:
                    snippet_sources.append('excerpt-cache')
            except Exception:
                snips = []
            if (not snips or len(snips) < max_snippets) and getattr(f, 'abstract', None):
                try:
                    extra_count = max(1, max_snippets - len(snips))
                    extra = _collect_snippets(getattr(f, 'abstract'), terms, max_snips=extra_count)
                    added = False
                    for item in extra:
                        if item and item not in snips:
                            snips.append(item)
                            added = True
                    if added:
                        snippet_sources.append('abstract')
                except Exception:
                    pass
            # усиление за близость: несколько терминов в одном сниппете
            prox_boost = 0.0
            if snips:
                for s in snips:
                    present = 0
                    sl = s.lower()
                    for w in terms:
                        if w in sl:
                            present += 1
                    if present >= 2:
                        prox_boost += (present - 1) * AI_BOOST_SNIPPET_COOCCUR
            results.append({
                "file_id": fid,
                "rel_path": f.rel_path,
                "title": f.title or f.filename,
                "score": round(sc + phrase_boost + coverage_boost + prox_boost, 3),
                "hits": hits.get(fid, []),
                "snippets": snips,
                "snippet_sources": snippet_sources,
            })
        # сортируем по убыванию балла, затем по дате изменения
        results.sort(key=lambda x: (x.get('score') or 0.0, id2file.get(x['file_id']).mtime or 0.0), reverse=True)
        total_ranked = len(results)
        progress.add(f"Ранжирование: {total_ranked} кандидатов")
        if total_ranked > max_candidates:
            progress.add(f"Ограничиваем до {max_candidates} лучших по баллу")
            results = results[:max_candidates]
        dismissed_file_ids = {int(row.file_id) for row in feedback_rows if getattr(row, 'file_id', None) and row.action == 'irrelevant'}
        if dismissed_file_ids:
            before_len = len(results)
            results = [res for res in results if int(res.get('file_id') or 0) not in dismissed_file_ids]
            after_len = len(results)
            if after_len < before_len:
                progress.add(f"Обратная связь: исключено {before_len - after_len} источников, помеченных как нерелевантные")

        for idx, res in enumerate(results, start=1):
            cand_title = (res.get('title') or res.get('rel_path') or f"file-{res.get('file_id')}").strip()
            progress.add(f"Кандидат [{idx}]: {cand_title}")
        if (deep_search or full_text) and results:
            if deep_search and full_text:
                scan_label = "Глубокий поиск + full-text"
                scan_step = "Глубокий поиск"
            elif deep_search:
                scan_label = "Глубокий поиск по контенту"
                scan_step = "Глубокий поиск"
            else:
                scan_label = "Полнотекстовый поиск"
                scan_step = "Full-text"
            scan_limit = len(results) if full_text else min(len(results), max(top_k * 2, 5))
            progress.add(f"{scan_label}: проверяем {scan_limit} файлов (чанк {chunk_chars}, чанков ≤ {max_chunks})")
            for idx, res in enumerate(results[:scan_limit], start=1):
                f = id2file.get(res['file_id'])
                if not f:
                    continue
                scan = _deep_scan_file(f, terms, query, chunk_chars=chunk_chars, max_chunks=max_chunks, max_snippets=max_snippets)
                boost = float(scan.get('boost') or 0.0)
                if boost:
                    res['score'] = round((res.get('score') or 0.0) + boost, 3)
                matched_terms = scan.get('matched_terms') or set()
                if matched_terms:
                    res['matched_terms'] = sorted({str(t) for t in matched_terms})
                if scan.get('hit'):
                    new_snips = scan.get('snippets') or []
                    if new_snips:
                        base_snips = res.get('snippets') or []
                        combined = base_snips + [sn for sn in new_snips if sn not in base_snips]
                        res['snippets'] = combined[:max_snippets]
                    sources = scan.get('sources') or []
                    if sources:
                        current_sources = set(res.get('snippet_sources') or [])
                        for src in sources:
                            if src not in current_sources:
                                current_sources.add(src)
                        res['snippet_sources'] = sorted(current_sources)
                title = (f.title or f.filename or f.rel_path or f"file-{f.id}").strip()
                srcs = ", ".join(scan.get('sources') or [])
                status = 'совпадения' if scan.get('hit') else 'нет'
                if srcs:
                    status += f"; источники: {srcs}"
                progress.add(f"{scan_step} [{idx}/{scan_limit}]: {title} — {status}")
            results.sort(key=lambda x: (x.get('score') or 0.0, id2file.get(x['file_id']).mtime or 0.0), reverse=True)
        durations['deep'] = time.monotonic() - stage_start
        stage_start = time.monotonic()
        results = results[:top_k]
        progress.add(f"Отобрано документов: {len(results)}")
        for idx, res in enumerate(results, start=1):
            title = (res.get('title') or res.get('rel_path') or f"file-{res.get('file_id')}").strip()
            sources = res.get('snippet_sources') or []
            if not sources:
                summary = 'не найдены'
            else:
                summary = ', '.join(sources)
            progress.add(f"Снипеты [{idx}] {title}: {summary}")
        if llm_snippets and results:
            total_targets = len(results)
            progress.add(f"LLM сниппеты: формируем для {total_targets} документов")
            for idx, res in enumerate(results, start=1):
                title = (res.get('title') or res.get('rel_path') or f"file-{res.get('file_id')}").strip()
                base_snippets = res.get('snippets') or []
                context = " ".join(base_snippets).strip()
                if not context:
                    progress.add(f"LLM сниппеты [{idx}/{total_targets}]: {title} — пропущено (нет контекста)")
                    continue
                cache_entry = _get_cached_snippet(res.get('file_id'), query_hash, True)
                if cache_entry and cache_entry.snippet:
                    translated_snippet, translated = _ensure_russian_text(cache_entry.snippet, label='snippet-cache')
                    res['llm_snippet'] = translated_snippet
                    progress.add(f"LLM сниппеты [{idx}/{total_targets}]: {title} — cache hit" + (' (перевод)' if translated else ''))
                    if translated:
                        _store_snippet_cache(res.get('file_id'), query_hash, True, translated_snippet, meta={'len': len(translated_snippet)}, ttl_hours=SNIPPET_CACHE_TTL_HOURS)
                    continue
                ctx = context[:1200]
                system = (
                    "Ты создаёшь короткий сниппет (до двух предложений) для поисковой выдачи. "
                    "Используй только предоставленный текст, не добавляй новых фактов. "
                    "Отвечай на русском языке."
                )
                user_msg = (
                    f"Запрос: {query}\n"
                    f"Фрагменты документа:\n{ctx}\n"
                    "Сформулируй 1-2 предложения с ключевыми фактами и сохрани смысл. Ответ дай на русском языке."
                )
                snippet_text = ""
                error_logged = False
                try:
                    snippet_text = (call_lmstudio_compose(system, user_msg, temperature=0.0, max_tokens=min(180, _lm_max_output_tokens())) or '').strip()
                except Exception as exc:
                    error_logged = True
                    progress.add(f"LLM сниппеты [{idx}/{total_targets}]: {title} — ошибка: {exc}")
                if snippet_text:
                    snippet_text, translated = _ensure_russian_text(snippet_text, label='snippet')
                    res['llm_snippet'] = snippet_text
                    note = f" ({len(snippet_text)} символов)"
                    if translated:
                        note += ", переведено"
                    progress.add(f"LLM сниппеты [{idx}/{total_targets}]: {title} — готово{note}")
                    _store_snippet_cache(res.get('file_id'), query_hash, True, snippet_text, meta={'len': len(snippet_text)})
                elif not error_logged:
                    progress.add(f"LLM сниппеты [{idx}/{total_targets}]: {title} — пустой ответ")
            durations['llm_snippets'] = time.monotonic() - stage_start
            stage_start = time.monotonic()
    else:
        progress.add("Совпадения не найдены")

    # Необязательный короткий ответ, используя сниппеты как контекст (поисковый промпт)
    answer = ""
    rag_context_sections: List[ContextSection] = []
    rag_validation: Optional[ValidationResult] = None
    rag_session_id: Optional[int] = None
    rag_bundle: Optional[Dict[str, Any]] = None
    rag_notes: list[str] = []
    context_chunk_limit = max(4, min(8, top_k * 2))
    language_filters = requested_languages if requested_languages else None
    if rag_enabled:
        if results:
            rag_bundle, rag_notes = _prepare_rag_context(
                query,
                results,
                language_filters=language_filters,
                max_chunks=context_chunk_limit,
                progress=progress,
            )
        else:
            rag_notes.append("Нет результатов поиска, RAG контекст пропущен.")
    else:
        pass
    fallback_required = True
    if rag_enabled and rag_bundle and rag_bundle.get("sections"):
        fallback_required = False
        rag_notes.extend(rag_bundle.get("notes", []))
        stage_start = time.monotonic()
        answer, rag_validation, rag_session_id, _, _ = _generate_rag_answer(
            query,
            rag_bundle,
            temperature=0.1,
            max_tokens=min(350, _lm_max_output_tokens()),
            progress=progress,
        )
        durations['llm_answer'] = time.monotonic() - stage_start
        rag_context_sections = rag_bundle.get("sections", [])
    if fallback_required:
        stage_start = time.monotonic()
        if rag_notes:
            prefix = "RAG заметка" if not rag_enabled else "RAG fallback"
            for note in rag_notes:
                progress.add(f"{prefix}: {note}")
        topn = results[:10]
        snippet_context: List[Dict[str, Any]] = []
        if topn:
            core_terms = [term for term in base_tokens if len(term) >= 4] or [
                term for term in terms if len(term) >= 4
            ]
            anchor_terms = [term for term in base_tokens if len(term) >= 3] or [query.lower()]
            snippet_answer, used_count, snippet_context = _compose_snippet_based_answer(
                query,
                topn,
                primary_terms=core_terms,
                anchor_terms=anchor_terms,
                term_idf=idf,
                max_items=max(3, top_k),
            )
            if snippet_answer:
                llm_snippet_answer = _generate_snippet_llm_summary(
                    query,
                    snippet_context,
                    progress=progress,
                )
                if llm_snippet_answer:
                    if snippet_context and re.search(r"недостаточн[оы]", llm_snippet_answer, flags=re.IGNORECASE):
                        # Считаем, что LLM осторожничает — используем консервативный ответ по сниппетам
                        answer = snippet_answer
                        progress.add("LLM ответ по сниппетам отклонён: сообщает о нехватке данных")
                    else:
                        answer = llm_snippet_answer
                        progress.add(f"LLM ответ по сниппетам: использовано {len(snippet_context)} источников")
                else:
                    answer = snippet_answer
                    progress.add(f"Сниппет-ответ: использовано {used_count} источников (LLM не применён)")
            else:
                answer = fallback_answer()
                progress.add("Сниппет-ответ: подходящих фрагментов нет, возвращаем заглушку")
        else:
            answer = fallback_answer()
            progress.add("Сниппет-ответ: список результатов пуст")
        if snippet_context:
            detail_refs = []
            for item in snippet_context:
                title = item.get("title") or f"источник {item.get('order')}"
                detail_refs.append(f"[{item['order']}] {title}")
            if detail_refs:
                details_text = "Подробнее см.: " + "; ".join(detail_refs)
                answer = (answer + "\n\n" + details_text) if answer else details_text
        durations['llm_answer'] = time.monotonic() - stage_start
        stage_start = time.monotonic()

    # Необязательно: лёгкое реранжирование топ-15 через LLM с контекстом сниппетов
    if _rt().ai_rerank_llm and results:
        try:
            top = results[:15]
            prompt_lines = [f"[{i+1}] id={it['file_id']} :: { (it.get('snippets') or [''])[0] }" for i, it in enumerate(top)]
            prompt = "\n".join(prompt_lines)
            sys = "Ты ранжируешь источники по релевантности к запросу. Верни JSON-массив id в порядке убывания релевантности."
            user_prompt = f"Запрос: {query}\nИсточники:\n{prompt}\nОтвети только JSON массивом id."
            order = None
            last_error: Exception | None = None
            for choice in _llm_iter_choices('rerank'):
                label = _llm_choice_label(choice)
                url = _llm_choice_url(choice)
                if not url:
                    app.logger.warning(f"LLM rerank endpoint некорректен ({label}): пустой URL")
                    continue
                payload = {
                    "model": choice.get('model') or LMSTUDIO_MODEL,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                }
                try:
                    rr = http_request(
                        'POST',
                        url,
                        headers=_llm_choice_headers(choice),
                        json=payload,
                        timeout=(HTTP_CONNECT_TIMEOUT, max(HTTP_DEFAULT_TIMEOUT, 60)),
                        logger=app.logger,
                    )
                    if _llm_response_indicates_busy(rr):
                        app.logger.info(f"LLM rerank endpoint занята ({label}), переключаемся")
                        last_error = RuntimeError('busy')
                        continue
                    rr.raise_for_status()
                    dd = rr.json()
                    content = (dd.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
                    try:
                        order = json.loads(content)
                    except Exception:
                        m = re.search(r"\[(?:\s*\d+\s*,?\s*)+\]", content)
                        if m:
                            order = json.loads(m.group(0))
                    if order is not None:
                        break
                except Exception as e:
                    last_error = e
                    if isinstance(e, RuntimeError) and str(e) == 'llm_cache_only_mode':
                        app.logger.info(f"LLM rerank пропущен (cache-only, {label})")
                    else:
                        app.logger.warning(f"LLM rerank failed ({label}): {e}")
                    continue
            if order is None and last_error and str(last_error) not in {'busy', 'llm_cache_only_mode'}:
                app.logger.warning(f"LLM rerank failed: {last_error}")
            if isinstance(order, list) and all(isinstance(x, int) for x in order):
                pos = {int(fid): i for i, fid in enumerate(order)}
                results.sort(key=lambda x: (pos.get(int(x['file_id']), 10**6), -(x.get('score') or 0.0)))
                progress.add("Реранжирование LLM: порядок обновлён")
        except Exception:
            pass

    term_usage_counts: Dict[str, int] = {}
    for matched in term_hits.values():
        for term in matched:
            term_usage_counts[term] = term_usage_counts.get(term, 0) + 1
    final_keywords: List[str] = []
    seen_keywords: set[str] = set()
    base_tokens_limit = base_tokens[:4]
    for term in terms:
        if term in seen_keywords:
            continue
        if (
            term_usage_counts.get(term, 0) > 0
            or term in mandatory_set
            or term in base_tokens_limit
        ):
            final_keywords.append(term)
            seen_keywords.add(term)
    if not final_keywords:
        for term in base_tokens + list(terms) + plan_must:
            if term and term not in seen_keywords:
                final_keywords.append(term)
                seen_keywords.add(term)
            if len(final_keywords) >= 6:
                break
    filtered_keywords = sorted({w for w in filtered_out_terms if w})
    durations['total'] = time.monotonic() - t_total_start
    rag_context_payload = [_serialize_context_section(section) for section in rag_context_sections]
    extra_meta = {
        'query_preview': query[:200],
        'answer_preview': (answer or '')[:400],
        'keyword_count': len(final_keywords),
        'filtered_keywords': filtered_keywords,
        'mandatory_terms': mandatory_terms,
    }
    extra_meta.update({
        'rag_context_count': len(rag_context_sections),
        'rag_fallback': bool(rag_context_sections) is False and rag_enabled,
        'rag_hallucination_warning': bool(rag_validation and rag_validation.hallucination_warning),
    })
    _record_search_metric(query_hash, durations, user, extra_meta)

    return {
        "query": query,
        "query_hash": query_hash,
        "keywords": final_keywords,
        "filtered_keywords": filtered_keywords,
        "mandatory_terms": mandatory_terms,
        "answer": answer,
        "items": results,
        "progress": progress.lines,
        "rag_context": rag_context_payload,
        "rag_validation": rag_validation.as_dict() if rag_validation else None,
        "rag_session_id": rag_session_id,
        "rag_notes": rag_notes,
        "rag_fallback": bool(rag_context_sections) is False and rag_enabled,
        "rag_enabled": rag_enabled,
    }


def _resolve_problem_score(problem: Dict[str, Any]) -> Optional[float]:
    score_keys = (
        'score',
        'humanScore',
        'human_score',
        'remoteScore',
        'remote_score',
        'autoScore',
        'auto_score',
    )
    for key in score_keys:
        if key in problem:
            value = problem.get(key)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                continue
    return None


def _split_text_for_dataset(
    text: str,
    *,
    min_chars: int = 120,
    max_chars: int = 700,
    max_segments: int = 6,
) -> List[str]:
    cleaned = (text or '').strip()
    if not cleaned:
        return []

    # Сначала пробуем разбить по пустым строкам
    candidates = [segment.strip() for segment in re.split(r"\n\s*\n+", cleaned) if segment.strip()]

    segments: List[str] = []
    for block in candidates:
        if len(block) <= max_chars:
            if len(block) >= min_chars:
                segments.append(block)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", block)
        buffer: List[str] = []
        current = ''
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            tentative = f"{current} {sentence}".strip() if current else sentence
            if len(tentative) > max_chars:
                if current:
                    segments.append(current.strip())
                if len(sentence) >= min_chars:
                    segments.append(sentence[:max_chars].strip())
                current = ''
            else:
                current = tentative
        if current and len(current) >= min_chars:
            segments.append(current.strip())

    if not segments:
        if len(cleaned) >= min_chars:
            segments = [cleaned[:max_chars]]

    return segments[:max_segments]


def _extract_pairs_from_response(raw: str) -> List[Dict[str, str]]:
    if not raw:
        return []
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return []
    snippet = raw[start:end + 1]
    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError:
        try:
            snippet = re.sub(r".*?(\{.*\})", r"\1", raw, flags=re.S)
            payload = json.loads(snippet)
        except Exception:
            return []
    items = payload.get('pairs')
    if not isinstance(items, list):
        return []
    pairs: List[Dict[str, str]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get('question') or '').strip()
        answer = str(entry.get('answer') or '').strip()
        if question and answer:
            pairs.append({'question': question, 'answer': answer})
    return pairs


def _generate_pairs_from_snippet(
    snippet: str,
    *,
    desired_pairs: int,
    temperature: float,
    max_tokens: int,
    problem_question: Optional[str],
) -> List[Dict[str, str]]:
    paragraph = (snippet or '').strip()
    if not paragraph:
        return []

    system_prompt = (
        'Ты — помощник по подготовке данных для обучения. Отвечай на русском языке, '
        'формируй содержательные вопросы и ответы и соблюдай формат JSON без дополнительных комментариев.'
    )
    intro = (
        "Тебе дан абзац текста. Сформируй {count} пар \"вопрос-ответ\" для тонкой настройки модели. "
        "В каждом вопросе делай акцент на фактах, причинно-следственных связях, числах или важных утверждениях из абзаца. "
        "Не упоминай, что вопрос взят из текста. Ответ должен быть вытекающим только из абзаца.\n\n"
    )
    if problem_question:
        intro += f"Проблемный пользовательский вопрос: {problem_question.strip()}\n"

    user_prompt = (
        f"{intro}Абзац:\n\"\"\"\n{paragraph}\n\"\"\"\n\n"
        "Формат ответа:\n"
        "{\n  \"pairs\": [\n    { \"question\": \"...\", \"answer\": \"...\" }\n  ]\n}\n"
        f"Количество элементов в массиве должно быть ровно {desired_pairs}. Не добавляй других полей."
    )

    try:
        raw = call_lmstudio_compose(
            system_prompt,
            user_prompt,
            temperature=max(0.0, float(temperature)),
            max_tokens=max(128, int(max_tokens)),
        )
    except Exception as exc:
        app.logger.warning(f"QA generation failed: {exc}")
        return []

    pairs = _extract_pairs_from_response(raw)
    return pairs


def _launch_fine_tune_job(
    dataset: Sequence[Dict[str, str]],
    fine_tune_config: Dict[str, Any],
) -> Dict[str, Any]:
    server_url = (fine_tune_config.get('server_url') or '').strip()
    base_model_path = (fine_tune_config.get('base_model_path') or '').strip()
    config_payload = fine_tune_config.get('config') or {}
    if not server_url or not base_model_path or not isinstance(config_payload, dict):
        raise ValueError('server_url, base_model_path и config обязательны для запуска дообучения')

    payload: Dict[str, Any] = {
        'base_model_path': base_model_path,
        'dataset': [
            {
                'input': item['input'],
                'output': item['output'],
                'source': item.get('source'),
            }
            for item in dataset
        ],
        'config': config_payload,
    }

    if fine_tune_config.get('output_dir'):
        payload['output_dir'] = fine_tune_config['output_dir']

    endpoint = server_url.rstrip('/') + '/v1/fine-tunes'
    headers = fine_tune_config.get('headers')
    timeout = float(fine_tune_config.get('timeout', 900))
    response = http_request(
        'POST',
        endpoint,
        json=payload,
        headers=headers,
        timeout=(HTTP_CONNECT_TIMEOUT, max(timeout, HTTP_DEFAULT_TIMEOUT)),
        logger=app.logger,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Fine-tune request failed: {response.status_code} {response.text[:200]}")
    return response.json()


def _build_dataset_from_problems(
    payload: Dict[str, Any],
    logs: List[str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    evaluation = payload.get('evaluation') or {}
    items = evaluation.get('items') or []
    if not isinstance(items, list) or not items:
        raise ValueError('evaluation.items должен быть непустым списком')

    generation_opts = payload.get('generation') or {}
    search_opts = payload.get('search') or {}

    score_threshold = payload.get('score_threshold', evaluation.get('quality_gate'))
    try:
        threshold_value = float(score_threshold) if score_threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None

    max_questions = int(generation_opts.get('max_questions') or payload.get('max_questions') or len(items))
    target_pairs = int(payload.get('target_pairs') or generation_opts.get('target_pairs') or generation_opts.get('total_pairs') or 20)
    pairs_per_snippet = max(1, int(generation_opts.get('pairs_per_snippet') or 1))
    include_reference_pair = bool(generation_opts.get('include_reference_pair', True))
    min_paragraph_chars = max(30, int(generation_opts.get('min_paragraph_chars', 120)))
    max_paragraph_chars = max(min_paragraph_chars, int(generation_opts.get('max_paragraph_chars', 700)))
    llm_temperature = float(generation_opts.get('temperature', 0.2))
    llm_max_tokens = int(generation_opts.get('max_tokens', 512))

    scored_items: List[Tuple[Optional[float], int, Dict[str, Any]]] = []
    for idx, raw in enumerate(items):
        if not isinstance(raw, dict):
            continue
        score = _resolve_problem_score(raw)
        if threshold_value is None or (score is None) or (score <= threshold_value):
            scored_items.append((score, idx, raw))

    if not scored_items:
        raise ValueError('Не найдено вопросов, требующих внимания — уточните порог или список')

    scored_items.sort(key=lambda entry: (entry[0] if entry[0] is not None else -1.0, entry[1]))
    selected_items = scored_items[:max(1, max_questions)]

    dataset: List[Dict[str, str]] = []
    dataset_dedup: set[Tuple[str, str]] = set()
    processed = 0

    for rank, (score, _, problem) in enumerate(selected_items, start=1):
        question = str(problem.get('question') or '').strip()
        reference = str(problem.get('referenceAnswer') or problem.get('reference') or '').strip()
        logs.append(f"[problem {rank}] score={score!r} question='{question[:80]}'")

        if include_reference_pair and question and reference:
            key = (question, reference)
            if key not in dataset_dedup:
                dataset.append({'input': question, 'output': reference, 'source': 'evaluation'})
                dataset_dedup.add(key)
                logs.append(f"  добавлена пара из эталонного ответа")
                if len(dataset) >= target_pairs:
                    break

        search_parts = [question]
        if reference:
            search_parts.append(reference)
        search_query = ' '.join(part for part in search_parts if part).strip()
        if not search_query:
            logs.append('  пропущено: пустой поисковый запрос')
            continue

        search_payload: Dict[str, Any] = {
            'query': search_query,
            'top_k': search_opts.get('top_k', 5),
            'deep_search': bool(search_opts.get('deep_search', True)),
            'llm_snippets': False,
            'max_candidates': search_opts.get('max_candidates', 20),
            'max_snippets': search_opts.get('max_snippets', 3),
            'chunk_chars': search_opts.get('chunk_chars', 5000),
            'max_chunks': search_opts.get('max_chunks', 25),
        }
        if 'sources' in search_opts:
            search_payload['sources'] = search_opts['sources']
        if 'collection_ids' in search_opts:
            search_payload['collection_ids'] = search_opts['collection_ids']
        if 'material_types' in search_opts:
            search_payload['material_types'] = search_opts['material_types']

        progress_lines: List[str] = []
        result = _ai_search_core(search_payload, progress_lines.append)
        for line in progress_lines:
            logs.append(f"  [search] {line}")

        items_found = result.get('items') or []
        logs.append(f"  найдено кандидатов: {len(items_found)}")

        for cand_index, candidate in enumerate(items_found, start=1):
            file_id = candidate.get('file_id')
            if not file_id:
                continue
            file_obj = File.query.get(int(file_id))
            if not file_obj:
                continue

            snippets: List[str] = []
            for snippet in candidate.get('snippets') or []:
                snippet = (snippet or '').strip()
                if len(snippet) >= min_paragraph_chars:
                    snippets.append(snippet)

            if not snippets:
                text_excerpt = _read_cached_excerpt_for_file(file_obj)
                snippets = [
                    chunk
                    for chunk in _split_text_for_dataset(
                        text_excerpt,
                        min_chars=min_paragraph_chars,
                        max_chars=max_paragraph_chars,
                        max_segments=search_opts.get('max_segments', 4),
                    )
                ]

            if not snippets:
                continue

            for sn_index, snippet in enumerate(snippets, start=1):
                if len(dataset) >= target_pairs:
                    break
                pairs = _generate_pairs_from_snippet(
                    snippet,
                    desired_pairs=pairs_per_snippet,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                    problem_question=question,
                )
                if not pairs:
                    continue
                for pair in pairs:
                    q_text = pair['question'].strip()
                    a_text = pair['answer'].strip()
                    if not q_text or not a_text:
                        continue
                    key = (q_text, a_text)
                    if key in dataset_dedup:
                        continue
                    dataset.append({
                        'input': q_text,
                        'output': a_text,
                        'source': f"agregator:{file_obj.rel_path or file_obj.filename}#s{sn_index}"
                    })
                    dataset_dedup.add(key)
                    logs.append(
                        f"  добавлена пара из файла '{file_obj.rel_path or file_obj.filename}' (сниппет {sn_index})"
                    )
                    if len(dataset) >= target_pairs:
                        break
            if len(dataset) >= target_pairs:
                break

        processed += 1
        if len(dataset) >= target_pairs:
            break

    meta = {
        'selected_problems': len(selected_items),
        'processed_problems': processed,
        'target_pairs': target_pairs,
    }
    return dataset, meta


def _add_pipeline_cors_headers(response: Response) -> Response:
    origin = request.headers.get('Origin')
    if origin:
        response.headers['Access-Control-Allow-Origin'] = origin
        vary = response.headers.get('Vary')
        response.headers['Vary'] = f"{vary}, Origin" if vary else 'Origin'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
    else:
        response.headers.setdefault('Access-Control-Allow-Origin', '*')
    response.headers.setdefault('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.setdefault('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response


def _pipeline_cors_preflight() -> Response:
    response = make_response('', 204)
    return _add_pipeline_cors_headers(response)


@app.route('/api/training/problem-pipeline', methods=['POST', 'OPTIONS'])
def api_training_problem_pipeline():
    if request.method == 'OPTIONS':
        return _pipeline_cors_preflight()
    allowed, failure_response = _check_pipeline_access()
    if not allowed:
        return _add_pipeline_cors_headers(failure_response)
    payload = request.get_json(silent=True) or {}
    logs: List[str] = []

    try:
        dataset, meta = _build_dataset_from_problems(payload, logs)
    except ValueError as exc:
        logs.append(f"Ошибка: {exc}")
        resp = jsonify({'ok': False, 'error': str(exc), 'logs': logs})
        resp.status_code = 400
        return _add_pipeline_cors_headers(resp)
    except Exception as exc:
        app.logger.exception('problem pipeline failed')
        logs.append(f'Неожиданная ошибка: {exc}')
        resp = jsonify({'ok': False, 'error': 'Не удалось построить датасет', 'logs': logs})
        resp.status_code = 500
        return _add_pipeline_cors_headers(resp)

    if not dataset:
        logs.append('Датасет пуст — нечего отправлять на дообучение')
        resp = jsonify({'ok': False, 'error': 'Датасет пуст', 'logs': logs, 'meta': meta})
        resp.status_code = 400
        return _add_pipeline_cors_headers(resp)

    dry_run = bool(payload.get('dry_run'))
    include_dataset = bool(payload.get('include_dataset'))
    job_info: Optional[Dict[str, Any]] = None

    if not dry_run and payload.get('fine_tune'):
        try:
            job_info = _launch_fine_tune_job(dataset, payload['fine_tune'])
            logs.append(f"Запущено дообучение: задача {job_info.get('id')}")
        except Exception as exc:
            app.logger.exception('fine-tune launch failed')
            logs.append(f"Ошибка запуска дообучения: {exc}")
            resp = jsonify({
                'ok': False,
                'error': str(exc),
                'logs': logs,
                'meta': meta,
                'dataset_size': len(dataset),
            })
            resp.status_code = 502
            return _add_pipeline_cors_headers(resp)

    preview = [
        {
            'input': item['input'][:160],
            'output': item['output'][:160],
            'source': item.get('source'),
        }
        for item in dataset[:3]
    ]

    response: Dict[str, Any] = {
        'ok': True,
        'dataset_size': len(dataset),
        'meta': meta,
        'logs': logs,
        'preview': preview,
    }
    if job_info is not None:
        response['fine_tune_job'] = job_info
    if include_dataset:
        response['dataset'] = dataset
    elif dry_run:
        response['dataset_preview_full'] = dataset[: min(len(dataset), 10)]
    return _add_pipeline_cors_headers(jsonify(response))


@app.route('/api/ai-search', methods=['POST'])
def api_ai_search():
    data = request.get_json(silent=True) or {}
    user = _load_current_user()
    query_preview = str(data.get('query') or '').strip()[:200]
    detail_obj = {
        'query_preview': query_preview,
        'top_k': data.get('top_k'),
        'deep_search': data.get('deep_search'),
        'sources': data.get('sources'),
    }
    try:
        detail_payload = json.dumps(detail_obj, ensure_ascii=False)
    except Exception:
        detail_payload = str(detail_obj)
    try:
        _log_user_action(user, 'ai_search', 'search', None, detail=detail_payload[:2000])
    except Exception:
        pass
    try:
        app.logger.info(
            "[user-action] user=%s action=ai_search query_preview=%s top_k=%s deep_search=%s",
            getattr(user, 'username', None) or 'anonymous',
            query_preview,
            detail_obj.get('top_k'),
            detail_obj.get('deep_search')
        )
    except Exception:
        pass
    try:
        result = _ai_search_core(data)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        logger.exception("AI search failed", exc_info=True)
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, **result})
@app.route('/api/ai-search/feedback', methods=['POST'])
def api_ai_search_feedback():
    user = _load_current_user()
    if not user:
        return jsonify({'ok': False, 'error': 'Не авторизовано'}), 401
    data = request.get_json(silent=True) or {}
    query_hash = str(data.get('query_hash') or '').strip()
    action = str(data.get('action') or '').strip().lower()
    if not query_hash or action not in {'click', 'relevant', 'irrelevant', 'ignored'}:
        return jsonify({'ok': False, 'error': 'Некорректные параметры'}), 400
    keyword = (data.get('keyword') or '').strip() or None
    detail = data.get('detail')
    file_id = data.get('file_id')
    try:
        file_id_int = int(file_id) if file_id is not None else None
    except Exception:
        file_id_int = None
    score = data.get('score')
    try:
        score_val = float(score) if score is not None else None
    except Exception:
        score_val = None
    entry = AiSearchKeywordFeedback(
        user_id=user.id,
        file_id=file_id_int,
        query_hash=query_hash,
        keyword=keyword,
        action=action,
        score=score_val,
        detail=json.dumps(detail, ensure_ascii=False) if isinstance(detail, (dict, list)) else (detail or None),
    )
    try:
        db.session.add(entry)
        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        app.logger.warning(f"[ai-feedback] failed to store feedback: {exc}")
        return jsonify({'ok': False, 'error': 'Не удалось сохранить отклик'}), 500
    return jsonify({'ok': True})


def _initialize_background_jobs():
    _start_cache_cleanup_scheduler()


if hasattr(app, "before_serving"):
    app.before_serving(_initialize_background_jobs)


@app.before_request
def _ensure_background_jobs_started():
    if not _CLEANUP_THREAD_STARTED:
        _initialize_background_jobs()


if __name__ == "__main__":
    setup_app()
    with app.app_context():
        ensure_collections_schema()
        ensure_llm_schema()
        ensure_default_admin()
    port = int(os.environ.get("AGREGATOR_PORT") or os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False, threaded=True)

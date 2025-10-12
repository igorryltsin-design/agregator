"""Application configuration helpers.

This module encapsulates environment-driven configuration for Agregator so
settings can be loaded via a structured `AppConfig` dataclass and injected into
Flask during application factory bootstrap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def _getenv_bool(name: str, default: bool = False) -> bool:
    """Return a boolean flag from environment variables."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv(base_dir: Path) -> None:
    env_path = base_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _sanitize_subdir(name: str) -> str:
    return name.strip().strip("/\\")


@dataclass(frozen=True)
class AppConfig:
    """Strongly typed application configuration container."""

    base_dir: Path
    login_backgrounds_dir: Path
    scan_root: Path
    extract_text: bool
    ocr_langs: str
    pdf_ocr_pages: int
    always_ocr_first_page_dissertation: bool
    default_use_llm: bool
    default_prune: bool
    collections_in_separate_dirs: bool
    collection_type_subdirs: bool
    lmstudio_api_base: str
    lmstudio_model: str
    lmstudio_api_key: str
    lm_default_provider: str
    transcribe_enabled: bool
    transcribe_backend: str
    transcribe_model_path: str
    transcribe_language: str
    images_vision_enabled: bool
    keywords_to_tags_enabled: bool
    type_detect_flow: str
    type_llm_override: bool
    rename_patterns: Dict[str, str]
    ai_rerank_llm: bool
    import_subdir: str
    move_on_rename: bool
    type_dirs: Dict[str, str]
    material_type_profiles: List[Dict[str, Any]]
    default_prompts: Dict[str, str]
    prompts: Dict[str, str]
    summarize_audio: bool
    audio_keywords_llm: bool
    fw_cache_dir: Path
    settings_store_path: Path
    search_facet_tag_keys: Optional[List[str]]
    graph_facet_tag_keys: Optional[List[str]]
    search_facet_include_types: bool
    flask_secret_key: str
    max_content_length: int
    http_timeout: float
    http_connect_timeout: float
    http_retries: int
    http_backoff_factor: float
    llm_cache_enabled: bool
    llm_cache_ttl_seconds: int
    llm_cache_max_items: int
    llm_cache_read_only: bool
    search_cache_enabled: bool
    search_cache_ttl_seconds: int
    search_cache_max_items: int
    lm_max_input_chars: int
    lm_max_output_tokens: int
    azure_openai_api_version: str
    log_level: str
    sentry_dsn: str
    sentry_environment: str
    default_admin_user: str
    default_admin_password: str
    legacy_access_code: str
    catalogue_db_path: Path = field(init=False)
    logs_dir: Path = field(init=False)
    log_file_path: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalogue_db_path", self.base_dir / "catalogue.db")
        logs_dir = self.base_dir / "logs"
        object.__setattr__(self, "logs_dir", logs_dir)
        object.__setattr__(self, "log_file_path", logs_dir / "agregator.log")

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "AppConfig":
        base_dir = base_dir or Path(__file__).resolve().parent.parent
        _load_dotenv(base_dir)

        scan_root = Path(os.getenv("SCAN_ROOT", str(base_dir / "sample_library")))
        rename_patterns: Dict[str, str] = {
            "dissertation": "{abbr}.{degree}.{title}.{author_last}",
            "dissertation_abstract": "{abbr}.{degree}.{title}.{author_last}",
            "article": "СТ.{title}.{author_last}",
            "textbook": "УЧ.{title}.{author_last}",
            "monograph": "МОНО.{title}.{author_last}",
            "image": "ИЗО.{title}",
            "audio": "АУД.{title}",
            "default": "{abbr}.{title}.{author_last}",
        }
        type_dirs: Dict[str, str] = {
            "dissertation": "dissertations",
            "dissertation_abstract": "dissertation_abstract",
            "article": "articles",
            "textbook": "textbooks",
            "monograph": "monographs",
            "report": "reports",
            "patent": "patents",
            "presentation": "presentations",
            "proceedings": "proceedings",
            "standard": "standards",
            "note": "notes",
            "document": "documents",
            "audio": "audio",
            "image": "images",
            "other": "other",
        }
        material_type_profiles: List[Dict[str, Any]] = [
            {
                "key": "image",
                "label": "Изображение",
                "description": "Фотографии, сканы и другие графические файлы.",
                "extensions": ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
                "priority": 120,
                "extension_weight": 3.0,
                "filename_weight": 1.5,
                "text_weight": 1.0,
                "llm_hint": "Изображение или фотография. Требуется визуальное описание и ключевые слова.",
            },
            {
                "key": "audio",
                "label": "Аудио",
                "description": "Звуковые записи, требующие транскрибации.",
                "extensions": ["mp3", "wav", "m4a", "flac", "ogg"],
                "priority": 110,
                "extension_weight": 3.0,
                "filename_weight": 1.5,
                "text_weight": 1.0,
                "llm_hint": "Аудиозапись. Сначала требуется транскрибация, затем анализ содержания.",
            },
            {
                "key": "dissertation_abstract",
                "label": "Автореферат",
                "description": "Автореферат диссертации, краткое изложение основных результатов.",
                "text_keywords": ["автореферат", "автореферат диссертац", "autoreferat", "автoref"],
                "filename_keywords": ["автореферат", "autoreferat", "автoref"],
                "priority": 95,
                "threshold": 1.0,
                "llm_hint": "Автореферат — краткое изложение диссертации с акцентом на цели, новизну и выводы.",
            },
            {
                "key": "dissertation",
                "label": "Диссертация",
                "description": "Полный текст диссертации на соискание учёной степени.",
                "text_keywords": ["диссертац", "на соискание степени", "диссертация"],
                "filename_keywords": ["диссер", "dissert", "thesis"],
                "priority": 90,
                "threshold": 1.0,
                "llm_hint": "Полная диссертация с подробным описанием исследования, структуры и приложений.",
            },
            {
                "key": "patent",
                "label": "Патент",
                "description": "Патентные документы с описанием изобретения и классификацией.",
                "text_keywords": ["патент", "patent", "mpk", "ipc"],
                "filename_keywords": ["патент", "patent", "ru", "wo", "ep"],
                "priority": 87,
                "threshold": 1.0,
                "llm_hint": "Патентное описание с формулой изобретения и ссылками на классификаторы.",
            },
            {
                "key": "standard",
                "label": "Стандарт",
                "description": "ГОСТ, ISO, IEC, СанПиН и другие нормативные документы.",
                "text_keywords": ["гост", "gost", "iso", "iec", "стб", "санпин", " сп ", "ту ", " сто "],
                "filename_keywords": ["гост", "gost", "iso", "iec", "санпин", "сто_", "сто ", "ту_", "tu_"],
                "priority": 88,
                "threshold": 1.0,
                "llm_hint": "Нормативный документ. Важно выявить номер стандарта, область применения и актуальность.",
            },
            {
                "key": "proceedings",
                "label": "Материалы конференции",
                "description": "Сборники трудов, тезисов и материалов конференций.",
                "text_keywords": ["материалы конференции", "сборник трудов", "proceedings", "conference", "symposium", "workshop", "тезисы"],
                "filename_keywords": ["proceedings", "conf", "symposium", "workshop"],
                "priority": 75,
                "threshold": 1.0,
                "llm_hint": "Материалы научной конференции или сборник тезисов участников.",
            },
            {
                "key": "journal",
                "label": "Журнал",
                "description": "Целый выпуск научного журнала с оглавлением.",
                "text_keywords": ["журнал", "вестник", "issn", "выпуск", "№"],
                "filename_keywords": ["журнал", "journal", "magazine", "issue", "vestnik", "выпуск"],
                "priority": 85,
                "threshold": 1.5,
                "special": {"journal_toc_required": True, "min_toc_entries": 5},
                "llm_hint": "Отдельный выпуск журнала с несколькими статьями и оглавлением.",
            },
            {
                "key": "article",
                "label": "Статья",
                "description": "Научная статья или отдельная публикация.",
                "text_keywords": ["статья", "журнал", "doi", "удк", "материалы конференц", "тезисы"],
                "priority": 60,
                "threshold": 1.0,
                "llm_hint": "Отдельная научная статья; выделяй аннотацию, авторов, ключевые слова.",
            },
            {
                "key": "textbook",
                "label": "Учебник",
                "description": "Учебные пособия и учебники для студентов.",
                "text_keywords": ["учебник", "учебное пособ", "пособие", "для студентов"],
                "priority": 55,
                "threshold": 1.0,
                "llm_hint": "Учебное пособие. Отмечай целевую аудиторию и структуру курса.",
            },
            {
                "key": "monograph",
                "label": "Монография",
                "description": "Научная монография или фундаментальное исследование.",
                "text_keywords": ["монография", "monograph"],
                "filename_keywords": ["монограф", "monograph"],
                "priority": 58,
                "threshold": 1.0,
                "llm_hint": "Научная монография — глубокое авторское исследование.",
            },
            {
                "key": "report",
                "label": "Отчёт",
                "description": "Отчётные документы, ТЗ, пояснительные записки.",
                "text_keywords": ["отчет", "отчёт", "техническое задание", "пояснительная записка", "technical specification"],
                "filename_keywords": ["отчет", "отчёт", "tz_", "тз_"],
                "priority": 65,
                "threshold": 1.0,
                "llm_hint": "Отчёт или техническая документация. Важно фиксировать заказчика, исполнителя и период.",
            },
            {
                "key": "presentation",
                "label": "Презентация",
                "description": "Слайды презентаций и докладов.",
                "text_keywords": ["презентация", "slides", "powerpoint", "слайды", "deck"],
                "filename_keywords": ["презентац", "slides", "ppt", "pptx", "keynote", "deck"],
                "extensions": ["ppt", "pptx", "key", "odp"],
                "priority": 62,
                "threshold": 1.0,
                "llm_hint": "Презентация в слайдах. Структура: слайды, разделы, основные тезисы.",
            },
            {
                "key": "note",
                "label": "Заметка",
                "description": "Короткие текстовые заметки и инструкции.",
                "text_keywords": ["заметка", "note"],
                "extensions": ["md", "txt", "rst"],
                "priority": 30,
                "threshold": 1.0,
                "llm_hint": "Краткая заметка или текстовый файл с минимальной структурой.",
            },
            {
                "key": "document",
                "label": "Документ",
                "description": "Общий тип по умолчанию, если ничего не подошло.",
                "priority": 0,
                "threshold": 0.0,
                "llm_hint": "Базовый тип документа, когда классификация не определена.",
            },
        ]
        default_prompts: Dict[str, str] = {
            "metadata_system": (
                "Ты помощник по каталогизации научных материалов. "
                "Определи material_type из списка: {{TYPE_LIST}}. "
                "Используй подсказки по типам:\n{{TYPE_HINTS}}\n"
                "Верни ТОЛЬКО валидный JSON без пояснений. "
                "Ключи: material_type, title, author, year, advisor, keywords (array), novelty (string), "
                "literature (array), organizations (array), classification (array). Если данных нет — пустые строки/массивы."
            ),
            "summarize_audio_system": (
                "Ты помощник. Суммаризируй стенограмму аудио в 3–6 предложениях на русском, "
                "выделив тему, основные тезисы и вывод."
            ),
            "keywords_system": (
                "Ты извлекаешь ключевые слова из стенограммы аудио. Верни только JSON-массив строк на русском: "
                "[\"ключ1\", \"ключ2\", ...]. Без пояснений, не более 12 слов/фраз."
            ),
            "ai_search_keywords_system": (
                "Ты помощник ИИ-поиска. По заданному запросу подбери 3-8 релевантных тегов (фраз) на русском "
                "или английском. Теги должны отражать суть запроса, быть краткими (1-3 слова) и без служебных слов. "
                "Если в запросе есть выражения в кавычках, включи их без изменений (без кавычек). Верни строго JSON-массив строк."
            ),
            "vision_system": (
                "Ты помощник по анализу изображений. Опиши изображение 2–4 предложениями на русском и верни 5–12 ключевых слов. "
                "Верни строго JSON: {\"description\":\"...\",\"keywords\":[\"...\"]}."
            ),
        }

        prompts = dict(default_prompts)

        return cls(
            base_dir=base_dir,
            login_backgrounds_dir=(base_dir / "static" / "login-backgrounds").resolve(),
            scan_root=scan_root,
            extract_text=_getenv_bool("EXTRACT_TEXT", True),
            ocr_langs=os.getenv("OCR_LANGS", "rus+eng"),
            pdf_ocr_pages=int(os.getenv("PDF_OCR_PAGES", "5")),
            always_ocr_first_page_dissertation=_getenv_bool("OCR_DISS_FIRST_PAGE", True),
            default_use_llm=_getenv_bool("DEFAULT_USE_LLM", False),
            default_prune=_getenv_bool("DEFAULT_PRUNE", True),
            collections_in_separate_dirs=_getenv_bool("COLLECTIONS_IN_SEPARATE_DIRS", False),
            collection_type_subdirs=_getenv_bool("COLLECTION_TYPE_SUBDIRS", False),
            lmstudio_api_base=os.getenv("LMSTUDIO_API_BASE", "http://localhost:1234/v1"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL", "google/gemma-3n-e4b"),
            lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", ""),
            lm_default_provider=(os.getenv("LM_PROVIDER") or "openai").strip().lower() or "openai",
            transcribe_enabled=_getenv_bool("TRANSCRIBE_ENABLED", True),
            transcribe_backend=os.getenv("TRANSCRIBE_BACKEND", "faster-whisper"),
            transcribe_model_path=os.getenv("TRANSCRIBE_MODEL_PATH", os.getenv("FASTER_WHISPER_DEFAULT_MODEL", "small")),
            transcribe_language=os.getenv("TRANSCRIBE_LANGUAGE", "ru"),
            images_vision_enabled=_getenv_bool("IMAGES_VISION_ENABLED", False),
            keywords_to_tags_enabled=_getenv_bool("KEYWORDS_TO_TAGS_ENABLED", True),
            type_detect_flow=os.getenv("TYPE_DETECT_FLOW", "extension,filename,heuristics,llm"),
            type_llm_override=_getenv_bool("TYPE_LLM_OVERRIDE", True),
            rename_patterns=rename_patterns,
            ai_rerank_llm=_getenv_bool("AI_RERANK_LLM", False),
            import_subdir=_sanitize_subdir(os.getenv("IMPORT_SUBDIR", "import")),
            move_on_rename=_getenv_bool("MOVE_ON_RENAME", True),
            type_dirs=type_dirs,
            material_type_profiles=material_type_profiles,
            default_prompts=default_prompts,
            prompts=prompts,
            summarize_audio=_getenv_bool("SUMMARIZE_AUDIO", False),
            audio_keywords_llm=_getenv_bool("AUDIO_KEYWORDS_LLM", True),
            fw_cache_dir=Path(os.getenv(
                "FASTER_WHISPER_CACHE_DIR",
                str((base_dir / "models" / "faster-whisper").resolve()),
            )),
            settings_store_path=base_dir / "runtime_settings.json",
            search_facet_tag_keys=None,
            graph_facet_tag_keys=None,
            search_facet_include_types=True,
            flask_secret_key=os.getenv("FLASK_SECRET_KEY", "dev-secret"),
            max_content_length=int(os.getenv("MAX_CONTENT_LENGTH", str(50 * 1024 * 1024))),
            http_timeout=float(os.getenv("AG_HTTP_TIMEOUT", "120") or 120),
            http_connect_timeout=float(os.getenv("AG_HTTP_CONNECT_TIMEOUT", "10") or 10),
            http_retries=int(os.getenv("AG_HTTP_RETRIES", "3") or 3),
            http_backoff_factor=float(os.getenv("AG_HTTP_BACKOFF", "0.5") or 0.5),
            llm_cache_enabled=_getenv_bool("LLM_CACHE_ENABLED", True),
            llm_cache_ttl_seconds=int(os.getenv("LLM_CACHE_TTL_SECONDS", "600") or 600),
            llm_cache_max_items=int(os.getenv("LLM_CACHE_MAX_ITEMS", "256") or 256),
            llm_cache_read_only=_getenv_bool("LLM_CACHE_ONLY_MODE", False),
            search_cache_enabled=_getenv_bool("SEARCH_CACHE_ENABLED", True),
            search_cache_ttl_seconds=int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "90") or 90),
            search_cache_max_items=int(os.getenv("SEARCH_CACHE_MAX_ITEMS", "64") or 64),
            lm_max_input_chars=max(500, int(os.getenv("LM_MAX_INPUT_CHARS", "4000") or 4000)),
            lm_max_output_tokens=max(16, int(os.getenv("LM_MAX_OUTPUT_TOKENS", "256") or 256)),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            sentry_dsn=os.getenv("SENTRY_DSN", ""),
            sentry_environment=os.getenv("SENTRY_ENVIRONMENT", "local"),
            default_admin_user=(os.getenv("DEFAULT_ADMIN_USER") or "admin").strip() or "admin",
            default_admin_password=(os.getenv("DEFAULT_ADMIN_PASSWORD") or "").strip(),
            legacy_access_code=(os.getenv("ACCESS_CODE") or "").strip(),
        )

    def to_flask_config(self) -> Dict[str, object]:
        """Return a dict with settings that should live inside `Flask.config`."""
        return {
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{self.catalogue_db_path}",
            "SQLALCHEMY_TRACK_MODIFICATIONS": False,
            "JSON_AS_ASCII": False,
            "UPLOAD_FOLDER": str(self.scan_root),
            "IMPORT_SUBDIR": self.import_subdir,
            "MOVE_ON_RENAME": self.move_on_rename,
            "COLLECTIONS_IN_SEPARATE_DIRS": self.collections_in_separate_dirs,
            "COLLECTION_TYPE_SUBDIRS": self.collection_type_subdirs,
            "TYPE_DIRS": self.type_dirs,
            "SEARCH_FACET_TAG_KEYS": self.search_facet_tag_keys,
            "GRAPH_FACET_TAG_KEYS": self.graph_facet_tag_keys,
            "SEARCH_FACET_INCLUDE_TYPES": self.search_facet_include_types,
            "MAX_CONTENT_LENGTH": self.max_content_length,
            "HTTP_DEFAULT_TIMEOUT": self.http_timeout,
            "HTTP_CONNECT_TIMEOUT": self.http_connect_timeout,
            "HTTP_RETRIES": self.http_retries,
            "HTTP_BACKOFF_FACTOR": self.http_backoff_factor,
            "LLM_CACHE_ENABLED": self.llm_cache_enabled,
            "LLM_CACHE_TTL_SECONDS": self.llm_cache_ttl_seconds,
            "LLM_CACHE_MAX_ITEMS": self.llm_cache_max_items,
            "LLM_CACHE_ONLY_MODE": self.llm_cache_read_only,
            "SEARCH_CACHE_ENABLED": self.search_cache_enabled,
            "SEARCH_CACHE_TTL_SECONDS": self.search_cache_ttl_seconds,
            "SEARCH_CACHE_MAX_ITEMS": self.search_cache_max_items,
            "LOG_LEVEL": self.log_level,
            "SENTRY_DSN": self.sentry_dsn,
            "SENTRY_ENVIRONMENT": self.sentry_environment,
        }


def load_app_config(base_dir: Optional[Path] = None) -> AppConfig:
    """Convenience shortcut mirroring legacy callers."""
    return AppConfig.load(base_dir=base_dir)

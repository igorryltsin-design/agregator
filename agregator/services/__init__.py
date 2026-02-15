"""Service layer helpers for Agregator."""

from .logging import (
    configure_logging,
    get_rotating_log_handler,
    list_system_log_files,
    resolve_log_name,
    tail_log_file,
)
from .browser import (
    BrowserManager,
    BrowserSettings,
    configure_browser,
    get_browser_manager,
)
from .http import HttpSettings, configure_http, get_http_session, get_http_settings, http_request
from .llm_cache import (
    CachedLLMResponse,
    configure_llm_cache,
    llm_cache_get,
    llm_cache_set,
    llm_cache_clear,
    llm_cache_stats,
)
from .llm_pool import (
    LlmPoolRejected,
    LlmPoolTimeout,
    configure_llm_pool,
    get_llm_pool,
)
from .facets import FacetQueryParams, FacetService
from .search_cache import (
    configure_search_cache,
    search_cache_get,
    search_cache_set,
    search_cache_clear,
    search_cache_stats,
)
from .search import SearchService
from .tasks import TaskQueue, get_task_queue
from .nlp import ru_tokens, lemma, lemmas, expand_synonyms
from .snippet import collect_snippets, split_text_chunks, iter_document_chunks

__all__ = [
    "configure_logging",
    "get_rotating_log_handler",
    "list_system_log_files",
    "resolve_log_name",
    "tail_log_file",
    "configure_browser",
    "get_browser_manager",
    "BrowserSettings",
    "BrowserManager",
    "configure_http",
    "get_http_session",
    "get_http_settings",
    "http_request",
    "HttpSettings",
    "configure_llm_cache",
    "llm_cache_get",
    "llm_cache_set",
    "llm_cache_clear",
    "llm_cache_stats",
    "CachedLLMResponse",
    "LlmPoolRejected",
    "LlmPoolTimeout",
    "configure_llm_pool",
    "get_llm_pool",
    "configure_search_cache",
    "search_cache_get",
    "search_cache_set",
    "search_cache_clear",
    "search_cache_stats",
    "SearchService",
    "FacetService",
    "FacetQueryParams",
    "TaskQueue",
    "get_task_queue",
    "ru_tokens",
    "lemma",
    "lemmas",
    "expand_synonyms",
    "collect_snippets",
    "split_text_chunks",
    "iter_document_chunks",
]

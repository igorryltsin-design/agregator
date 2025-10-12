"""TTL-кэш для ответов LLM."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class CachedLLMResponse:
    """Упрощённый объект ответа, имитирующий интерфейс requests.Response."""

    def __init__(self, status_code: int, data: Any, text: str, headers: Optional[dict[str, str]] = None) -> None:
        self.status_code = status_code
        self._data = data
        self.text = text
        self.headers = headers or {}

    def json(self) -> Any:
        return self._data


class LlmCache:
    def __init__(self, max_items: int = 256, ttl_seconds: int = 600) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, tuple[float, CachedLLMResponse]]" = OrderedDict()
        self.enabled = True

    def configure(self, *, enabled: bool, max_items: int, ttl_seconds: int) -> None:
        with self._lock:
            self.enabled = enabled
            self.max_items = max(1, max_items)
            self.ttl_seconds = max(1, ttl_seconds)
            if not enabled:
                self._data.clear()

    def _purge_expired(self) -> None:
        now = time.time()
        while self._data:
            key, (ts, _) = next(iter(self._data.items()))
            if now - ts <= self.ttl_seconds:
                break
            self._data.popitem(last=False)

    def get(self, key: str) -> Optional[CachedLLMResponse]:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            ts, response = entry
            if time.time() - ts > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            # Переносим в конец, чтобы соблюсти семантику LRU
            self._data.move_to_end(key)
            return response

    def set(self, key: str, response: CachedLLMResponse) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = (time.time(), response)
            self._data.move_to_end(key)
            if len(self._data) > self.max_items:
                self._data.popitem(last=False)
            self._purge_expired()


_default_cache = LlmCache()


def configure_llm_cache(*, enabled: bool, max_items: int, ttl_seconds: int) -> None:
    _default_cache.configure(enabled=enabled, max_items=max_items, ttl_seconds=ttl_seconds)


def llm_cache_get(key: str) -> Optional[CachedLLMResponse]:
    return _default_cache.get(key)


def llm_cache_set(key: str, response: CachedLLMResponse) -> None:
    _default_cache.set(key, response)

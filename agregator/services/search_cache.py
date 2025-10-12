"""Кэш результатов поиска для Agregator."""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class CachedSearchResult:
    def __init__(self, payload: Any, created_at: float | None = None) -> None:
        self.payload = payload
        self.created_at = created_at or time.time()


class SearchCache:
    def __init__(self, max_items: int = 128, ttl_seconds: int = 120) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self.enabled = True
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, CachedSearchResult]" = OrderedDict()

    def configure(self, *, enabled: bool, max_items: int, ttl_seconds: int) -> None:
        with self._lock:
            self.enabled = enabled
            self.max_items = max(1, max_items)
            self.ttl_seconds = max(1, ttl_seconds)
            if not enabled:
                self._data.clear()

    def get(self, key: str) -> Optional[CachedSearchResult]:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            if time.time() - entry.created_at > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return entry

    def set(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        snap = CachedSearchResult(payload)
        with self._lock:
            self._data[key] = snap
            self._data.move_to_end(key)
            if len(self._data) > self.max_items:
                self._data.popitem(last=False)


_default_cache = SearchCache()


def configure_search_cache(*, enabled: bool, max_items: int, ttl_seconds: int) -> None:
    _default_cache.configure(enabled=enabled, max_items=max_items, ttl_seconds=ttl_seconds)


def search_cache_get(key: str) -> Optional[Any]:
    entry = _default_cache.get(key)
    if entry is None:
        return None
    return entry.payload


def search_cache_set(key: str, payload: Any) -> None:
    _default_cache.set(key, payload)

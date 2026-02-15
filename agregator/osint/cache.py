"""TTL cache for OSINT intermediate artifacts."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(slots=True)
class CachedValue:
    payload: Any
    created_at: float


class OsintCache:
    """Simple LRU cache with TTL semantics."""

    def __init__(self, *, max_items: int = 64, ttl_seconds: int = 900) -> None:
        self.max_items = max(1, max_items)
        self.ttl_seconds = max(1, ttl_seconds)
        self.enabled = True
        self._lock = threading.Lock()
        self._data: "OrderedDict[str, CachedValue]" = OrderedDict()

    def configure(self, *, enabled: bool, max_items: int, ttl_seconds: int) -> None:
        with self._lock:
            self.enabled = enabled
            self.max_items = max(1, max_items)
            self.ttl_seconds = max(1, ttl_seconds)
            if not enabled:
                self._data.clear()

    def get(self, key: str) -> Optional[CachedValue]:
        if not self.enabled:
            return None
        with self._lock:
            snap = self._data.get(key)
            if snap is None:
                return None
            if time.time() - snap.created_at > self.ttl_seconds:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return snap

    def set(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = CachedValue(payload=payload, created_at=time.time())
            self._data.move_to_end(key)
            if len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "items": len(self._data),
                "max_items": self.max_items,
                "ttl_seconds": self.ttl_seconds,
            }


_DEFAULT_CACHE = OsintCache()


def configure_osint_cache(*, enabled: bool, max_items: int, ttl_seconds: int) -> None:
    _DEFAULT_CACHE.configure(enabled=enabled, max_items=max_items, ttl_seconds=ttl_seconds)


def osint_cache_get(key: str) -> Any | None:
    snap = _DEFAULT_CACHE.get(key)
    if snap is None:
        return None
    return snap.payload


def osint_cache_set(key: str, payload: Any) -> None:
    _DEFAULT_CACHE.set(key, payload)


def osint_cache_clear() -> None:
    _DEFAULT_CACHE.clear()


def osint_cache_stats() -> dict[str, Any]:
    return _DEFAULT_CACHE.stats()


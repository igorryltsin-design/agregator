"""Thread-safe TTL cache with LRU eviction.

This is the single source of truth for ``TimedCache``.  Previously the class
was defined **twice** inside ``app.py`` (lines ~606 and ~813).  All call-sites
should now import from here.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional, Tuple


class TimedCache:
    """In-memory cache with per-item TTL and LRU eviction.

    Parameters
    ----------
    max_items:
        Maximum number of entries to keep.
    ttl:
        Time-to-live in seconds for each entry.
    """

    def __init__(self, max_items: int = 128, ttl: float = 60.0) -> None:
        self.max_items: int = max(1, int(max_items))
        self.ttl: float = float(ttl)
        self._store: OrderedDict[tuple, Tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove expired and overflowing entries (caller must hold lock)."""
        now = time.time()
        dead = [key for key, (expires, _val) in self._store.items() if expires <= now]
        for key in dead:
            self._store.pop(key, None)
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: tuple) -> Optional[Any]:
        """Return cached value or ``None`` if missing/expired."""
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

    def set(self, key: tuple, value: Any) -> None:
        """Store *value* under *key* with the configured TTL."""
        with self._lock:
            self._store[key] = (time.time() + self.ttl, value)
            self._store.move_to_end(key)
            self._prune()

    def get_or_set(self, key: tuple, factory: Callable[[], Any]) -> Any:
        """Return cached value, calling *factory* on miss."""
        cached = self.get(key)
        if cached is not None:
            return cached
        value = factory()
        self.set(key, value)
        return value

    def clear(self) -> None:
        """Drop all entries."""
        with self._lock:
            self._store.clear()

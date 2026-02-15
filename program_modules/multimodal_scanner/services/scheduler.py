"""Simple periodic scheduler."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict


@dataclass
class Task:
    name: str
    interval: float
    handler: Callable[[], None]
    last_run: float = 0.0


class Scheduler:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def add_task(self, name: str, interval: float, handler: Callable[[], None]) -> None:
        self._tasks[name] = Task(name=name, interval=interval, handler=handler)

    def start(self) -> None:
        if self._thread:
            return

        def _loop():
            while not self._stop.is_set():
                now = time.time()
                for task in self._tasks.values():
                    if now - task.last_run >= task.interval:
                        try:
                            task.handler()
                        finally:
                            task.last_run = now
                self._stop.wait(1.0)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None
            self._stop.clear()

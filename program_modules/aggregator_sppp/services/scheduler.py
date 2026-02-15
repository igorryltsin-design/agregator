"""Background scheduler that triggers periodic maintenance tasks."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict

from ..config import AppConfig

LOGGER = logging.getLogger("aggregator.scheduler")


@dataclass
class ScheduledTask:
    name: str
    interval: float
    handler: Callable[[], None]
    last_run: float = 0.0


class SchedulerService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._tasks: Dict[str, ScheduledTask] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def add_task(self, name: str, interval: float, handler: Callable[[], None]) -> None:
        self._tasks[name] = ScheduledTask(name=name, interval=interval, handler=handler)

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
                            task.last_run = now
                        except Exception as exc:
                            LOGGER.exception("Scheduled task %s failed: %s", task.name, exc)
                self._stop.wait(1.0)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None
            self._stop.clear()

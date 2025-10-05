"""Простая очередь фоновых задач для Agregator."""

from __future__ import annotations

import atexit
import os
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

import logging


@dataclass(slots=True)
class BackgroundJob:
    task_id: str
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    description: str | None = None


class TaskQueue:
    """Небольшая очередь на основе потоков, предотвращающая блокировки веб-процесса."""

    def __init__(self, name: str, max_workers: int = 2, logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.max_workers = max_workers
        self._queue: queue.Queue[Optional[BackgroundJob]] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._shutdown = threading.Event()
        self._logger = logger or logging.getLogger(f"agregator.task_queue.{name}")
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for idx in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"{self.name}-worker-{idx+1}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()
            atexit.register(self.shutdown)
            self._logger.info("Task queue '%s' запущена с %s потоками", self.name, self.max_workers)

    def submit(
        self,
        func: Callable[..., Any],
        *args: Any,
        description: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self._started:
            self.start()
        job = BackgroundJob(
            task_id=str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            description=description,
        )
        self._queue.put(job)
        self._logger.debug("Фоновая задача %s поставлена в очередь (%s)", job.task_id, description or func.__name__)
        return job.task_id

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            try:
                worker.join(timeout=2)
            except Exception:
                pass
        self._workers.clear()
        self._logger.info("Task queue '%s' остановлена", self.name)

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break
            self._execute(job)
            self._queue.task_done()

    def _execute(self, job: BackgroundJob) -> None:
        desc = job.description or job.func.__name__
        self._logger.debug("Запуск задачи %s (%s)", job.task_id, desc)
        try:
            job.func(*job.args, **job.kwargs)
        except Exception as exc:  # noqa: BLE001 - хотим любой эксепшн
            self._logger.exception("Ошибка фоновой задачи %s (%s): %s", job.task_id, desc, exc)
        else:
            self._logger.debug("Задача %s завершена", job.task_id)


    def stats(self) -> dict[str, Any]:
        """Return basic queue statistics."""
        return {
            "name": self.name,
            "workers": len(self._workers),
            "max_workers": self.max_workers,
            "queued": self._queue.qsize(),
            "shutdown": self._shutdown.is_set(),
            "started": self._started,
        }


_default_queue: TaskQueue | None = None


def get_task_queue() -> TaskQueue:
    global _default_queue
    if _default_queue is None:
        max_workers = int(os.getenv("AGREGATOR_TASK_WORKERS", "2") or "2")
        _default_queue = TaskQueue(name="agregator", max_workers=max_workers)
    return _default_queue

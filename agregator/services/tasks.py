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
    owner_id: int | None = None


class TaskQueue:
    """Небольшая очередь на основе потоков, предотвращающая блокировки веб-процесса."""

    def __init__(
        self,
        name: str,
        max_workers: int = 2,
        logger: Optional[logging.Logger] = None,
        per_owner_max_workers: int = 1,
    ) -> None:
        self.name = name
        self.max_workers = max_workers
        self.per_owner_max_workers = max(1, int(per_owner_max_workers or 1))
        self._queue: queue.Queue[Optional[BackgroundJob]] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._shutdown = threading.Event()
        self._logger = logger or logging.getLogger(f"agregator.task_queue.{name}")
        self._started = False
        self._lock = threading.Lock()
        self._active_by_owner: dict[int, int] = {}

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            worker_count = self.max_workers
            if worker_count <= 0:
                worker_count = 1
                self._logger.warning(
                    "Количество воркеров очереди %s скорректировано до 1 для продолжения работы.",
                    self.name,
                )
            if worker_count != self.max_workers:
                self.max_workers = worker_count
            for idx in range(worker_count):
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
        owner_id: int | None = None,
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
            owner_id=owner_id,
        )
        if not self._workers:
            self._logger.warning(
                "Очередь '%s' не имеет активных воркеров, задача %s (%s) будет выполнена во временном потоке.",
                self.name,
                job.task_id,
                description or func.__name__,
            )
            threading.Thread(
                target=self._run_ad_hoc,
                args=(job,),
                name=f"{self.name}-adhoc-{job.task_id[:8]}",
                daemon=True,
            ).start()
        else:
            self._queue.put(job)
            self._logger.debug(
                "Фоновая задача %s поставлена в очередь (%s)",
                job.task_id,
                description or func.__name__,
            )
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
            if not self._reserve_owner_slot(job):
                # Возвращаем задачу обратно в очередь, если лимит owner исчерпан.
                self._queue.put(job)
                self._queue.task_done()
                self._shutdown.wait(0.05)
                continue
            self._execute(job)
            self._release_owner_slot(job)
            self._queue.task_done()

    def _reserve_owner_slot(self, job: BackgroundJob) -> bool:
        owner_id = job.owner_id
        if owner_id is None:
            return True
        with self._lock:
            active = int(self._active_by_owner.get(owner_id, 0))
            if active >= self.per_owner_max_workers:
                return False
            self._active_by_owner[owner_id] = active + 1
            return True

    def _release_owner_slot(self, job: BackgroundJob) -> None:
        owner_id = job.owner_id
        if owner_id is None:
            return
        with self._lock:
            active = int(self._active_by_owner.get(owner_id, 0))
            if active <= 1:
                self._active_by_owner.pop(owner_id, None)
            else:
                self._active_by_owner[owner_id] = active - 1

    def _execute(self, job: BackgroundJob) -> None:
        desc = job.description or job.func.__name__
        self._logger.debug("Запуск задачи %s (%s)", job.task_id, desc)
        try:
            job.func(*job.args, **job.kwargs)
        except Exception as exc:  # noqa: BLE001 - хотим любой эксепшн
            self._logger.exception("Ошибка фоновой задачи %s (%s): %s", job.task_id, desc, exc)
        else:
            self._logger.debug("Задача %s завершена", job.task_id)

    def _run_ad_hoc(self, job: BackgroundJob) -> None:
        """Execute a job outside the queue when no workers are running."""
        self._execute(job)


    def stats(self) -> dict[str, Any]:
        """Return basic queue statistics."""
        return {
            "name": self.name,
            "workers": len(self._workers),
            "max_workers": self.max_workers,
            "per_owner_max_workers": self.per_owner_max_workers,
            "queued": self._queue.qsize(),
            "active_by_owner": dict(self._active_by_owner),
            "shutdown": self._shutdown.is_set(),
            "started": self._started,
        }


_default_queue: TaskQueue | None = None


def _resolve_worker_count() -> int:
    """Resolve and sanitize background worker count from environment."""
    raw_value = os.getenv("AGREGATOR_TASK_WORKERS", "2")
    logger = logging.getLogger("agregator.task_queue")
    max_workers: int
    try:
        max_workers = int(str(raw_value).strip() or "2")
    except (TypeError, ValueError):
        logger.warning("Некорректное значение AGREGATOR_TASK_WORKERS=%r, используем 2 потока.", raw_value)
        max_workers = 2
    if max_workers <= 0:
        logger.warning("AGREGATOR_TASK_WORKERS=%s, увеличено до 1 для корректной работы фоновых задач.", max_workers)
        max_workers = 1
    return max_workers


def _resolve_per_owner_worker_count() -> int:
    """Resolve and sanitize per-owner worker limit from environment."""
    raw_value = os.getenv("AGREGATOR_TASK_PER_OWNER_MAX", "1")
    logger = logging.getLogger("agregator.task_queue")
    per_owner_max: int
    try:
        per_owner_max = int(str(raw_value).strip() or "1")
    except (TypeError, ValueError):
        logger.warning("Некорректное значение AGREGATOR_TASK_PER_OWNER_MAX=%r, используем 1.", raw_value)
        per_owner_max = 1
    if per_owner_max <= 0:
        logger.warning("AGREGATOR_TASK_PER_OWNER_MAX=%s, увеличено до 1.", per_owner_max)
        per_owner_max = 1
    return per_owner_max


def get_task_queue() -> TaskQueue:
    global _default_queue
    if _default_queue is None:
        max_workers = _resolve_worker_count()
        per_owner_max_workers = _resolve_per_owner_worker_count()
        _default_queue = TaskQueue(
            name="agregator",
            max_workers=max_workers,
            per_owner_max_workers=per_owner_max_workers,
        )
    return _default_queue

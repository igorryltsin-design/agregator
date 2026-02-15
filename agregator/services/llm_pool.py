"""Owner-aware LLM request pool with bounded queue."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional


class LlmPoolRejected(RuntimeError):
    """Raised when request cannot be queued due to backpressure."""


class LlmPoolTimeout(RuntimeError):
    """Raised when queued/running request exceeds timeout."""


@dataclass(slots=True)
class LlmJob:
    job_id: str
    owner_id: int | None
    fn: Callable[[], Any]
    submitted_at: float
    timeout_sec: float
    done: threading.Event
    result: Any = None
    error: Exception | None = None
    started_at: float | None = None
    finished_at: float | None = None


class LlmWorkerPool:
    """Thread-based queue for non-blocking LLM scheduling."""

    def __init__(self, *, workers: int = 4, per_user_limit: int = 1, max_queue: int = 100) -> None:
        self.workers = max(1, int(workers or 1))
        self.per_user_limit = max(1, int(per_user_limit or 1))
        self.max_queue = max(1, int(max_queue or 1))
        self._queue: queue.Queue[LlmJob] = queue.Queue(maxsize=self.max_queue)
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._active_by_owner: dict[int, int] = {}
        self._running_jobs: dict[str, LlmJob] = {}
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for idx in range(self.workers):
                thread = threading.Thread(target=self._worker_loop, daemon=True, name=f"llm-pool-{idx+1}")
                self._threads.append(thread)
                thread.start()

    def submit_and_wait(
        self,
        *,
        owner_id: int | None,
        fn: Callable[[], Any],
        timeout_sec: float,
        per_user_queue_max: int = 10,
    ) -> Any:
        if not self._started:
            self.start()
        timeout_sec = max(1.0, float(timeout_sec or 1.0))
        if self._queue.full():
            raise LlmPoolRejected("llm_queue_full")
        if owner_id is not None:
            with self._lock:
                queued_for_owner = sum(1 for job in list(self._queue.queue) if job.owner_id == owner_id)
                running_for_owner = self._active_by_owner.get(owner_id, 0)
                if queued_for_owner + running_for_owner >= max(1, int(per_user_queue_max)):
                    raise LlmPoolRejected("llm_owner_queue_full")
        job = LlmJob(
            job_id=uuid.uuid4().hex,
            owner_id=owner_id,
            fn=fn,
            submitted_at=time.time(),
            timeout_sec=timeout_sec,
            done=threading.Event(),
        )
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            raise LlmPoolRejected("llm_queue_full") from exc
        # wait includes queue wait + run
        finished = job.done.wait(timeout=timeout_sec)
        if not finished:
            raise LlmPoolTimeout("llm_timeout")
        if job.error is not None:
            raise job.error
        return job.result

    def stats(self) -> dict[str, Any]:
        with self._lock:
            queue_snapshot = list(self._queue.queue)
            now = time.time()
            wait_samples = [max(0.0, now - job.submitted_at) for job in queue_snapshot]
            avg_wait = sum(wait_samples) / len(wait_samples) if wait_samples else 0.0
            p95_wait = sorted(wait_samples)[int(max(0, len(wait_samples) - 1) * 0.95)] if wait_samples else 0.0
            return {
                "started": self._started,
                "workers": self.workers,
                "per_user_limit": self.per_user_limit,
                "max_queue": self.max_queue,
                "queued": len(queue_snapshot),
                "running": len(self._running_jobs),
                "active_by_owner": dict(self._active_by_owner),
                "avg_queue_wait_sec": round(avg_wait, 3),
                "p95_queue_wait_sec": round(float(p95_wait), 3),
            }

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if not self._reserve_owner_slot(job.owner_id):
                self._queue.put(job)
                self._queue.task_done()
                self._shutdown.wait(0.03)
                continue
            with self._lock:
                self._running_jobs[job.job_id] = job
            job.started_at = time.time()
            try:
                job.result = job.fn()
            except Exception as exc:  # noqa: BLE001
                job.error = exc
            finally:
                job.finished_at = time.time()
                job.done.set()
                with self._lock:
                    self._running_jobs.pop(job.job_id, None)
                self._release_owner_slot(job.owner_id)
                self._queue.task_done()

    def _reserve_owner_slot(self, owner_id: int | None) -> bool:
        if owner_id is None:
            return True
        with self._lock:
            active = int(self._active_by_owner.get(owner_id, 0))
            if active >= self.per_user_limit:
                return False
            self._active_by_owner[owner_id] = active + 1
            return True

    def _release_owner_slot(self, owner_id: int | None) -> None:
        if owner_id is None:
            return
        with self._lock:
            active = int(self._active_by_owner.get(owner_id, 0))
            if active <= 1:
                self._active_by_owner.pop(owner_id, None)
            else:
                self._active_by_owner[owner_id] = active - 1


_default_pool: LlmWorkerPool | None = None


def configure_llm_pool(*, workers: int, per_user_limit: int, max_queue: int) -> LlmWorkerPool:
    global _default_pool
    _default_pool = LlmWorkerPool(workers=workers, per_user_limit=per_user_limit, max_queue=max_queue)
    _default_pool.start()
    return _default_pool


def get_llm_pool() -> LlmWorkerPool:
    global _default_pool
    if _default_pool is None:
        _default_pool = LlmWorkerPool()
        _default_pool.start()
    return _default_pool

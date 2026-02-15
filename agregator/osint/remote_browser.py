"""Remote interactive browser handling for OSINT captcha flows."""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timedelta
from typing import Dict

from agregator.services.browser import get_browser_manager


class _RemoteBrowserTask:
    def __init__(self, action: str, payload: Dict[str, float | str]) -> None:
        self.action = action
        self.payload = payload
        self.result: bytes | None = None
        self.error: Exception | None = None
        self._event = threading.Event()

    def set_result(self, result: bytes | None) -> None:
        self.result = result
        self._event.set()

    def set_error(self, exc: Exception) -> None:
        self.error = exc
        self._event.set()

    def wait(self, timeout: float | None = 30.0) -> None:
        self._event.wait(timeout=timeout)


class RemoteBrowserSession:
    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._queue: "queue.Queue[_RemoteBrowserTask]" = queue.Queue()
        self._last_snapshot: bytes | None = None
        self._last_activity = datetime.utcnow()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop_event = threading.Event()
        self._thread.start()
        self._initialized = threading.Event()
        self._init_error: Exception | None = None

    def _run(self) -> None:
        try:
            context = get_browser_manager().new_context()
            page = context.new_page()
        except Exception as exc:
            self._init_error = exc
            self._initialized.set()
            return
        self._initialized.set()
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if task.action == "shutdown":
                    break
                try:
                    self._handle_task(task, page)
                finally:
                    self._queue.task_done()
        finally:
            try:
                page.close()
            except Exception:
                pass
            try:
                context.close()
            except Exception:
                pass

    def _handle_task(self, task: _RemoteBrowserTask, page) -> None:
        action = task.action
        payload = task.payload
        try:
            if action == "navigate":
                page.goto(payload["url"], wait_until="networkidle")
            elif action == "click":
                page.mouse.click(payload["x"], payload["y"])
            elif action == "type":
                page.keyboard.type(payload["text"])
            elif action == "screenshot":
                image = page.screenshot(full_page=True)
                self._last_snapshot = image
                task.set_result(image)
                return
            else:
                task.set_error(RuntimeError("unsupported_action"))
                return
            self._last_activity = datetime.utcnow()
            task.set_result(self._last_snapshot)
        except Exception as exc:
            task.set_error(exc)

    def _enqueue(self, action: str, payload: Dict[str, float | str]) -> bytes | None:
        if self._init_error:
            raise self._init_error
        self._initialized.wait()
        if self._init_error:
            raise self._init_error
        task = _RemoteBrowserTask(action, payload)
        self._queue.put(task)
        task.wait()
        if task.error:
            raise task.error
        return task.result

    def navigate(self, url: str) -> None:
        self._enqueue("navigate", {"url": url})

    def click(self, x: float, y: float) -> None:
        self._enqueue("click", {"x": x, "y": y})

    def type_text(self, text: str) -> None:
        self._enqueue("type", {"text": text})

    def screenshot(self) -> bytes | None:
        return self._enqueue("screenshot", {})

    def close(self) -> None:
        self._stop_event.set()
        self._queue.put(_RemoteBrowserTask("shutdown", {}))
        self._thread.join(timeout=3)

    def is_idle(self, timeout_seconds: int) -> bool:
        return (datetime.utcnow() - self._last_activity) > timedelta(seconds=timeout_seconds)


class RemoteBrowserManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, RemoteBrowserSession] = {}
        self._lock = threading.Lock()

    def get(self, source_id: str) -> RemoteBrowserSession | None:
        with self._lock:
            return self._sessions.get(source_id)

    def create(self, source_id: str, url: str) -> RemoteBrowserSession:
        with self._lock:
            session = self._sessions.get(source_id)
            if session:
                session.close()
            session = RemoteBrowserSession(source_id)
            self._sessions[source_id] = session
        session.navigate(url)
        self._cleanup()
        return session

    def close(self, source_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(source_id, None)
        if session:
            session.close()

    def _cleanup(self) -> None:
        timeout = 300
        with self._lock:
            ids = list(self._sessions.keys())
        for sid in ids:
            session = self._sessions.get(sid)
            if session and session.is_idle(timeout):
                self.close(sid)


remote_browser_manager = RemoteBrowserManager()

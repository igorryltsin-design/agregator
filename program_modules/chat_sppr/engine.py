"""Engine orchestrating dialog logic."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List

from sqlalchemy import select

from . import models, utils
from .config import AppConfig
from .database import Database
from .repositories import AuditRepository, SessionRepository, TelemetryRepository
from .services.intent import IntentService
from .services.responder import ResponseService
from .services.retrieval import RetrievalService
from .services.scheduler import Scheduler
from .services.session import SessionService
from .services.watchdog import WatchdogService
from .telemetry import Telemetry


@dataclass
class EngineResult:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None


class ChatEngine:
    def __init__(self, config: AppConfig, database: Database):
        self.config = config
        self.database = database
        self.session_service = SessionService(config)
        self.intent_service = IntentService(config)
        self.retrieval_service = RetrievalService(config)
        self.response_service = ResponseService(config)
        self.watchdog_service = WatchdogService(config)
        self.telemetry = Telemetry(max_samples=config.telemetry.window)
        self.scheduler = Scheduler()
        self._setup_background_tasks()

    def _setup_background_tasks(self) -> None:
        if self.config.telemetry.enable:
            self.telemetry.configure_export(
                interval=self.config.telemetry.interval.total_seconds(),
                callback=self._persist_telemetry,
            )
            self.telemetry.ensure_running()
            self.scheduler.add_task(
                "telemetry",
                self.config.telemetry.interval.total_seconds(),
                lambda: None,
            )
        self.scheduler.start()

    @contextmanager
    def _session(self):
        with self.database.session() as session:
            yield session

    def ensure_user(self, payload: dict) -> EngineResult:
        with self._session() as session:
            result = self.session_service.ensure_user(
                session,
                payload["username"],
                payload.get("display_name", payload["username"]),
                payload.get("role", "operator"),
            )
            return EngineResult(True, result)

    def start_session(self, payload: dict) -> EngineResult:
        required = {"user_id", "title"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"missing fields: {', '.join(sorted(missing))}")
        with self._session() as session:
            result = self.session_service.start_session(session, payload["user_id"], payload["title"])
            self.telemetry.observe("sessions.started", 1.0)
            return EngineResult(True, result)

    def ask(self, payload: dict) -> EngineResult:
        required = {"session_id", "user_message"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"missing fields: {', '.join(sorted(missing))}")
        question = utils.normalize_text(payload["user_message"])
        with self._session() as session:
            history_turn = self.session_service.append_turn(
                session,
                session_id=payload["session_id"],
                role="user",
                content=question,
                tokens=len(utils.tokenize(question)),
                references=[],
                confidence=0.0,
            )
            intent = self.intent_service.detect(session, payload["session_id"], question)
            suggestions = self.intent_service.suggestions(question)
            references = self.retrieval_service.retrieve(session, question)
            response = self.response_service.compose(question, references)
            bot_turn = self.session_service.append_turn(
                session,
                session_id=payload["session_id"],
                role="assistant",
                content=response["text"],
                tokens=response["tokens"],
                references=response["references"],
                confidence=response["confidence"],
            )
            self.telemetry.observe("turns.total", 1.0)
            self.telemetry.observe("turns.tokens", response["tokens"])
            self.session_service.queue_suggestion(
                session,
                payload["session_id"],
                {"intent": intent["intent"], "hints": suggestions},
            )
            return EngineResult(
                True,
                {
                    "turn_id": bot_turn["turn_id"],
                    "response": response["text"],
                    "confidence": response["confidence"],
                    "references": references,
                    "intent": intent,
                    "suggestions": suggestions,
                },
            )

    def history(self, payload: dict) -> EngineResult:
        with self._session() as session:
            history = self.session_service.history(session, payload["session_id"], limit=payload.get("limit", 50))
            suggestions = self.session_service.fetch_suggestions(session, payload["session_id"])
            return EngineResult(True, {"history": history, "suggestions": suggestions})

    def close_session(self, payload: dict) -> EngineResult:
        with self._session() as session:
            self.session_service.close_session(session, payload["session_id"])
            self.telemetry.observe("sessions.closed", 1.0)
            return EngineResult(True, {"session_id": payload["session_id"], "status": "closed"})

    def feedback(self, payload: dict) -> EngineResult:
        with self._session() as session:
            result = self.session_service.feedback(
                session,
                turn_id=payload["turn_id"],
                score=payload["score"],
                comment=payload.get("comment"),
            )
            self.telemetry.observe("feedback.score", payload["score"])
            return EngineResult(True, result)

    def audit(self) -> EngineResult:
        with self._session() as session:
            repo = AuditRepository(session)
            entries = repo.latest()
            return EngineResult(
                True,
                {
                    "events": [
                        {"type": entry.event_type, "payload": entry.payload, "created_at": entry.created_at.isoformat()}
                        for entry in entries
                    ]
                },
            )

    def telemetry_snapshot(self) -> EngineResult:
        snapshot = self.telemetry.snapshot()
        return EngineResult(True, {"telemetry": snapshot})

    def report(self) -> EngineResult:
        with self._session() as session:
            repo = TelemetryRepository(session)
            snapshots = repo.recent("chat.metrics", limit=5)
            return EngineResult(
                True,
                {
                    "snapshots": [
                        {
                            "window_start": snap.window_start.isoformat(),
                            "window_end": snap.window_end.isoformat(),
                            "payload": snap.payload,
                        }
                        for snap in snapshots
                    ]
                },
            )

    def watchdog(self) -> EngineResult:
        with self._session() as session:
            summary = self.watchdog_service.inspect(session)
            payload = {
                "stale_sessions": summary.stale_sessions,
                "unanswered_turns": summary.unanswered_turns,
                "feedback": summary.low_feedback,
            }
            return EngineResult(True, payload)

    def shutdown(self) -> None:
        self.scheduler.stop()
        self.telemetry.stop()

    def _persist_telemetry(self, snapshot: Dict[str, Dict[str, float]]) -> None:
        with self._session() as session:
            repo = TelemetryRepository(session)
            now = utils.utc_now()
            repo.store(
                "chat.metrics",
                window_start=now - self.config.telemetry.interval,
                window_end=now,
                payload=snapshot,
            )

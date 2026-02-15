"""Repository helpers for chat module."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


class BaseRepository:
    def __init__(self, session: Session):
        self.session = session


class UserRepository(BaseRepository):
    def get_or_create(self, username: str, display_name: str, role: str) -> models.User:
        stmt = select(models.User).where(models.User.username == username)
        user = self.session.scalar(stmt)
        if user:
            return user
        user = models.User(username=username, display_name=display_name, role=role)
        self.session.add(user)
        self.session.flush()
        return user


class SessionRepository(BaseRepository):
    def create(self, user_id: int, title: str) -> models.ChatSession:
        session = models.ChatSession(user_id=user_id, title=title)
        self.session.add(session)
        self.session.flush()
        return session

    def resolve(self, session_id: int) -> models.ChatSession | None:
        return self.session.get(models.ChatSession, session_id)

    def close(self, session_id: int) -> None:
        chat_session = self.resolve(session_id)
        if chat_session:
            chat_session.status = "closed"
            chat_session.updated_at = datetime.utcnow()

    def list_active(self, user_id: int, limit: int = 10) -> List[models.ChatSession]:
        stmt = (
            select(models.ChatSession)
            .where(models.ChatSession.user_id == user_id, models.ChatSession.status == "active")
            .order_by(models.ChatSession.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class TurnRepository(BaseRepository):
    def add_turn(
        self,
        session_id: int,
        role: str,
        content: str,
        tokens: int,
        references: List[int],
        confidence: float,
    ) -> models.Turn:
        position = self._next_position(session_id)
        turn = models.Turn(
            session_id=session_id,
            position=position,
            role=role,
            content=content,
            tokens=tokens,
            references=references,
            confidence=confidence,
        )
        self.session.add(turn)
        self.session.flush()
        chat_session = self.session.get(models.ChatSession, session_id)
        if chat_session:
            chat_session.updated_at = datetime.utcnow()
        return turn

    def history(self, session_id: int, limit: int = 50) -> List[models.Turn]:
        stmt = (
            select(models.Turn)
            .where(models.Turn.session_id == session_id)
            .order_by(models.Turn.position.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def _next_position(self, session_id: int) -> int:
        stmt = select(func.max(models.Turn.position)).where(models.Turn.session_id == session_id)
        current = self.session.scalar(stmt) or 0
        return current + 1


class IntentRepository(BaseRepository):
    def record(self, session_id: int, intent: str, confidence: float, attributes: Dict) -> models.IntentSuggestion:
        suggestion = models.IntentSuggestion(
            session_id=session_id,
            intent=intent,
            confidence=confidence,
            attributes=attributes,
        )
        self.session.add(suggestion)
        self.session.flush()
        return suggestion

    def latest(self, session_id: int, limit: int = 5) -> List[models.IntentSuggestion]:
        stmt = (
            select(models.IntentSuggestion)
            .where(models.IntentSuggestion.session_id == session_id)
            .order_by(models.IntentSuggestion.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class ReferenceRepository(BaseRepository):
    def attach(self, turn_id: int, references: Sequence[Dict[str, str]]) -> None:
        for payload in references:
            reference = models.KnowledgeReference(
                turn_id=turn_id,
                source_type=payload["source_type"],
                source_id=payload["source_id"],
                snippet=payload["snippet"],
                score=payload.get("score", 0.0),
            )
            self.session.add(reference)

    def list_for_turn(self, turn_id: int) -> List[models.KnowledgeReference]:
        stmt = select(models.KnowledgeReference).where(models.KnowledgeReference.turn_id == turn_id)
        return list(self.session.scalars(stmt))


class SuggestionRepository(BaseRepository):
    def enqueue(self, session_id: int, payload: Dict, expires_at: datetime) -> models.SuggestionQueue:
        entry = models.SuggestionQueue(session_id=session_id, payload=payload, expires_at=expires_at)
        self.session.add(entry)
        self.session.flush()
        return entry

    def pull(self, session_id: int) -> List[models.SuggestionQueue]:
        stmt = (
            select(models.SuggestionQueue)
            .where(
                models.SuggestionQueue.session_id == session_id,
                models.SuggestionQueue.processed.is_(False),
                models.SuggestionQueue.expires_at > datetime.utcnow(),
            )
            .order_by(models.SuggestionQueue.id.asc())
        )
        entries = list(self.session.scalars(stmt))
        for entry in entries:
            entry.processed = True
        return entries


class FeedbackRepository(BaseRepository):
    def add(self, turn_id: int, score: int, comment: str | None) -> models.Feedback:
        feedback = models.Feedback(turn_id=turn_id, score=score, comment=comment)
        self.session.add(feedback)
        self.session.flush()
        return feedback

    def stats(self, since: datetime) -> Dict[str, float]:
        stmt = select(func.avg(models.Feedback.score)).where(models.Feedback.created_at >= since)
        avg_score = self.session.scalar(stmt) or 0.0
        count = self.session.scalar(
            select(func.count(models.Feedback.id)).where(models.Feedback.created_at >= since)
        ) or 0
        return {"avg_score": float(avg_score), "count": int(count)}


class AuditRepository(BaseRepository):
    def add(self, event_type: str, payload: Dict) -> None:
        entry = models.AuditLog(event_type=event_type, payload=payload)
        self.session.add(entry)

    def latest(self, limit: int = 50) -> List[models.AuditLog]:
        stmt = select(models.AuditLog).order_by(models.AuditLog.created_at.desc()).limit(limit)
        return list(self.session.scalars(stmt))


class TelemetryRepository(BaseRepository):
    def store(self, name: str, window_start: datetime, window_end: datetime, payload: Dict) -> None:
        snapshot = models.TelemetrySnapshot(
            metric_name=name,
            window_start=window_start,
            window_end=window_end,
            payload=payload,
        )
        self.session.add(snapshot)

    def recent(self, name: str, limit: int = 5) -> List[models.TelemetrySnapshot]:
        stmt = (
            select(models.TelemetrySnapshot)
            .where(models.TelemetrySnapshot.metric_name == name)
            .order_by(models.TelemetrySnapshot.window_end.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

"""Session management service."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import (
    AuditRepository,
    FeedbackRepository,
    SessionRepository,
    SuggestionRepository,
    TurnRepository,
    UserRepository,
)
from .. import utils

LOGGER = logging.getLogger("chat.session")


class SessionService:
    def __init__(self, config: AppConfig):
        self.config = config

    def ensure_user(self, session: Session, username: str, display_name: str, role: str) -> dict:
        repo = UserRepository(session)
        user = repo.get_or_create(username, display_name, role)
        return {"user_id": user.id, "username": user.username, "role": user.role}

    def start_session(self, session: Session, user_id: int, title: str) -> dict:
        repo = SessionRepository(session)
        chat_session = repo.create(user_id, title=title)
        LOGGER.info("Session %s started for user %s", chat_session.id, user_id)
        return {"session_id": chat_session.id, "title": chat_session.title}

    def append_turn(
        self,
        session: Session,
        session_id: int,
        role: str,
        content: str,
        tokens: int,
        references: list[int] | None = None,
        confidence: float = 0.0,
    ) -> dict:
        reference_ids = references or []
        turn_repo = TurnRepository(session)
        turn = turn_repo.add_turn(
            session_id=session_id,
            role=role,
            content=content,
            tokens=tokens,
            references=reference_ids,
            confidence=confidence,
        )
        return {
            "turn_id": turn.id,
            "position": turn.position,
            "content": turn.content,
            "role": turn.role,
            "references": turn.references,
        }

    def history(self, session_db: Session, session_id: int, limit: int = 50) -> list[dict]:
        repo = TurnRepository(session_db)
        turns = repo.history(session_id, limit=limit)
        return [
            {
                "position": turn.position,
                "role": turn.role,
                "content": turn.content,
                "confidence": turn.confidence,
                "created_at": turn.created_at.isoformat(),
            }
            for turn in turns
        ]

    def queue_suggestion(self, session: Session, session_id: int, payload: dict) -> None:
        repo = SuggestionRepository(session)
        expires_at = utils.utc_now() + timedelta(seconds=self.config.intents.cooldown_sec)
        repo.enqueue(session_id, payload, expires_at)

    def fetch_suggestions(self, session: Session, session_id: int) -> list[dict]:
        repo = SuggestionRepository(session)
        items = repo.pull(session_id)
        return [{"payload": item.payload, "expires_at": item.expires_at.isoformat()} for item in items]

    def close_session(self, session: Session, session_id: int) -> None:
        repo = SessionRepository(session)
        repo.close(session_id)
        LOGGER.info("Session %s closed", session_id)

    def feedback(self, session: Session, turn_id: int, score: int, comment: str | None) -> dict:
        repo = FeedbackRepository(session)
        record = repo.add(turn_id, score, comment)
        return {"feedback_id": record.id, "score": record.score}

"""History service keeps sequential log of pipeline events."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import HistoryRepository, ReplayCursorRepository


class HistoryService:
    def __init__(self, config: AppConfig):
        self.config = config

    def append(self, session: Session, event_type: str, payload: dict) -> int:
        repo = HistoryRepository(session)
        event = repo.append(event_type, payload, token=event_type[:8])
        return event.id

    def replay(self, session: Session, cursor_name: str, limit: int = 50) -> List[dict]:
        cursor_repo = ReplayCursorRepository(session)
        history_repo = HistoryRepository(session)

        cursor = cursor_repo.get(cursor_name)
        since_id = cursor.last_event_id if cursor else 0
        events = history_repo.fetch_window(since_id, limit=limit)
        if events:
            cursor_repo.update(cursor_name, events[-1].id)
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "payload": event.payload,
                "created_at": event.created_at.isoformat(),
            }
            for event in events
        ]

    def purge(self, session: Session) -> int:
        repo = HistoryRepository(session)
        cutoff = datetime.utcnow() - timedelta(days=self.config.history.retention_days)
        return repo.purge_older_than(cutoff)

"""Statistics aggregation for scanner."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import StatRepository, modality_stats


class StatisticsService:
    def __init__(self, config: AppConfig):
        self.config = config

    def refresh(self, session: Session) -> Dict:
        stats_repo = StatRepository(session)
        window = self.config.stats_window
        data = modality_stats(session, window)
        now = datetime.utcnow()
        stats_repo.store(
            "modality",
            window_start=now - window,
            window_end=now,
            payload=data,
        )
        return data

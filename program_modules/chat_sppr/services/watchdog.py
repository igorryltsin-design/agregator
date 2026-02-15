"""Watchdog service for chat module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import models, utils


@dataclass
class WatchdogSummary:
    stale_sessions: List[int]
    unanswered_turns: int
    low_feedback: Dict[str, float]


class WatchdogService:
    def __init__(self, config):
        self.config = config

    def inspect(self, session: Session) -> WatchdogSummary:
        cutoff = utils.utc_now() - self.config.session.idle_timeout
        stale = session.scalars(
            select(models.ChatSession.id)
            .where(models.ChatSession.status == "active")
            .where(models.ChatSession.updated_at < cutoff)
        ).all()
        unanswered = session.scalar(
            select(func.count(models.Turn.id)).where(
                models.Turn.role == "user",
                ~models.Turn.id.in_(select(models.KnowledgeReference.turn_id)),
            )
        ) or 0
        avg = session.scalar(select(func.avg(models.Feedback.score))) or 0.0
        return WatchdogSummary(
            stale_sessions=stale,
            unanswered_turns=int(unanswered),
            low_feedback={"avg_score": float(avg)},
        )

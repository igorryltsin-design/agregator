"""Watchdog service detecting anomalies in scan results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .. import models
from ..config import AppConfig


@dataclass
class WatchdogReport:
    error_entries: int
    high_entropy: List[Dict]
    duplicates: List[Dict]


class WatchdogService:
    def __init__(self, config: AppConfig):
        self.config = config

    def inspect(self, session: Session, limit: int = 10) -> WatchdogReport:
        errors = session.scalars(
            select(models.ScanLog)
            .where(models.ScanLog.level == "error")
            .order_by(models.ScanLog.created_at.desc())
            .limit(limit)
        ).all()

        entropy_threshold = 0.9
        high_entropy_rows = session.scalars(
            select(models.Artifact)
            .where(models.Artifact.entropy >= entropy_threshold)
            .order_by(models.Artifact.entropy.desc())
            .limit(limit)
        ).all()
        high_entropy = [
            {"id": row.id, "path": row.path, "entropy": row.entropy} for row in high_entropy_rows
        ]

        duplicates_rows = session.execute(
            select(models.Artifact.signature, func.count(models.Artifact.id))
            .group_by(models.Artifact.signature)
            .having(func.count(models.Artifact.id) > 1)
            .limit(limit)
        ).all()
        duplicates = [{"signature": signature, "count": count} for signature, count in duplicates_rows]

        return WatchdogReport(
            error_entries=len(errors),
            high_entropy=high_entropy,
            duplicates=duplicates,
        )

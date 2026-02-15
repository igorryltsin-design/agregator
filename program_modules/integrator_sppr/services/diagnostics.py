"""Diagnostics helpers for quick health summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models


@dataclass
class NodeStatus:
    name: str
    status: str
    latency_ms: float


class DiagnosticService:
    def snapshot(self, session: Session) -> List[NodeStatus]:
        rows = session.scalars(
            select(models.Heartbeat).join(models.Node).order_by(models.Node.name)
        ).all()
        return [
            NodeStatus(
                name=row.node.name if row.node else "unknown",
                status=row.status,
                latency_ms=row.latency_ms,
            )
            for row in rows
        ]

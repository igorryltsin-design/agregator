"""Repository layer that encapsulates persistence logic."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


class BaseRepository:
    def __init__(self, session: Session):
        self.session = session


class DataSourceRepository(BaseRepository):
    def create(
        self,
        name: str,
        modality: str,
        reliability: float,
        priority: int,
        description: str | None = None,
    ) -> models.DataSource:
        entity = models.DataSource(
            name=name,
            modality=modality,
            reliability_baseline=reliability,
            priority=priority,
            description=description,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def get_by_name(self, name: str) -> models.DataSource | None:
        return self.session.scalar(
            select(models.DataSource).where(models.DataSource.name == name)
        )

    def list_all(self) -> List[models.DataSource]:
        return list(self.session.scalars(select(models.DataSource)))


class SignalRepository(BaseRepository):
    def add(
        self,
        source_id: int,
        recorded_at: datetime,
        signal_type: str,
        payload_ref: str,
        vector: list[float],
        attributes: dict,
        intensity: float,
        reliability: float,
    ) -> models.Signal:
        entity = models.Signal(
            source_id=source_id,
            recorded_at=recorded_at,
            signal_type=signal_type,
            payload_ref=payload_ref,
            vector=vector,
            attributes=attributes,
            intensity=intensity,
            reliability=reliability,
            normalized=True,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def fetch_window(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> List[models.Signal]:
        stmt = (
            select(models.Signal)
            .where(
                models.Signal.recorded_at >= window_start,
                models.Signal.recorded_at < window_end,
            )
            .order_by(models.Signal.recorded_at.asc())
        )
        return list(self.session.scalars(stmt))

    def latest_signals(self, limit: int = 50) -> List[models.Signal]:
        stmt = (
            select(models.Signal)
            .order_by(models.Signal.recorded_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class FusionRepository(BaseRepository):
    def create_sample(
        self,
        start_at: datetime,
        end_at: datetime,
        summary: str,
        context_score: float,
        coherence: float,
    ) -> models.FusionSample:
        entity = models.FusionSample(
            start_at=start_at,
            end_at=end_at,
            summary=summary,
            context_score=context_score,
            coherence=coherence,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def attach_signals(
        self, sample_id: int, signal_ids: Sequence[int], weights: Sequence[float]
    ) -> None:
        for signal_id, weight in zip(signal_ids, weights, strict=False):
            link = models.FusionLink(
                sample_id=sample_id,
                signal_id=signal_id,
                weight=weight,
            )
            self.session.add(link)

    def latest_samples(self, limit: int = 20) -> List[models.FusionSample]:
        stmt = (
            select(models.FusionSample)
            .order_by(models.FusionSample.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class GraphRepository(BaseRepository):
    def upsert_relation(
        self,
        from_signal: int,
        to_signal: int,
        relation_type: str,
        similarity: float,
        weight: float,
        window_start: datetime,
        window_end: datetime,
    ) -> models.RelationEdge:
        stmt = select(models.RelationEdge).where(
            models.RelationEdge.from_signal_id == from_signal,
            models.RelationEdge.to_signal_id == to_signal,
            models.RelationEdge.relation_type == relation_type,
        )
        edge = self.session.scalar(stmt)
        if edge:
            edge.similarity = similarity
            edge.weight = weight
            edge.window_start = window_start
            edge.window_end = window_end
        else:
            edge = models.RelationEdge(
                from_signal_id=from_signal,
                to_signal_id=to_signal,
                relation_type=relation_type,
                similarity=similarity,
                weight=weight,
                window_start=window_start,
                window_end=window_end,
            )
            self.session.add(edge)
        self.session.flush()
        return edge

    def neighborhood(self, signal_id: int, limit: int = 20) -> List[models.RelationEdge]:
        stmt = (
            select(models.RelationEdge)
            .where(
                (models.RelationEdge.from_signal_id == signal_id)
                | (models.RelationEdge.to_signal_id == signal_id)
            )
            .order_by(models.RelationEdge.similarity.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class HistoryRepository(BaseRepository):
    def append(self, event_type: str, payload: dict, token: str) -> models.HistoryEvent:
        entity = models.HistoryEvent(
            event_type=event_type,
            payload=payload,
            replay_token=token,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def fetch_window(self, since_id: int, limit: int = 100) -> List[models.HistoryEvent]:
        stmt = (
            select(models.HistoryEvent)
            .where(models.HistoryEvent.id > since_id)
            .order_by(models.HistoryEvent.id.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def purge_older_than(self, cutoff: datetime) -> int:
        stmt = select(models.HistoryEvent.id).where(
            models.HistoryEvent.created_at < cutoff
        )
        ids = list(self.session.scalars(stmt))
        if not ids:
            return 0
        deleted = (
            self.session.query(models.HistoryEvent)
            .filter(models.HistoryEvent.id.in_(ids))
            .delete(synchronize_session=False)
        )
        return deleted


class LogRepository(BaseRepository):
    def add(self, component: str, level: str, message: str, extra: dict | None = None):
        entity = models.ProcessingLog(
            component=component,
            level=level,
            message=message,
            extra=extra,
        )
        self.session.add(entity)

    def latest(self, limit: int = 50) -> List[models.ProcessingLog]:
        stmt = (
            select(models.ProcessingLog)
            .order_by(models.ProcessingLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class ReplayCursorRepository(BaseRepository):
    def get(self, name: str) -> models.ReplayCursor | None:
        return self.session.scalar(
            select(models.ReplayCursor).where(models.ReplayCursor.name == name)
        )

    def update(self, name: str, event_id: int) -> models.ReplayCursor:
        cursor = self.get(name)
        if cursor:
            cursor.last_event_id = event_id
            cursor.updated_at = datetime.utcnow()
        else:
            cursor = models.ReplayCursor(name=name, last_event_id=event_id)
            self.session.add(cursor)
        self.session.flush()
        return cursor


def signal_activity_stats(session: Session, window: timedelta) -> dict:
    now = datetime.utcnow()
    since = now - window
    stmt = select(func.count(models.Signal.id)).where(
        models.Signal.created_at >= since
    )
    total = session.scalar(stmt) or 0
    by_modality = dict(
        session.execute(
            select(models.DataSource.modality, func.count(models.Signal.id))
            .join(models.Signal)
            .where(models.Signal.created_at >= since)
            .group_by(models.DataSource.modality)
        ).all()
    )
    return {"total": int(total), "by_modality": by_modality}

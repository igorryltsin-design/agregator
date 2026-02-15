"""Repository helpers for Contextual Analytics SPPR."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


class BaseRepository:
    def __init__(self, session: Session):
        self.session = session


class SourceRepository(BaseRepository):
    def get_or_create(
        self,
        name: str,
        modality: str,
        description: str | None,
        relevance_bias: float,
        latency_expectation: int,
    ) -> models.SourceChannel:
        entity = self.session.scalar(
            select(models.SourceChannel).where(models.SourceChannel.name == name)
        )
        if entity:
            return entity
        entity = models.SourceChannel(
            name=name,
            modality=modality,
            description=description,
            relevance_bias=relevance_bias,
            latency_expectation=latency_expectation,
        )
        self.session.add(entity)
        self.session.flush()
        return entity


class ObservationRepository(BaseRepository):
    def add(
        self,
        source_id: int,
        recorded_at: datetime,
        modality: str,
        content_ref: str,
        annotation: str | None,
        attributes: Dict,
        intensity: float,
        confidence: float,
    ) -> models.Observation:
        entity = models.Observation(
            source_id=source_id,
            recorded_at=recorded_at,
            modality=modality,
            content_ref=content_ref,
            annotation=annotation,
            attributes=attributes,
            intensity=intensity,
            confidence=confidence,
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def fetch_unprocessed(self, limit: int = 64) -> List[models.Observation]:
        stmt = (
            select(models.Observation)
            .where(models.Observation.processed.is_(False))
            .order_by(models.Observation.recorded_at.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def mark_processed(self, observation_ids: Sequence[int]) -> None:
        if not observation_ids:
            return
        self.session.query(models.Observation).filter(
            models.Observation.id.in_(observation_ids)
        ).update({"processed": True}, synchronize_session=False)


class VectorRepository(BaseRepository):
    def attach(self, observation_id: int, vector_type: str, values: List[float], norm: float) -> None:
        vector = models.FeatureVector(
            observation_id=observation_id,
            vector_type=vector_type,
            values=values,
            norm=norm,
        )
        self.session.add(vector)

    def collect(self, observation_ids: Sequence[int]) -> List[models.FeatureVector]:
        if not observation_ids:
            return []
        stmt = select(models.FeatureVector).where(
            models.FeatureVector.observation_id.in_(observation_ids)
        )
        return list(self.session.scalars(stmt))


class EventRepository(BaseRepository):
    def create_event(
        self,
        title: str,
        synopsis: str,
        start_at: datetime,
        end_at: datetime,
        coherence: float,
        density: float,
        prominence: float,
    ) -> models.ContextEvent:
        event = models.ContextEvent(
            title=title,
            synopsis=synopsis,
            start_at=start_at,
            end_at=end_at,
            coherence=coherence,
            density=density,
            prominence=prominence,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def attach_observations(
        self, event_id: int, observation_ids: Sequence[int], weights: Sequence[float]
    ) -> None:
        for observation_id, weight in zip(observation_ids, weights, strict=False):
            link = models.ContextEventLink(
                event_id=event_id,
                observation_id=observation_id,
                weight=weight,
            )
            self.session.add(link)

    def latest(self, limit: int = 50) -> List[models.ContextEvent]:
        stmt = (
            select(models.ContextEvent)
            .order_by(models.ContextEvent.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class RelationRepository(BaseRepository):
    def upsert(
        self,
        from_event: int,
        to_event: int,
        relation_type: str,
        similarity: float,
        causality_score: float,
    ) -> models.ContextRelation:
        stmt = select(models.ContextRelation).where(
            models.ContextRelation.from_event_id == from_event,
            models.ContextRelation.to_event_id == to_event,
            models.ContextRelation.relation_type == relation_type,
        )
        edge = self.session.scalar(stmt)
        if edge:
            edge.similarity = similarity
            edge.causality_score = causality_score
        else:
            edge = models.ContextRelation(
                from_event_id=from_event,
                to_event_id=to_event,
                relation_type=relation_type,
                similarity=similarity,
                causality_score=causality_score,
            )
            self.session.add(edge)
        self.session.flush()
        return edge

    def neighborhood(self, event_id: int, limit: int = 20) -> List[models.ContextRelation]:
        stmt = (
            select(models.ContextRelation)
            .where(
                (models.ContextRelation.from_event_id == event_id)
                | (models.ContextRelation.to_event_id == event_id)
            )
            .order_by(models.ContextRelation.similarity.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class EntityRepository(BaseRepository):
    def get_or_create(self, canonical_name: str, category: str) -> models.EntityProfile:
        stmt = select(models.EntityProfile).where(
            models.EntityProfile.canonical_name == canonical_name
        )
        entity = self.session.scalar(stmt)
        if entity:
            entity.latest_seen = datetime.utcnow()
            return entity
        entity = models.EntityProfile(
            canonical_name=canonical_name,
            category=category,
            first_seen=datetime.utcnow(),
            latest_seen=datetime.utcnow(),
        )
        self.session.add(entity)
        self.session.flush()
        return entity

    def attach(self, entity_id: int, observation_id: int, salience: float) -> None:
        link = models.EntityObservation(
            entity_id=entity_id,
            observation_id=observation_id,
            salience=salience,
        )
        self.session.add(link)

    def top_entities(self, limit: int = 20) -> List[models.EntityProfile]:
        stmt = (
            select(models.EntityProfile)
            .order_by(models.EntityProfile.prominence.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class EvidenceRepository(BaseRepository):
    def create_case(self, title: str, hypothesis: str, confidence: float) -> models.EvidenceCase:
        case = models.EvidenceCase(
            title=title,
            hypothesis=hypothesis,
            confidence=confidence,
            updated_at=datetime.utcnow(),
        )
        self.session.add(case)
        self.session.flush()
        return case

    def add_artifact(
        self,
        case_id: int,
        event_id: int,
        description: str,
        strength: float,
    ) -> None:
        artifact = models.EvidenceArtifact(
            case_id=case_id,
            event_id=event_id,
            description=description,
            strength=strength,
        )
        self.session.add(artifact)

    def latest_cases(self, limit: int = 10) -> List[models.EvidenceCase]:
        stmt = (
            select(models.EvidenceCase)
            .order_by(models.EvidenceCase.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class TimelineRepository(BaseRepository):
    def upsert_marker(self, date: datetime, summary: str, event_ids: List[int]) -> models.TimelineMarker:
        stmt = select(models.TimelineMarker).where(models.TimelineMarker.date == date.date())
        marker = self.session.scalar(stmt)
        if marker:
            marker.summary = summary
            marker.event_ids = event_ids
        else:
            marker = models.TimelineMarker(
                date=date.date(),
                summary=summary,
                event_ids=event_ids,
            )
            self.session.add(marker)
        self.session.flush()
        return marker

    def list_markers(self, limit: int = 30) -> List[models.TimelineMarker]:
        stmt = (
            select(models.TimelineMarker)
            .order_by(models.TimelineMarker.date.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class MetricRepository(BaseRepository):
    def store(self, name: str, window_start: datetime, window_end: datetime, payload: Dict) -> None:
        snapshot = models.MetricSnapshot(
            metric_name=name,
            window_start=window_start,
            window_end=window_end,
            payload=payload,
        )
        self.session.add(snapshot)

    def recent(self, name: str, limit: int = 5) -> List[models.MetricSnapshot]:
        stmt = (
            select(models.MetricSnapshot)
            .where(models.MetricSnapshot.metric_name == name)
            .order_by(models.MetricSnapshot.window_end.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class LogRepository(BaseRepository):
    def add(self, component: str, level: str, message: str, extra: Dict | None = None) -> None:
        record = models.ProcessingLog(
            component=component,
            level=level,
            message=message,
            extra=extra,
        )
        self.session.add(record)

    def latest(self, limit: int = 50) -> List[models.ProcessingLog]:
        stmt = (
            select(models.ProcessingLog)
            .order_by(models.ProcessingLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

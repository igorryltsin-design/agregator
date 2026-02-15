"""ORM models for Contextual Analytics SPPR."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

Modality = Enum(
    "audio",
    "text",
    "image",
    "radar",
    "video",
    "composite",
    name="context_modality",
)


class SourceChannel(Base):
    __tablename__ = "source_channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    modality: Mapped[str] = mapped_column(Modality, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    relevance_bias: Mapped[float] = mapped_column(Float, default=0.5)
    latency_expectation: Mapped[int] = mapped_column(Integer, default=5)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    observations: Mapped[List["Observation"]] = relationship(
        "Observation", back_populates="source"
    )


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("source_channels.id"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    modality: Mapped[str] = mapped_column(Modality, nullable=False)
    content_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    annotation: Mapped[Optional[str]] = mapped_column(Text)
    attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    intensity: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped["SourceChannel"] = relationship("SourceChannel", back_populates="observations")
    vectors: Mapped[List["FeatureVector"]] = relationship(
        "FeatureVector", back_populates="observation", cascade="all, delete-orphan"
    )
    events: Mapped[List["ContextEvent"]] = relationship(
        "ContextEventLink", back_populates="observation", cascade="all, delete-orphan"
    )


class FeatureVector(Base):
    __tablename__ = "feature_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"))
    vector_type: Mapped[str] = mapped_column(String(64), nullable=False)
    values: Mapped[List[float]] = mapped_column(JSON, nullable=False)
    norm: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    observation: Mapped["Observation"] = relationship("Observation", back_populates="vectors")


class ContextEvent(Base):
    __tablename__ = "context_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    synopsis: Mapped[str] = mapped_column(Text, nullable=False)
    start_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    coherence: Mapped[float] = mapped_column(Float, default=0.0)
    density: Mapped[float] = mapped_column(Float, default=0.0)
    prominence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    links: Mapped[List["ContextEventLink"]] = relationship(
        "ContextEventLink",
        back_populates="event",
        cascade="all, delete-orphan",
    )
    relations_from: Mapped[List["ContextRelation"]] = relationship(
        "ContextRelation",
        foreign_keys="ContextRelation.from_event_id",
        back_populates="from_event",
    )
    relations_to: Mapped[List["ContextRelation"]] = relationship(
        "ContextRelation",
        foreign_keys="ContextRelation.to_event_id",
        back_populates="to_event",
    )


class ContextEventLink(Base):
    __tablename__ = "context_event_links"
    __table_args__ = (
        UniqueConstraint("event_id", "observation_id", name="uix_event_observation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("context_events.id"))
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"))
    weight: Mapped[float] = mapped_column(Float, default=0.0)

    event: Mapped["ContextEvent"] = relationship("ContextEvent", back_populates="links")
    observation: Mapped["Observation"] = relationship("Observation", back_populates="events")


class ContextRelation(Base):
    __tablename__ = "context_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_event_id: Mapped[int] = mapped_column(ForeignKey("context_events.id"))
    to_event_id: Mapped[int] = mapped_column(ForeignKey("context_events.id"))
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    causality_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    from_event: Mapped["ContextEvent"] = relationship(
        "ContextEvent", foreign_keys=[from_event_id], back_populates="relations_from"
    )
    to_event: Mapped["ContextEvent"] = relationship(
        "ContextEvent", foreign_keys=[to_event_id], back_populates="relations_to"
    )


class EntityProfile(Base):
    __tablename__ = "entity_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    latest_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    prominence: Mapped[float] = mapped_column(Float, default=0.0)
    stability: Mapped[float] = mapped_column(Float, default=0.0)
    extra: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class EntityObservation(Base):
    __tablename__ = "entity_observations"
    __table_args__ = (
        UniqueConstraint("entity_id", "observation_id", name="uix_entity_observation"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity_profiles.id"))
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"))
    salience: Mapped[float] = mapped_column(Float, default=0.0)

    entity: Mapped["EntityProfile"] = relationship("EntityProfile")
    observation: Mapped["Observation"] = relationship("Observation")


class EvidenceCase(Base):
    __tablename__ = "evidence_cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String(32), default="draft")

    artifacts: Mapped[List["EvidenceArtifact"]] = relationship(
        "EvidenceArtifact",
        back_populates="case",
        cascade="all, delete-orphan",
    )


class EvidenceArtifact(Base):
    __tablename__ = "evidence_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("evidence_cases.id"))
    event_id: Mapped[int] = mapped_column(ForeignKey("context_events.id"))
    description: Mapped[str] = mapped_column(Text, nullable=False)
    strength: Mapped[float] = mapped_column(Float, default=0.0)

    case: Mapped["EvidenceCase"] = relationship("EvidenceCase", back_populates="artifacts")
    event: Mapped["ContextEvent"] = relationship("ContextEvent")


class TimelineMarker(Base):
    __tablename__ = "timeline_markers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    date: Mapped[Date] = mapped_column(Date, nullable=False, unique=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    event_ids: Mapped[List[int]] = mapped_column(JSON, default=list)


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

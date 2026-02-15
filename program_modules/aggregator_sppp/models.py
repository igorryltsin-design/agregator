"""ORM models for the Aggregator SPPR module."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
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
    "visual",
    "radar",
    "composite",
    name="modality",
)


class DataSource(Base):
    __tablename__ = "data_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    modality: Mapped[str] = mapped_column(Modality, nullable=False)
    reliability_baseline: Mapped[float] = mapped_column(Float, default=0.5)
    priority: Mapped[int] = mapped_column(Integer, default=5)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    signals: Mapped[List["Signal"]] = relationship("Signal", back_populates="source")


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("data_sources.id"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    signal_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_ref: Mapped[str] = mapped_column(String(512), nullable=False)
    vector: Mapped[List[float] | None] = mapped_column(JSON)
    attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    intensity: Mapped[float] = mapped_column(Float, default=0.0)
    reliability: Mapped[float] = mapped_column(Float, default=0.0)
    normalized: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped["DataSource"] = relationship("DataSource", back_populates="signals")
    fusion_links: Mapped[List["FusionLink"]] = relationship(
        "FusionLink", back_populates="signal", cascade="all, delete-orphan"
    )
    outgoing_edges: Mapped[List["RelationEdge"]] = relationship(
        "RelationEdge",
        foreign_keys="RelationEdge.from_signal_id",
        back_populates="from_signal",
    )
    incoming_edges: Mapped[List["RelationEdge"]] = relationship(
        "RelationEdge",
        foreign_keys="RelationEdge.to_signal_id",
        back_populates="to_signal",
    )


class FusionSample(Base):
    __tablename__ = "fusion_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    start_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    context_score: Mapped[float] = mapped_column(Float, default=0.0)
    coherence: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    links: Mapped[List["FusionLink"]] = relationship(
        "FusionLink", back_populates="sample", cascade="all, delete-orphan"
    )


class FusionLink(Base):
    __tablename__ = "fusion_links"
    __table_args__ = (
        UniqueConstraint("sample_id", "signal_id", name="uix_sample_signal"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey("fusion_samples.id"))
    signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"))
    weight: Mapped[float] = mapped_column(Float, default=0.0)

    sample: Mapped["FusionSample"] = relationship("FusionSample", back_populates="links")
    signal: Mapped["Signal"] = relationship("Signal", back_populates="fusion_links")


class RelationEdge(Base):
    __tablename__ = "relation_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"))
    to_signal_id: Mapped[int] = mapped_column(ForeignKey("signals.id"))
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    similarity: Mapped[float] = mapped_column(Float, default=0.0)
    weight: Mapped[float] = mapped_column(Float, default=0.0)
    window_start: Mapped[datetime] = mapped_column(DateTime)
    window_end: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    from_signal: Mapped["Signal"] = relationship(
        "Signal",
        foreign_keys=[from_signal_id],
        back_populates="outgoing_edges",
    )
    to_signal: Mapped["Signal"] = relationship(
        "Signal",
        foreign_keys=[to_signal_id],
        back_populates="incoming_edges",
    )


class HistoryEvent(Base):
    __tablename__ = "history_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    replay_token: Mapped[str] = mapped_column(String(64), nullable=False)


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    component: Mapped[str] = mapped_column(String(64), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReplayCursor(Base):
    __tablename__ = "replay_cursors"
    __table_args__ = (
        UniqueConstraint("name", name="uix_replay_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    last_event_id: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

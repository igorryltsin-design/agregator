"""ORM models describing integration topology and telemetry."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
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

ModuleKind = Enum(
    "ingestion",
    "analytics",
    "dialog",
    "storage",
    "external",
    name="module_kind",
)


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(ModuleKind, nullable=False)
    api_url: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="unknown")
    priority: Mapped[int] = mapped_column(Integer, default=3)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    outgoing_edges: Mapped[List["Edge"]] = relationship(
        "Edge", foreign_keys="Edge.from_node_id", back_populates="from_node"
    )
    incoming_edges: Mapped[List["Edge"]] = relationship(
        "Edge", foreign_keys="Edge.to_node_id", back_populates="to_node"
    )


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (UniqueConstraint("from_node_id", "to_node_id", name="uix_edge"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id"))
    to_node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id"))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_bandwidth: Mapped[float] = mapped_column(Float, default=100.0)
    current_bandwidth: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    packet_loss: Mapped[float] = mapped_column(Float, default=0.0)
    reliability: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    from_node: Mapped["Node"] = relationship("Node", foreign_keys=[from_node_id], back_populates="outgoing_edges")
    to_node: Mapped["Node"] = relationship("Node", foreign_keys=[to_node_id], back_populates="incoming_edges")
    channels: Mapped[List["Channel"]] = relationship(
        "Channel", back_populates="edge", cascade="all, delete-orphan"
    )


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    edge_id: Mapped[int] = mapped_column(ForeignKey("edges.id"))
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    modality: Mapped[str] = mapped_column(String(32), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=3)
    target_rate: Mapped[float] = mapped_column(Float, default=10.0)
    actual_rate: Mapped[float] = mapped_column(Float, default=0.0)
    dropped_packets: Mapped[int] = mapped_column(Integer, default=0)
    retries: Mapped[int] = mapped_column(Integer, default=0)
    attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    edge: Mapped["Edge"] = relationship("Edge", back_populates="channels")
    flows: Mapped[List["FlowStat"]] = relationship(
        "FlowStat", back_populates="channel", cascade="all, delete-orphan"
    )


class FlowStat(Base):
    __tablename__ = "flow_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id"))
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    avg_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    reliability: Mapped[float] = mapped_column(Float, default=1.0)
    pressure: Mapped[float] = mapped_column(Float, default=0.0)
    snapshot: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

    channel: Mapped["Channel"] = relationship("Channel", back_populates="flows")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ReportSnapshot(Base):
    __tablename__ = "report_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)


class Heartbeat(Base):
    __tablename__ = "heartbeats"
    __table_args__ = (UniqueConstraint("node_id", name="uix_heartbeat_node"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    node_id: Mapped[int] = mapped_column(ForeignKey("nodes.id"))
    status: Mapped[str] = mapped_column(String(32), default="unknown")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    node: Mapped["Node"] = relationship("Node")

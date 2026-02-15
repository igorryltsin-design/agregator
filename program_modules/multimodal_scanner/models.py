"""ORM models describing scanned objects and similarity graph."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

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

Modality = Enum(
    "text",
    "audio",
    "image",
    "video",
    "archive",
    "unknown",
    name="scanner_modality",
)


class ScanSource(Base):
    __tablename__ = "scan_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    root: Mapped[str] = mapped_column(String(512), nullable=False)
    recursive: Mapped[bool] = mapped_column(Boolean, default=True)
    include_patterns: Mapped[List[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    artifacts: Mapped[List["Artifact"]] = relationship(
        "Artifact", back_populates="source", cascade="all, delete-orphan"
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("scan_sources.id"))
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    modality: Mapped[str] = mapped_column(Modality, default="unknown")
    signature: Mapped[str] = mapped_column(String(128), nullable=False)
    hash_alg: Mapped[str] = mapped_column(String(16), default="sha256")
    entropy: Mapped[float] = mapped_column(Float, default=0.0)
    attributes: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    source: Mapped["ScanSource"] = relationship("ScanSource", back_populates="artifacts")
    signatures: Mapped[List["SignatureBlock"]] = relationship(
        "SignatureBlock", back_populates="artifact", cascade="all, delete-orphan"
    )
    edges_from: Mapped[List["SimilarityEdge"]] = relationship(
        "SimilarityEdge",
        foreign_keys="SimilarityEdge.from_artifact_id",
        back_populates="from_artifact",
    )
    edges_to: Mapped[List["SimilarityEdge"]] = relationship(
        "SimilarityEdge",
        foreign_keys="SimilarityEdge.to_artifact_id",
        back_populates="to_artifact",
    )


class SignatureBlock(Base):
    __tablename__ = "signature_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    artifact_id: Mapped[int] = mapped_column(ForeignKey("artifacts.id"))
    offset: Mapped[int] = mapped_column(Integer, nullable=False)
    signature: Mapped[str] = mapped_column(String(128), nullable=False)
    block_size: Mapped[int] = mapped_column(Integer, nullable=False)

    artifact: Mapped["Artifact"] = relationship("Artifact", back_populates="signatures")


class SimilarityEdge(Base):
    __tablename__ = "similarity_edges"
    __table_args__ = (
        UniqueConstraint("from_artifact_id", "to_artifact_id", name="uix_similarity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    from_artifact_id: Mapped[int] = mapped_column(ForeignKey("artifacts.id"))
    to_artifact_id: Mapped[int] = mapped_column(ForeignKey("artifacts.id"))
    score: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    from_artifact: Mapped["Artifact"] = relationship(
        "Artifact", foreign_keys=[from_artifact_id], back_populates="edges_from"
    )
    to_artifact: Mapped["Artifact"] = relationship(
        "Artifact", foreign_keys=[to_artifact_id], back_populates="edges_to"
    )


class ScanRun(Base):
    __tablename__ = "scan_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed_files: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    memo: Mapped[str | None] = mapped_column(Text)


class ScanLog(Base):
    __tablename__ = "scan_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("scan_runs.id"))
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ScanStat(Base):
    __tablename__ = "scan_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)

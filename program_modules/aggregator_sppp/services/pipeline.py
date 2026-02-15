"""High level pipeline orchestrating ingestion, fusion and graph updates."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict

from .. import models
from ..config import AppConfig
from ..database import Database
from ..repositories import (
    DataSourceRepository,
    FusionRepository,
    LogRepository,
    SignalRepository,
)
from .fusion import FusionEngine
from .graph import GraphBuilder
from .history import HistoryService
from .ingestion import StreamIngestor
from .validators import FusionWindowPayload, SignalPayload, SourcePayload, ValidationError

LOGGER = logging.getLogger("aggregator.pipeline")


@dataclass
class PipelineResult:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None


class AggregationPipeline:
    def __init__(self, config: AppConfig, database: Database, telemetry) -> None:
        self.config = config
        self.database = database
        self.telemetry = telemetry
        self.ingestion = StreamIngestor(config, telemetry)
        self.fusion = FusionEngine(config, telemetry)
        self.graph = GraphBuilder(config, telemetry)
        self.history = HistoryService(config)

    @contextmanager
    def _session(self):
        with self.database.session() as session:
            yield session

    def register_source(self, payload: dict) -> PipelineResult:
        try:
            source = SourcePayload.from_dict(payload)
        except ValidationError as exc:
            return PipelineResult(ok=False, payload={}, error=str(exc))
        with self._session() as session:
            result = self.ingestion.register_source(session, source)
            self.history.append(
                session,
                "source_registered",
                {"name": source.name, "created": result["created"]},
            )
            return PipelineResult(ok=True, payload=result)

    def ingest_signal(self, payload: dict) -> PipelineResult:
        try:
            signal = SignalPayload.from_dict(payload)
        except ValidationError as exc:
            return PipelineResult(ok=False, payload={}, error=str(exc))
        with self._session() as session:
            try:
                result = self.ingestion.ingest_signal(session, signal)
            except ValidationError as exc:
                return PipelineResult(ok=False, payload={}, error=str(exc))
            if not result.get("duplicate"):
                self.history.append(
                    session,
                    "signal_ingested",
                    {
                        "signal_id": result["id"],
                        "source_name": signal.source_name,
                    },
                )
            return PipelineResult(ok=True, payload=result)

    def build_fusion(self, payload: dict) -> PipelineResult:
        window = FusionWindowPayload.from_dict(payload)
        with self._session() as session:
            result = self.fusion.fuse_window(session, window)
            repo = SignalRepository(session)
            for sample in result["samples"]:
                signals = [
                    self._load_signal(repo, signal_id)
                    for signal_id in sample["signal_ids"]
                ]
                signals = [signal for signal in signals if signal]
                if signals:
                    self.graph.update_graph(session, signals)
            if result["samples"]:
                self.history.append(
                    session,
                    "fusion_completed",
                    {
                        "sample_ids": [sample["sample_id"] for sample in result["samples"]],
                        "count": len(result["samples"]),
                    },
                )
            return PipelineResult(ok=True, payload=result)

    def list_sources(self) -> PipelineResult:
        with self._session() as session:
            repo = DataSourceRepository(session)
            sources = repo.list_all()
            return PipelineResult(
                ok=True,
                payload={
                    "sources": [
                        {
                            "id": source.id,
                            "name": source.name,
                            "modality": source.modality,
                            "reliability_baseline": source.reliability_baseline,
                            "priority": source.priority,
                        }
                        for source in sources
                    ]
                },
            )

    def latest_signals(self, limit: int = 50) -> PipelineResult:
        with self._session() as session:
            repo = SignalRepository(session)
            items = repo.latest_signals(limit=limit)
            return PipelineResult(
                ok=True,
                payload={
                    "signals": [
                        {
                            "id": signal.id,
                            "source": signal.source.name,
                            "modality": signal.source.modality,
                            "intensity": signal.intensity,
                            "reliability": signal.reliability,
                            "recorded_at": signal.recorded_at.isoformat(),
                        }
                        for signal in items
                    ]
                },
            )

    def latest_samples(self, limit: int = 20) -> PipelineResult:
        with self._session() as session:
            repo = FusionRepository(session)
            samples = repo.latest_samples(limit=limit)
            return PipelineResult(
                ok=True,
                payload={
                    "samples": [
                        {
                            "id": sample.id,
                            "summary": sample.summary,
                            "context_score": sample.context_score,
                            "coherence": sample.coherence,
                        }
                        for sample in samples
                    ]
                },
            )

    def history_window(self, cursor: str, limit: int = 50) -> PipelineResult:
        with self._session() as session:
            events = self.history.replay(session, cursor_name=cursor, limit=limit)
            return PipelineResult(ok=True, payload={"events": events})

    def stats(self) -> PipelineResult:
        with self._session() as session:
            signal_repo = SignalRepository(session)
            log_repo = LogRepository(session)
            signals = signal_repo.latest_signals(limit=10)
            logs = log_repo.latest(limit=10)
            return PipelineResult(
                ok=True,
                payload={
                    "recent_signals": [
                        {
                            "id": signal.id,
                            "source": signal.source.name,
                            "type": signal.signal_type,
                            "recorded_at": signal.recorded_at.isoformat(),
                        }
                        for signal in signals
                    ],
                    "logs": [
                        {
                            "component": log.component,
                            "level": log.level,
                            "message": log.message,
                            "created_at": log.created_at.isoformat(),
                        }
                        for log in logs
                    ],
                },
            )

    def _load_signal(self, repo: SignalRepository, signal_id: int):
        return repo.session.get(models.Signal, signal_id)

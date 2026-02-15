"""High-level application service for Contextual Analytics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List

from .config import AppConfig
from .database import Database
from .repositories import EvidenceRepository, LogRepository, ObservationRepository
from .services.analysis import AnalysisService
from .services.evidence import EvidenceService
from .services.ingestion import IngestionService
from .services.timeline import TimelineService
from .vectorizers import Vectorizer


@dataclass
class EngineResult:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None


class ContextEngine:
    def __init__(self, config: AppConfig, database: Database):
        self.config = config
        self.database = database
        self.vectorizer = Vectorizer(config.vectorizer)
        self.ingestion = IngestionService(config, self.vectorizer)
        self.analysis = AnalysisService(config, self.vectorizer)
        self.timeline = TimelineService(config)
        self.evidence = EvidenceService(config)

    @contextmanager
    def _session(self):
        with self.database.session() as session:
            yield session

    def register_source(self, payload: dict) -> EngineResult:
        required = {"name", "modality"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        with self._session() as session:
            result = self.ingestion.register_source(
                session=session,
                name=payload["name"],
                modality=payload["modality"],
                description=payload.get("description"),
                relevance_bias=float(payload.get("relevance_bias", 0.5)),
                latency_expectation=int(payload.get("latency_expectation", 5)),
            )
            return EngineResult(True, result)

    def ingest_observation(self, payload: dict) -> EngineResult:
        required = {"source", "modality", "content_ref"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        recorded = payload.get("recorded_at")
        recorded_at = datetime.fromisoformat(recorded) if recorded else datetime.utcnow()
        with self._session() as session:
            result = self.ingestion.ingest_observation(
                session=session,
                source_name=payload["source"],
                modality=payload["modality"],
                content_ref=payload["content_ref"],
                recorded_at=recorded_at,
                annotation=payload.get("annotation"),
                attributes=payload.get("attributes", {}),
                intensity=float(payload.get("intensity", 0.5)),
                confidence=float(payload.get("confidence", 0.5)),
            )
            return EngineResult(True, result)

    def analyse(self, limit: int = 64) -> EngineResult:
        with self._session() as session:
            summaries = self.analysis.run_window(session, limit=limit)
            markers = self.timeline.refresh(session)
            cases = self.evidence.refresh_cases(session)
            return EngineResult(
                True,
                {
                    "events": [summary.__dict__ for summary in summaries],
                    "timeline": markers,
                    "evidence": cases,
                },
            )

    def stats(self) -> EngineResult:
        with self._session() as session:
            log_repo = LogRepository(session)
            observation_repo = ObservationRepository(session)
            pending = observation_repo.fetch_unprocessed(limit=10)
            logs = log_repo.latest(limit=10)
            evidence_repo = EvidenceRepository(session)
            cases = evidence_repo.latest_cases(limit=5)
            return EngineResult(
                True,
                {
                    "pending_observations": [
                        {
                            "id": obs.id,
                            "source": obs.source.name if obs.source else None,
                            "modality": obs.modality,
                            "recorded_at": obs.recorded_at.isoformat(),
                        }
                        for obs in pending
                    ],
                    "logs": [
                        {
                            "component": record.component,
                            "level": record.level,
                            "message": record.message,
                            "created_at": record.created_at.isoformat(),
                        }
                        for record in logs
                    ],
                    "evidence": [
                        {
                            "case_id": case.id,
                            "title": case.title,
                            "confidence": case.confidence,
                        }
                        for case in cases
                    ],
                },
            )

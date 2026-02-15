"""High-level orchestrator for the multimodal scanner."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List

from sqlalchemy import select

from . import models
from .config import AppConfig
from .database import Database
from .repositories import ArtifactRepository, LogRepository, RunRepository
from .services.classifier import ModalityClassifier
from .services.graph import SimilarityGraph
from .services.statistics import StatisticsService
from .services.walker import ScanWalker
from .services.indexer import KeywordIndex
from .services.scheduler import Scheduler
from .services.reporting import ReportingService
from .services.watchdog import WatchdogService
from .telemetry import Telemetry


@dataclass
class EngineResult:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None


class ScannerEngine:
    def __init__(self, config: AppConfig, database: Database):
        self.config = config
        self.database = database
        self.walker = ScanWalker(config)
        self.graph = SimilarityGraph(config)
        self.classifier = ModalityClassifier(config)
        self.stats = StatisticsService(config)
        self.reporting = ReportingService(config)
        self.index = KeywordIndex()
        self.watchdog_service = WatchdogService(config)
        self.telemetry = Telemetry()
        self.scheduler = Scheduler()
        self._last_telemetry: Dict[str, Dict[str, float]] = {}
        self.telemetry.configure_export(
            interval=config.stats_window.total_seconds(),
            callback=self._capture_telemetry,
        )
        self.telemetry.start()
        self.scheduler.add_task("stats", config.stats_window.total_seconds(), self._scheduled_stats)
        self.scheduler.start()

    @contextmanager
    def _session(self):
        with self.database.session() as session:
            yield session

    def scan(self, memo: str | None = None) -> EngineResult:
        with self._session() as session:
            result = self.walker.run(session, memo=memo)
            self.telemetry.observe("scan.total", float(result["total"]))
            self.telemetry.observe("scan.processed", float(result["processed"]))
            self.index.build(session)
            return EngineResult(True, result)

    def classify(self, artifact_id: int) -> EngineResult:
        with self._session() as session:
            artifact = session.get(models.Artifact, artifact_id)
            if not artifact:
                return EngineResult(False, {}, "artifact_not_found")
            result = self.classifier.classify(artifact.modality, artifact.entropy, artifact.attributes)
            artifact.attributes = artifact.attributes | {"classification": result}
            self.telemetry.observe("classification.score", result["score"])
            return EngineResult(True, {"artifact_id": artifact_id, "classification": result})

    def build_graph(self, artifact_ids: List[int]) -> EngineResult:
        with self._session() as session:
            result = self.graph.build(session, artifact_ids)
            return EngineResult(True, result)

    def stats_snapshot(self) -> EngineResult:
        with self._session() as session:
            data = self.stats.refresh(session)
            return EngineResult(True, {"modality": data})

    def latest_artifacts(self, limit: int = 20) -> EngineResult:
        with self._session() as session:
            repo = ArtifactRepository(session)
            artifacts = repo.list_recent(limit=limit)
            return EngineResult(
                True,
                {
                    "artifacts": [
                        {
                            "id": artifact.id,
                            "path": artifact.path,
                            "modality": artifact.modality,
                            "updated_at": artifact.updated_at.isoformat(),
                        }
                        for artifact in artifacts
                    ]
                },
            )

    def logs(self, limit: int = 50) -> EngineResult:
        with self._session() as session:
            repo = LogRepository(session)
            entries = repo.latest(limit=limit)
            return EngineResult(
                True,
                {
                    "logs": [
                        {
                            "level": entry.level,
                            "message": entry.message,
                            "created_at": entry.created_at.isoformat(),
                            "payload": entry.payload,
                        }
                        for entry in entries
                    ]
                },
            )

    def report(self) -> EngineResult:
        with self._session() as session:
            report = self.reporting.generate(session, self._last_telemetry)
            return EngineResult(True, report)

    def search(self, query: str) -> EngineResult:
        results = self.index.search(query)
        return EngineResult(True, {"query": query, "artifact_ids": results})

    def watchdog(self) -> EngineResult:
        with self._session() as session:
            summary = self.watchdog_service.inspect(session)
            return EngineResult(
                True,
                {
                    "errors": summary.error_entries,
                    "high_entropy": summary.high_entropy,
                    "duplicates": summary.duplicates,
                },
            )

    def shutdown(self) -> None:
        self.scheduler.stop()
        self.telemetry.stop()

    def _capture_telemetry(self, payload: Dict[str, Dict[str, float]]) -> None:
        self._last_telemetry = payload

    def _scheduled_stats(self) -> None:
        with self._session() as session:
            self.stats.refresh(session)

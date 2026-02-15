"""Integrator engine coordinating topology, streams, monitoring and reporting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict

from .config import AppConfig
from .database import Database
from .repositories import AuditRepository, ReportRepository
from .services.monitor import MonitoringService
from .services.reporting import ReportingService
from .services.streams import StreamService
from .services.topology import TopologyService
from .services.scheduler import SchedulerService
from .services.diagnostics import DiagnosticService


@dataclass
class EngineResult:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None


class IntegratorEngine:
    def __init__(self, config: AppConfig, database: Database):
        self.config = config
        self.database = database
        self.topology = TopologyService(config)
        self.streams = StreamService(config)
        self.monitor = MonitoringService(config)
        self.reporting = ReportingService(config)
        self.diagnostics = DiagnosticService()
        self.scheduler = SchedulerService()
        self._configure_scheduler()

    def _configure_scheduler(self) -> None:
        interval = self.config.reporting.snapshot_interval.total_seconds()
        self.scheduler.add_task("reporting", interval, self._scheduled_snapshot)
        self.scheduler.start()

    def shutdown(self) -> None:
        self.scheduler.stop()

    @contextmanager
    def _session(self):
        with self.database.session() as session:
            yield session

    def register_node(self, payload: dict) -> EngineResult:
        required = {"name", "kind", "api_url"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        with self._session() as session:
            priority = (
                int(payload["priority"])
                if payload.get("priority") is not None
                else self.config.stream.default_priority
            )
            result = self.topology.register_node(
                session,
                payload["name"],
                payload["kind"],
                payload["api_url"],
                priority=priority,
            )
            return EngineResult(True, result)

    def connect_nodes(self, payload: dict) -> EngineResult:
        required = {"from", "to"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        with self._session() as session:
            result = self.topology.connect(
                session,
                payload["from"],
                payload["to"],
                max_bandwidth=float(payload.get("max_bandwidth", self.config.stream.max_bandwidth / 10)),
            )
            return EngineResult(True, result)

    def heartbeat(self, node: str) -> EngineResult:
        with self._session() as session:
            result = self.topology.heartbeat(session, node)
            return EngineResult(True, result)

    def add_channel(self, payload: dict) -> EngineResult:
        required = {"edge_id", "label", "modality"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        with self._session() as session:
            result = self.streams.add_channel(
                session,
                edge_id=int(payload["edge_id"]),
                label=payload["label"],
                modality=payload["modality"],
                priority=int(payload["priority"]) if payload.get("priority") is not None else None,
                target_rate=float(payload["target_rate"]) if payload.get("target_rate") is not None else None,
            )
            return EngineResult(True, result)

    def channel_metrics(self, payload: dict) -> EngineResult:
        required = {"channel_id", "actual_rate", "dropped", "retries"}
        missing = required - payload.keys()
        if missing:
            return EngineResult(False, {}, f"Отсутствуют поля: {', '.join(sorted(missing))}")
        with self._session() as session:
            result = self.streams.update_channel_metrics(
                session,
                channel_id=int(payload["channel_id"]),
                actual_rate=float(payload["actual_rate"]),
                dropped=int(payload["dropped"]),
                retries=int(payload["retries"]),
            )
            ok = "error" not in result
            return EngineResult(ok, result if ok else {}, None if ok else "channel_not_found")

    def snapshot(self) -> EngineResult:
        with self._session() as session:
            telemetry = self.monitor.evaluate(session)
            breaches = self.monitor.detect_breaches(session)
            report = self.reporting.generate(session, telemetry, breaches)
            nodes = [
                status.__dict__
                for status in self.diagnostics.snapshot(session)
            ]
            return EngineResult(True, {"telemetry": telemetry, "breaches": breaches, "report": report, "nodes": nodes})

    def audit_log(self) -> EngineResult:
        with self._session() as session:
            repo = AuditRepository(session)
            records = repo.latest(limit=50)
            return EngineResult(
                True,
                {
                    "logs": [
                        {
                            "type": record.event_type,
                            "level": record.level,
                            "message": record.message,
                            "created_at": record.created_at.isoformat(),
                        }
                        for record in records
                    ]
                },
            )

    def reports(self) -> EngineResult:
        with self._session() as session:
            repo = ReportRepository(session)
            snapshots = repo.list_reports(limit=10)
            return EngineResult(
                True,
                {
                    "reports": [
                        {
                            "title": snapshot.title,
                            "generated_at": snapshot.generated_at.isoformat(),
                            "metrics": snapshot.metrics,
                        }
                        for snapshot in snapshots
                    ]
                },
            )

    def _scheduled_snapshot(self) -> None:
        with self._session() as session:
            telemetry = self.monitor.evaluate(session)
            breaches = self.monitor.detect_breaches(session)
            self.reporting.generate(session, telemetry, breaches)

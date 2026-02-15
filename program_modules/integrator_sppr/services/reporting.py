"""Reporting utilities for Integrator SPPR."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import AuditRepository, ReportRepository


class ReportingService:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(self, session: Session, telemetry: Dict[str, float], breaches: List[dict]) -> dict:
        repo = ReportRepository(session)
        title = f"Отчёт интегратора {datetime.utcnow():%Y-%m-%d %H:%M}"
        body = self._format_body(telemetry, breaches)
        report = repo.store(title, body, metrics=telemetry)
        cutoff = datetime.utcnow() - self.config.reporting.retention
        repo.purge(cutoff)
        return {"report_id": report.id, "title": report.title}

    def _format_body(self, telemetry: Dict[str, float], breaches: List[dict]) -> str:
        lines = [
            "Сводка потоков:",
            f"- Объём данных: {telemetry['volume']:.2f} МБ",
            f"- Средняя задержка: {telemetry['latency']:.2f} мс",
            f"- Надёжность: {telemetry['reliability']:.2f}",
            "",
            "Выявленные события:",
        ]
        if not breaches:
            lines.append("- Отклонения не зафиксированы")
        else:
            for breach in breaches:
                line = f"- Канал {breach.get('channel_id')} "
                if "reliability" in breach:
                    line += f"надёжность {breach['reliability']:.2f}"
                if "latency_ms" in breach:
                    line += f" задержка {breach['latency_ms']:.2f} мс"
                lines.append(line)
        return "\n".join(lines)

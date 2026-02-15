"""Report generation service."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import LogRepository, StatRepository


class ReportingService:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(self, session: Session, telemetry: Dict[str, Dict[str, float]]) -> Dict:
        stat_repo = StatRepository(session)
        stats = stat_repo.recent("modality", limit=1)
        latest = stats[0].payload if stats else {}
        body_lines = [
            f"Отчёт сканера {datetime.utcnow():%Y-%m-%d %H:%M}",
            "",
            "Телеметрия:",
        ]
        for name, values in telemetry.items():
            body_lines.append(f"- {name}: avg={values['average']:.2f} latest={values['latest']:.2f}")
        body_lines.append("")
        body_lines.append("Распределение по модальностям:")
        if not latest:
            body_lines.append("- данных нет")
        else:
            for modality, count in latest.items():
                body_lines.append(f"- {modality}: {count}")
        return {"title": body_lines[0], "body": "\n".join(body_lines)}

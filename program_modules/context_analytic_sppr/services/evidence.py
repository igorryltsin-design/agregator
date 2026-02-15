"""Evidence synthesis for decision support."""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import EvidenceRepository, EventRepository


class EvidenceService:
    def __init__(self, config: AppConfig):
        self.config = config

    def refresh_cases(self, session: Session) -> List[dict]:
        event_repo = EventRepository(session)
        evidence_repo = EvidenceRepository(session)

        events = event_repo.latest(limit=10)
        if not events:
            return []
        cases: List[dict] = []
        for event in events:
            hypothesis = f"Влияние события «{event.title}» на текущий контур решений."
            confidence = min(1.0, event.coherence * 0.4 + event.prominence * 0.6)
            if confidence < self.config.evidence.confidence_floor:
                continue
            case = evidence_repo.create_case(event.title, hypothesis, confidence)
            narrative = self._narrative(event)
            evidence_repo.add_artifact(
                case_id=case.id,
                event_id=event.id,
                description=narrative,
                strength=confidence,
            )
            cases.append(
                {
                    "case_id": case.id,
                    "event_id": event.id,
                    "confidence": confidence,
                    "narrative": narrative,
                }
            )
        return cases

    def _narrative(self, event) -> str:
        base = [
            f"Сигналы между {event.start_at:%H:%M} и {event.end_at:%H:%M} сформировали кластер.",
            f"Плотность наблюдений: {event.density:.2f}.",
            f"Когерентность: {event.coherence:.2f}, значимость: {event.prominence:.2f}.",
        ]
        text = " ".join(base)
        return text[: self.config.evidence.narrative_max_len]

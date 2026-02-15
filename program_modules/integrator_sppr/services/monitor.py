"""Monitoring service: computes telemetry, detects breaches."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import AppConfig
from ..repositories import AuditRepository, bandwidth_summary

LOGGER = logging.getLogger("integrator.monitor")


class MonitoringService:
    def __init__(self, config: AppConfig):
        self.config = config

    def evaluate(self, session: Session) -> Dict[str, float]:
        summary = bandwidth_summary(session, self.config.reporting.snapshot_interval)
        LOGGER.info(
            "Telemetry: volume=%.2f latency=%.2f reliability=%.2f",
            summary["volume"],
            summary["latency"],
            summary["reliability"],
        )
        return summary

    def detect_breaches(self, session: Session) -> List[dict]:
        flows = session.scalars(
            select(models.FlowStat).order_by(models.FlowStat.window_end.desc()).limit(50)
        ).all()
        breaches: List[dict] = []
        audit = AuditRepository(session)
        for flow in flows:
            if flow.reliability < self.config.reliability.breach_threshold:
                payload = {
                    "channel_id": flow.channel_id,
                    "reliability": flow.reliability,
                    "window_end": flow.window_end.isoformat(),
                }
                audit.add("reliability_breach", "warning", "Reliability below threshold", payload)
                breaches.append(payload)
            if flow.avg_latency_ms > self.config.reliability.latency_budget_ms:
                payload = {
                    "channel_id": flow.channel_id,
                    "latency_ms": flow.avg_latency_ms,
                    "window_end": flow.window_end.isoformat(),
                }
                audit.add("latency_breach", "warning", "Latency above budget", payload)
                breaches.append(payload)
        return breaches

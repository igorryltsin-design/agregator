"""Stream orchestration: balancing channels, recording flow stats."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models
from ..config import AppConfig
from ..repositories import ChannelRepository, EdgeRepository, FlowRepository
from .. import utils

LOGGER = logging.getLogger("integrator.streams")


class StreamService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.rate_history: Dict[int, List[float]] = {}

    def add_channel(
        self,
        session: Session,
        edge_id: int,
        label: str,
        modality: str,
        priority: int | None = None,
        target_rate: float | None = None,
    ) -> dict:
        repo = ChannelRepository(session)
        channel = repo.add(
            edge_id=edge_id,
            label=label,
            modality=modality,
            priority=priority or self.config.stream.default_priority,
            target_rate=target_rate or 10.0,
        )
        LOGGER.info("Channel %s attached to edge %s", label, edge_id)
        return {"channel_id": channel.id, "label": channel.label}

    def update_channel_metrics(
        self,
        session: Session,
        channel_id: int,
        actual_rate: float,
        dropped: int,
        retries: int,
    ) -> dict:
        channel = session.get(models.Channel, channel_id)
        if not channel:
            return {"error": "channel_not_found"}
        backpressure = utils.clamp(actual_rate / max(channel.target_rate, 1.0))
        reliability = utils.reliability_index(channel.edge.latency_ms, channel.edge.packet_loss, retries)
        new_priority = utils.adaptive_priority(channel.priority, reliability, backpressure)
        channel.priority = new_priority
        repo = ChannelRepository(session)
        repo.update_stats(
            channel_id=channel_id,
            actual_rate=actual_rate,
            dropped=dropped,
            retries=retries,
            metadata={
                "reliability": reliability,
                "backpressure": backpressure,
            },
        )
        LOGGER.debug(
            "Channel %s metrics updated: rate=%s reliability=%s priority=%s",
            channel.label,
            actual_rate,
            reliability,
            new_priority,
        )
        return {
            "channel_id": channel_id,
            "priority": new_priority,
            "reliability": reliability,
            "backpressure": backpressure,
        }

    def record_flow(
        self,
        session: Session,
        channel_id: int,
        window: timedelta,
        volume: float,
        latency_ms: float,
        reliability: float,
        pressure: float,
    ) -> dict:
        repo = FlowRepository(session)
        now = datetime.utcnow()
        stat = repo.add_snapshot(
            channel_id=channel_id,
            window_start=now - window,
            window_end=now,
            volume=volume,
            avg_latency_ms=latency_ms,
            reliability=reliability,
            pressure=pressure,
            snapshot={
                "bandwidth": volume / max(window.total_seconds(), 1.0),
                "timestamp": utils.timestamp(),
            },
        )
        LOGGER.debug("Flow snapshot stored for channel %s", channel_id)
        return {"flow_id": stat.id}

    def balance_edges(self, session: Session) -> List[dict]:
        edges = session.scalars(select(models.Edge)).all()
        adjustments: List[dict] = []
        for edge in edges:
            utilization = edge.current_bandwidth / max(edge.max_bandwidth, 1.0)
            if utilization > 0.95:
                edge.max_bandwidth *= 1.1
                action = "increase"
            elif utilization < 0.4:
                edge.max_bandwidth *= 0.9
                action = "decrease"
            else:
                action = "keep"
            adjustments.append(
                {
                    "edge_id": edge.id,
                    "from": edge.from_node.name if edge.from_node else None,
                    "to": edge.to_node.name if edge.to_node else None,
                    "action": action,
                    "max_bandwidth": edge.max_bandwidth,
                }
            )
        return adjustments

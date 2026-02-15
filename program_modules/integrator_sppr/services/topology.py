"""Topology service for linking modules and tracking heartbeats."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import EdgeRepository, HeartbeatRepository, NodeRepository
from .. import utils

LOGGER = logging.getLogger("integrator.topology")


class TopologyService:
    def __init__(self, config: AppConfig):
        self.config = config

    def register_node(
        self,
        session: Session,
        name: str,
        kind: str,
        api_url: str,
        priority: int | None = None,
    ) -> dict:
        repo = NodeRepository(session)
        node = repo.upsert(
            name=name,
            kind=kind,
            api_url=api_url,
            priority=priority or self.config.stream.default_priority,
        )
        LOGGER.info("Node %s registered as %s", node.name, node.kind)
        return {"id": node.id, "name": node.name}

    def connect(
        self,
        session: Session,
        from_node: str,
        to_node: str,
        max_bandwidth: float,
    ) -> dict:
        node_repo = NodeRepository(session)
        edge_repo = EdgeRepository(session)
        a = node_repo.upsert(from_node, "external", "http://localhost", self.config.stream.default_priority)
        b = node_repo.upsert(to_node, "external", "http://localhost", self.config.stream.default_priority)
        edge = edge_repo.link(a.id, b.id, max_bandwidth=max_bandwidth)
        LOGGER.info("Edge %s->%s configured", from_node, to_node)
        return {"edge_id": edge.id, "from": from_node, "to": to_node}

    def heartbeat(self, session: Session, node_name: str, latency_ms: float | None = None) -> dict:
        node_repo = NodeRepository(session)
        hb_repo = HeartbeatRepository(session)
        node = node_repo.upsert(node_name, "external", "http://localhost", self.config.stream.default_priority)
        latency = latency_ms or utils.simulated_latency()
        heartbeat = hb_repo.update(node.id, status="alive", latency_ms=latency)
        return {"node": node.name, "latency_ms": latency, "updated_at": heartbeat.updated_at.isoformat()}

    def prune_stale(self, session: Session, threshold: timedelta) -> List[str]:
        repo = HeartbeatRepository(session)
        stale = repo.stale(datetime.utcnow() - threshold)
        for entry in stale:
            entry.status = "stale"
        return [entry.node.name for entry in stale if entry.node]

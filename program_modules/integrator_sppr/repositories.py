"""Repository layer for Integrator SPPR."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


class BaseRepository:
    def __init__(self, session: Session):
        self.session = session


class NodeRepository(BaseRepository):
    def upsert(self, name: str, kind: str, api_url: str, priority: int) -> models.Node:
        stmt = select(models.Node).where(models.Node.name == name)
        node = self.session.scalar(stmt)
        now = datetime.utcnow()
        if node:
            node.api_url = api_url
            node.kind = kind
            node.priority = priority
            node.updated_at = now
        else:
            node = models.Node(
                name=name,
                kind=kind,
                api_url=api_url,
                priority=priority,
                created_at=now,
                updated_at=now,
            )
            self.session.add(node)
            self.session.flush()
        return node

    def list_nodes(self) -> List[models.Node]:
        return list(self.session.scalars(select(models.Node).order_by(models.Node.name)))


class EdgeRepository(BaseRepository):
    def link(
        self,
        from_node: int,
        to_node: int,
        max_bandwidth: float,
    ) -> models.Edge:
        stmt = select(models.Edge).where(
            models.Edge.from_node_id == from_node,
            models.Edge.to_node_id == to_node,
        )
        edge = self.session.scalar(stmt)
        now = datetime.utcnow()
        if edge:
            edge.max_bandwidth = max_bandwidth
            edge.updated_at = now
        else:
            edge = models.Edge(
                from_node_id=from_node,
                to_node_id=to_node,
                max_bandwidth=max_bandwidth,
                created_at=now,
                updated_at=now,
            )
            self.session.add(edge)
            self.session.flush()
        return edge

    def set_metrics(
        self,
        edge_id: int,
        bandwidth: float,
        latency_ms: float,
        reliability: float,
        packet_loss: float,
    ) -> None:
        edge = self.session.get(models.Edge, edge_id)
        if not edge:
            return
        edge.current_bandwidth = bandwidth
        edge.latency_ms = latency_ms
        edge.reliability = reliability
        edge.packet_loss = packet_loss
        edge.updated_at = datetime.utcnow()

    def edges_for_node(self, node_id: int) -> List[models.Edge]:
        stmt = select(models.Edge).where(
            (models.Edge.from_node_id == node_id) | (models.Edge.to_node_id == node_id)
        )
        return list(self.session.scalars(stmt))


class ChannelRepository(BaseRepository):
    def add(
        self,
        edge_id: int,
        label: str,
        modality: str,
        priority: int,
        target_rate: float,
    ) -> models.Channel:
        channel = models.Channel(
            edge_id=edge_id,
            label=label,
            modality=modality,
            priority=priority,
            target_rate=target_rate,
            attributes={},
        )
        self.session.add(channel)
        self.session.flush()
        return channel

    def update_stats(
        self,
        channel_id: int,
        actual_rate: float,
        dropped: int,
        retries: int,
        metadata: Dict,
    ) -> None:
        channel = self.session.get(models.Channel, channel_id)
        if not channel:
            return
        channel.actual_rate = actual_rate
        channel.dropped_packets = dropped
        channel.retries = retries
        channel.attributes = metadata
        channel.updated_at = datetime.utcnow()

    def list_channels(self) -> List[models.Channel]:
        stmt = select(models.Channel).order_by(models.Channel.created_at.desc())
        return list(self.session.scalars(stmt))


class FlowRepository(BaseRepository):
    def add_snapshot(
        self,
        channel_id: int,
        window_start: datetime,
        window_end: datetime,
        volume: float,
        avg_latency_ms: float,
        reliability: float,
        pressure: float,
        snapshot: Dict,
    ) -> models.FlowStat:
        stat = models.FlowStat(
            channel_id=channel_id,
            window_start=window_start,
            window_end=window_end,
            volume=volume,
            avg_latency_ms=avg_latency_ms,
            reliability=reliability,
            pressure=pressure,
            snapshot=snapshot,
        )
        self.session.add(stat)
        self.session.flush()
        return stat

    def recent(self, channel_id: int, limit: int = 10) -> List[models.FlowStat]:
        stmt = (
            select(models.FlowStat)
            .where(models.FlowStat.channel_id == channel_id)
            .order_by(models.FlowStat.window_end.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class AuditRepository(BaseRepository):
    def add(self, event_type: str, level: str, message: str, payload: Dict) -> None:
        record = models.AuditLog(
            event_type=event_type,
            level=level,
            message=message,
            payload=payload,
        )
        self.session.add(record)

    def latest(self, limit: int = 100) -> List[models.AuditLog]:
        stmt = (
            select(models.AuditLog)
            .order_by(models.AuditLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class ReportRepository(BaseRepository):
    def store(self, title: str, body: str, metrics: Dict[str, Any]) -> models.ReportSnapshot:
        report = models.ReportSnapshot(
            title=title,
            body=body,
            metrics=metrics,
        )
        self.session.add(report)
        self.session.flush()
        return report

    def list_reports(self, limit: int = 10) -> List[models.ReportSnapshot]:
        stmt = (
            select(models.ReportSnapshot)
            .order_by(models.ReportSnapshot.generated_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def purge(self, older_than: datetime) -> int:
        stmt = select(models.ReportSnapshot.id).where(
            models.ReportSnapshot.generated_at < older_than
        )
        ids = list(self.session.scalars(stmt))
        if not ids:
            return 0
        deleted = (
            self.session.query(models.ReportSnapshot)
            .filter(models.ReportSnapshot.id.in_(ids))
            .delete(synchronize_session=False)
        )
        return deleted


class HeartbeatRepository(BaseRepository):
    def update(self, node_id: int, status: str, latency_ms: float) -> models.Heartbeat:
        stmt = select(models.Heartbeat).where(models.Heartbeat.node_id == node_id)
        heartbeat = self.session.scalar(stmt)
        now = datetime.utcnow()
        if heartbeat:
            heartbeat.status = status
            heartbeat.latency_ms = latency_ms
            heartbeat.updated_at = now
        else:
            heartbeat = models.Heartbeat(
                node_id=node_id,
                status=status,
                latency_ms=latency_ms,
                updated_at=now,
            )
            self.session.add(heartbeat)
            self.session.flush()
        return heartbeat

    def stale(self, older_than: datetime) -> List[models.Heartbeat]:
        stmt = select(models.Heartbeat).where(models.Heartbeat.updated_at < older_than)
        return list(self.session.scalars(stmt))


def bandwidth_summary(session: Session, window: timedelta) -> Dict[str, float]:
    now = datetime.utcnow()
    since = now - window
    stmt = select(
        func.sum(models.FlowStat.volume),
        func.avg(models.FlowStat.avg_latency_ms),
        func.avg(models.FlowStat.reliability),
    ).where(models.FlowStat.window_end >= since)
    total, latency, reliability = session.execute(stmt).one_or_none() or (0, 0, 0)
    return {
        "volume": float(total or 0.0),
        "latency": float(latency or 0.0),
        "reliability": float(reliability or 0.0),
    }

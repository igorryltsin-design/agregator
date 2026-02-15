"""Graph builder that links signals by similarity and shared context."""

from __future__ import annotations

import itertools
import logging
from datetime import datetime
from typing import Iterable, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import GraphRepository
from .. import utils

LOGGER = logging.getLogger("aggregator.graph")


class GraphBuilder:
    def __init__(self, config: AppConfig, telemetry) -> None:
        self.config = config
        self.telemetry = telemetry

    def update_graph(self, session: Session, signals: Iterable) -> List[int]:
        repo = GraphRepository(session)
        created_edges: List[int] = []
        signals = list(signals)
        for a, b in itertools.combinations(signals, 2):
            similarity = utils.cosine_similarity(a.vector or [], b.vector or [])
            if similarity < self.config.graph.similarity_threshold:
                continue
            relation_type = self._relation_type(a, b)
            weight = similarity * self.config.graph.decay_factor
            edge = repo.upsert_relation(
                from_signal=a.id,
                to_signal=b.id,
                relation_type=relation_type,
                similarity=similarity,
                weight=weight,
                window_start=min(a.recorded_at, b.recorded_at),
                window_end=max(a.recorded_at, b.recorded_at),
            )
            created_edges.append(edge.id)
        if created_edges:
            LOGGER.info("Graph updated with %s edges", len(created_edges))
            self.telemetry.observe("graph.edges", float(len(created_edges)))
        return created_edges

    def _relation_type(self, a, b) -> str:
        if a.source.modality == b.source.modality:
            return f"{a.source.modality}_coherence"
        if a.recorded_at.date() == b.recorded_at.date():
            return "temporal_proximity"
        return "context_bridge"

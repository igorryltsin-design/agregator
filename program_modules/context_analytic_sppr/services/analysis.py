"""Core analytical routines: grouping observations, scoring events, updating relations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import (
    EntityRepository,
    EventRepository,
    LogRepository,
    ObservationRepository,
    RelationRepository,
    VectorRepository,
)
from ..scoring import blend, moving_prominence, similarity, temporal_weight
from ..vectorizers import Vectorizer


@dataclass
class EventSummary:
    event_id: int
    observation_ids: List[int]
    prominence: float
    coherence: float
    density: float


class AnalysisService:
    def __init__(self, config: AppConfig, vectorizer: Vectorizer):
        self.config = config
        self.vectorizer = vectorizer

    def run_window(self, session: Session, limit: int = 64) -> List[EventSummary]:
        observation_repo = ObservationRepository(session)
        vector_repo = VectorRepository(session)
        event_repo = EventRepository(session)
        relation_repo = RelationRepository(session)
        entity_repo = EntityRepository(session)
        log_repo = LogRepository(session)

        observations = observation_repo.fetch_unprocessed(limit=limit)
        if not observations:
            return []
        vectors = vector_repo.collect([item.id for item in observations])
        vector_map = {vector.observation_id: vector for vector in vectors}

        batches = self._group_by_modality(observations)
        summaries: List[EventSummary] = []
        for modality, items in batches.items():
            if not items:
                continue
            clusters = self._cluster(items, vector_map)
            for cluster in clusters:
                summary = self._build_event(event_repo, cluster, vector_map)
                observation_repo.mark_processed([obs.id for obs in cluster])
                self._update_entities(entity_repo, cluster)
                self._update_relations(relation_repo, summary.event_id)
                summaries.append(summary)
        log_repo.add(
            component="analysis",
            level="info",
            message="Processed observations",
            extra={"count": len(observations)},
        )
        return summaries

    def _group_by_modality(self, observations) -> Dict[str, List]:
        buckets: Dict[str, List] = {}
        for item in observations:
            buckets.setdefault(item.modality, []).append(item)
        return buckets

    def _cluster(self, observations, vector_map) -> List[List]:
        clusters: List[List] = []
        for observation in observations:
            vector = vector_map.get(observation.id)
            if not vector:
                continue
            assigned = False
            for cluster in clusters:
                reference = vector_map[cluster[0].id]
                sim = similarity(vector.values, reference.values, self.config.scoring)
                if sim.accepted:
                    cluster.append(observation)
                    assigned = True
                    break
            if not assigned:
                clusters.append([observation])
        return clusters

    def _build_event(
        self,
        repo: EventRepository,
        cluster: Sequence,
        vector_map,
    ) -> EventSummary:
        title = f"Событие ({cluster[0].modality})"
        synopsis = " ".join((obs.annotation or obs.content_ref)[:120] for obs in cluster[:5])
        start_at = min(obs.recorded_at for obs in cluster)
        end_at = max(obs.recorded_at for obs in cluster)
        coherence, density, prominence = self._metrics(cluster, vector_map)
        event = repo.create_event(title, synopsis[:512], start_at, end_at, coherence, density, prominence)
        weights = self._weights(cluster, vector_map)
        repo.attach_observations(event.id, [obs.id for obs in cluster], weights)
        return EventSummary(
            event_id=event.id,
            observation_ids=[obs.id for obs in cluster],
            prominence=prominence,
            coherence=coherence,
            density=density,
        )

    def _metrics(self, observations, vector_map):
        vectors = [vector_map[obs.id].values for obs in observations if obs.id in vector_map]
        coherence = 0.0
        if len(vectors) > 1:
            sims = []
            base = vectors[0]
            for vector in vectors[1:]:
                sims.append(similarity(base, vector, self.config.scoring).score)
            coherence = sum(sims) / len(sims)
        density = len(observations) / max(
            (max(obs.recorded_at for obs in observations) - min(obs.recorded_at for obs in observations)).total_seconds(),
            1.0,
        )
        prominence = moving_prominence(
            [
                blend(obs.intensity, obs.confidence, temporal_weight(obs.recorded_at, self.config.scoring))
                for obs in observations
            ],
            self.config.scoring.prominence_decay,
        )
        return coherence, density, prominence

    def _weights(self, observations, vector_map):
        raw = [
            blend(obs.intensity, vector_map[obs.id].norm if obs.id in vector_map else 1.0)
            for obs in observations
        ]
        total = sum(raw) or 1.0
        return [value / total for value in raw]

    def _update_entities(self, repo: EntityRepository, cluster: Sequence) -> None:
        for observation in cluster:
            entity = repo.get_or_create(canonical_name=f"entity-{observation.id}", category="generic")
            repo.attach(entity.id, observation.id, salience=0.5)

    def _update_relations(self, repo: RelationRepository, event_id: int) -> None:
        neighborhood = repo.neighborhood(event_id, limit=3)
        for relation in neighborhood:
            repo.upsert(
                from_event=event_id,
                to_event=relation.to_event_id,
                relation_type="follow_up",
                similarity=min(1.0, relation.similarity + 0.05),
                causality_score=min(1.0, relation.causality_score + 0.02),
            )

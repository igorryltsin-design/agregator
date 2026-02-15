"""Similarity graph builder for artifact relations."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List

from sqlalchemy.orm import Session

from .. import models
from ..config import AppConfig
from ..repositories import ArtifactRepository, SimilarityRepository

LOGGER = logging.getLogger("scanner.graph")


class SimilarityGraph:
    def __init__(self, config: AppConfig):
        self.config = config

    def build(self, session: Session, artifact_ids: List[int]) -> dict:
        repo = ArtifactRepository(session)
        similarity_repo = SimilarityRepository(session)
        artifacts = [session.get(models.Artifact, aid) for aid in artifact_ids]
        artifacts = [artifact for artifact in artifacts if artifact]
        established = []
        for a, b in combinations(artifacts, 2):
            score = self._similarity(a, b)
            if score >= self.config.graph_threshold:
                edge = similarity_repo.upsert(a.id, b.id, score, reason="signature_overlap")
                established.append({"edge_id": edge.id, "score": score, "from": a.id, "to": b.id})
        LOGGER.info("Graph built with %s edges", len(established))
        return {"edges": established}

    def _similarity(self, a: models.Artifact, b: models.Artifact) -> float:
        if not a.signatures or not b.signatures:
            return 0.0
        sigs_a = {block.signature for block in a.signatures}
        sigs_b = {block.signature for block in b.signatures}
        if not sigs_a or not sigs_b:
            return 0.0
        intersection = len(sigs_a & sigs_b)
        union = len(sigs_a | sigs_b)
        return intersection / union

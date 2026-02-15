"""Ingestion layer for Contextual Analytics."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import ObservationRepository, SourceRepository, VectorRepository
from ..vectorizers import Vectorizer


class IngestionService:
    def __init__(self, config: AppConfig, vectorizer: Vectorizer):
        self.config = config
        self.vectorizer = vectorizer

    def register_source(
        self,
        session: Session,
        name: str,
        modality: str,
        description: str | None,
        relevance_bias: float,
        latency_expectation: int,
    ) -> dict:
        repo = SourceRepository(session)
        source = repo.get_or_create(
            name=name,
            modality=modality,
            description=description,
            relevance_bias=relevance_bias,
            latency_expectation=latency_expectation,
        )
        return {"id": source.id, "name": source.name, "modality": source.modality}

    def ingest_observation(
        self,
        session: Session,
        source_name: str,
        modality: str,
        content_ref: str,
        recorded_at: datetime,
        annotation: str | None,
        attributes: Dict,
        intensity: float,
        confidence: float,
    ) -> dict:
        source_repo = SourceRepository(session)
        observation_repo = ObservationRepository(session)
        vector_repo = VectorRepository(session)

        source = source_repo.get_or_create(
            name=source_name,
            modality=modality,
            description=None,
            relevance_bias=0.5,
            latency_expectation=5,
        )

        observation = observation_repo.add(
            source_id=source.id,
            recorded_at=recorded_at,
            modality=modality,
            content_ref=content_ref,
            annotation=annotation,
            attributes=attributes,
            intensity=intensity,
            confidence=confidence,
        )

        vector = self._vectorize(modality, annotation or content_ref)
        vector_repo.attach(
            observation_id=observation.id,
            vector_type=f"default:{modality}",
            values=vector,
            norm=max(1e-6, sum(value * value for value in vector) ** 0.5),
        )

        return {"id": observation.id, "source": source.name}

    def _vectorize(self, modality: str, payload: str):
        if modality == "text":
            return self.vectorizer.encode_text(payload)
        if modality == "audio":
            return self.vectorizer.encode_audio(payload)
        if modality == "radar":
            return self.vectorizer.encode_radar(payload)
        if modality in {"image", "video"}:
            return self.vectorizer.encode_visual(payload)
        return self.vectorizer.encode_text(payload)

"""Ingestion service: registers sources and normalizes signals."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from typing import Deque, Tuple

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import DataSourceRepository, LogRepository, SignalRepository
from .. import utils
from .validators import SignalPayload, SourcePayload, ValidationError

LOGGER = logging.getLogger("aggregator.ingestion")


class StreamIngestor:
    """Handles incoming signals, deduplication and normalization."""

    def __init__(self, config: AppConfig, telemetry) -> None:
        self.config = config
        self.telemetry = telemetry
        self._recent_signatures: Deque[Tuple[str, datetime]] = deque()

    def register_source(
        self, session: Session, payload: SourcePayload
    ) -> dict:
        repo = DataSourceRepository(session)
        existing = repo.get_by_name(payload.name)
        if existing:
            LOGGER.debug("Source %s already exists", payload.name)
            return {"id": existing.id, "created": False}

        entity = repo.create(
            name=payload.name,
            modality=payload.modality,
            reliability=payload.reliability,
            priority=payload.priority,
            description=payload.description,
        )
        LOGGER.info("Registered data source %s", payload.name)
        return {"id": entity.id, "created": True}

    def ingest_signal(self, session: Session, payload: SignalPayload) -> dict:
        ds_repo = DataSourceRepository(session)
        signal_repo = SignalRepository(session)
        log_repo = LogRepository(session)

        source = ds_repo.get_by_name(payload.source_name)
        if not source:
            raise ValidationError(f"Unknown source {payload.source_name}")

        signature = utils.hash_payload(payload.payload_ref)
        if self._is_duplicate(signature, payload.recorded_at):
            LOGGER.warning("Duplicate signal detected for %s", payload.payload_ref)
            log_repo.add(
                component="ingestion",
                level="warning",
                message="Duplicate signal rejected",
                extra={"payload_ref": payload.payload_ref},
            )
            return {"duplicate": True}

        vector = payload.metadata.get("vector") or utils.random_vector()
        recency_sec = (utils.utc_now() - payload.recorded_at).total_seconds()
        reliability = utils.reliability_score(
            base=source.reliability_baseline,
            intensity=payload.intensity,
            recency_sec=recency_sec,
            modality_bias=utils.modality_bias(source.modality),
        )

        entity = signal_repo.add(
            source_id=source.id,
            recorded_at=payload.recorded_at,
            signal_type=payload.signal_type,
            payload_ref=utils.expand_payload_ref(payload.payload_ref),
            vector=vector,
            attributes=payload.metadata,
            intensity=utils.clamp(payload.intensity),
            reliability=reliability,
        )

        log_repo.add(
            component="ingestion",
            level="info",
            message="Signal ingested",
            extra={
                "signal_id": entity.id,
                "source": payload.source_name,
                "reliability": reliability,
            },
        )
        self.telemetry.observe("signals.ingested", 1.0)
        return {
            "id": entity.id,
            "reliability": reliability,
            "intensity": entity.intensity,
        }

    def _is_duplicate(self, signature: str, recorded_at: datetime) -> bool:
        # purge old signatures
        window = self.config.ingestion.deduplicate_window
        while self._recent_signatures:
            sig, ts = self._recent_signatures[0]
            if (recorded_at - ts) > window:
                self._recent_signatures.popleft()
            else:
                break
        if any(sig == signature for sig, _ in self._recent_signatures):
            return True
        self._recent_signatures.append((signature, recorded_at))
        return False

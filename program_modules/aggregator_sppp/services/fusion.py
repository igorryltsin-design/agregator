"""FusionEngine builds coherent samples from normalized signals."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import FusionRepository, SignalRepository
from .. import utils
from .validators import FusionWindowPayload

LOGGER = logging.getLogger("aggregator.fusion")


class FusionEngine:
    def __init__(self, config: AppConfig, telemetry) -> None:
        self.config = config
        self.telemetry = telemetry

    def fuse_window(self, session: Session, payload: FusionWindowPayload) -> dict:
        repo = SignalRepository(session)
        fusion_repo = FusionRepository(session)

        signals = repo.fetch_window(payload.window_start, payload.window_end)
        if not signals:
            return {"samples": []}

        grouped = self._group_signals(signals)
        samples = []
        for group in grouped:
            summary = utils.summarize_signals(
                [signal.attributes.get("text_summary") or signal.payload_ref for signal in group]
            )
            weights = utils.softmax([signal.reliability for signal in group])
            context_score = sum(
                signal.intensity * weight for signal, weight in zip(group, weights, strict=False)
            )
            coherence = self._estimate_coherence(group)

            sample = fusion_repo.create_sample(
                start_at=group[0].recorded_at,
                end_at=group[-1].recorded_at,
                summary=summary,
                context_score=context_score,
                coherence=coherence,
            )
            fusion_repo.attach_signals(
                sample.id,
                [signal.id for signal in group],
                weights,
            )
            samples.append(
                {
                    "sample_id": sample.id,
                    "signal_ids": [signal.id for signal in group],
                    "context_score": context_score,
                    "coherence": coherence,
                }
            )
            LOGGER.info(
                "Fusion sample %s created with %s signals", sample.id, len(group)
            )
            self.telemetry.observe("fusion.samples", 1.0)
        return {"samples": samples}

    def _group_signals(self, signals: Sequence) -> List[List]:
        if not signals:
            return []
        window_map: Dict[Tuple[str, int], List] = defaultdict(list)
        threshold = self.config.fusion.alignment_threshold
        span_limit = self.config.fusion.max_group_span
        for signal in signals:
            bucket_key = signal.source.modality
            assigned = False
            for key, bucket in window_map.items():
                head = bucket[0]
                time_delta = abs(
                    (signal.recorded_at - head.recorded_at).total_seconds()
                )
                similarity = utils.cosine_similarity(
                    signal.vector or [], head.vector or []
                )
                if (
                    time_delta <= span_limit.total_seconds()
                    and similarity >= threshold
                    and key[0] == signal.source.modality
                ):
                    bucket.append(signal)
                    assigned = True
                    break
            if not assigned:
                window_map[(bucket_key, signal.id)].append(signal)
        return [sorted(bucket, key=lambda s: s.recorded_at) for bucket in window_map.values()]

    def _estimate_coherence(self, signals: Sequence) -> float:
        if len(signals) < 2:
            return 1.0
        vectors = [signal.vector or [] for signal in signals]
        similarities = []
        for idx in range(len(vectors) - 1):
            similarities.append(
                utils.cosine_similarity(vectors[idx], vectors[idx + 1])
            )
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        intensities = [signal.intensity for signal in signals]
        intensity_spread = max(intensities) - min(intensities)
        coherence = avg_similarity - 0.1 * intensity_spread
        return utils.clamp(coherence, 0.0, 1.0)

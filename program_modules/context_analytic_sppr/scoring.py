"""Scoring helpers for contextual prominence and evidence building."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Sequence

from .config import ScoringConfig
from .vectorizers import cosine


@dataclass(slots=True)
class SimilarityResult:
    score: float
    threshold: float
    accepted: bool


def similarity(a: Sequence[float], b: Sequence[float], config: ScoringConfig) -> SimilarityResult:
    score = cosine(list(a), list(b))
    accepted = score >= config.similarity_threshold
    return SimilarityResult(score=score, threshold=config.similarity_threshold, accepted=accepted)


def moving_prominence(values: Iterable[float], decay: float) -> float:
    accumulator = 0.0
    for value in values:
        accumulator = accumulator * decay + value
    return accumulator


def temporal_weight(recorded_at: datetime, config: ScoringConfig) -> float:
    delta = datetime.utcnow() - recorded_at
    minutes = max(delta.total_seconds() / 60, 0.1)
    hot_window = config.hot_window_minutes
    if minutes <= hot_window:
        return 1.0
    return math.exp(-minutes / (hot_window * 4))


def blend(*values: float) -> float:
    filled = [value for value in values if value is not None]
    return sum(filled) / len(filled) if filled else 0.0

"""Utility helpers for ingestion, normalization and scoring."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def normalize_timestamp(ts: datetime | str) -> datetime:
    if isinstance(ts, datetime):
        if ts.tzinfo:
            return ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
        timezone.utc
    ).replace(tzinfo=None)


def hash_payload(payload_ref: str) -> str:
    return hashlib.sha256(payload_ref.encode("utf-8")).hexdigest()


def random_vector(dim: int = 32) -> List[float]:
    return [round(random.uniform(-1, 1), 6) for _ in range(dim)]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    if total == 0:
        return [0.0 for _ in scores]
    return [s / total for s in exp_scores]


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def moving_average(values: Iterable[float], window: int = 5) -> List[float]:
    buffer: List[float] = []
    result: List[float] = []
    for value in values:
        buffer.append(value)
        if len(buffer) > window:
            buffer.pop(0)
        result.append(sum(buffer) / len(buffer))
    return result


def summarize_signals(texts: Sequence[str]) -> str:
    chunks = [text for text in texts if text]
    if not chunks:
        return "No signals available."
    if len(chunks) == 1:
        return chunks[0][:512]
    head = " | ".join(chunks[:5])
    return head[:1024]


def reliability_score(
    base: float,
    intensity: float,
    recency_sec: float,
    modality_bias: float,
) -> float:
    decay = math.exp(-recency_sec / 3600)
    score = base * 0.4 + intensity * 0.3 + modality_bias * 0.2 + decay * 0.1
    return clamp(score)


MODALITY_BIAS = {
    "audio": 0.5,
    "text": 0.6,
    "visual": 0.55,
    "radar": 0.7,
    "composite": 0.65,
}


def modality_bias(modality: str) -> float:
    return MODALITY_BIAS.get(modality, 0.5)


@dataclass(slots=True)
class WindowStats:
    intensity_avg: float
    reliability_avg: float
    spread_seconds: float


def window_stats(signals: Sequence[dict]) -> WindowStats:
    if not signals:
        return WindowStats(0.0, 0.0, 0.0)
    intensities = [item["intensity"] for item in signals]
    reliabilities = [item["reliability"] for item in signals]
    times = [item["timestamp"].timestamp() for item in signals]
    return WindowStats(
        intensity_avg=float(statistics.mean(intensities)),
        reliability_avg=float(statistics.mean(reliabilities)),
        spread_seconds=max(times) - min(times) if len(times) > 1 else 0.0,
    )


def expand_payload_ref(ref: str) -> str:
    if ref.startswith("file://"):
        return ref
    return f"file://{ref}"

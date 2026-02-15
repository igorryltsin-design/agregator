"""Utility helpers for stream normalization and load-balancing."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Iterable, List


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def smoothing(values: Iterable[float], window: int) -> float:
    values = list(values)[-window:]
    if not values:
        return 0.0
    return sum(values) / len(values)


def adaptive_priority(current: int, reliability: float, backpressure: float) -> int:
    delta = 0
    if reliability < 0.8:
        delta -= 1
    if backpressure > 0.8:
        delta -= 1
    if reliability > 0.95 and backpressure < 0.5:
        delta += 1
    return max(1, min(5, current + delta))


@dataclass
class SlidingGauge:
    size: int
    values: Deque[float] = field(default_factory=deque)

    def push(self, value: float) -> float:
        self.values.append(value)
        if len(self.values) > self.size:
            self.values.popleft()
        return sum(self.values) / len(self.values)


def reliability_index(latency_ms: float, packet_loss: float, retries: int) -> float:
    latency_penalty = clamp(latency_ms / 1000.0)
    loss_penalty = clamp(packet_loss)
    retry_penalty = clamp(retries / 10.0)
    score = 1.0 - (0.4 * latency_penalty + 0.4 * loss_penalty + 0.2 * retry_penalty)
    return clamp(score)


def simulated_latency() -> float:
    return round(random.uniform(50, 600), 2)


def simulated_loss() -> float:
    return round(random.uniform(0.0, 0.2), 3)


def timestamp() -> str:
    return datetime.utcnow().isoformat()

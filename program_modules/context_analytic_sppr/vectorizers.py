"""Simple deterministic vectorizers for multimodal content."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, List

from .config import VectorizerConfig


def _seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass
class Vectorizer:
    config: VectorizerConfig

    def _rand(self, key: str):
        seed = _seed(key)
        rng = random.Random(seed)
        return rng

    def _create_vector(self, key: str, dim: int | None = None) -> List[float]:
        size = dim or self.config.default_dim
        rng = self._rand(key)
        return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(size)]

    def encode_text(self, text: str) -> List[float]:
        return self._create_vector(f"text::{text}")

    def encode_audio(self, fingerprint: str) -> List[float]:
        return [
            round(math.sin(idx + _seed(fingerprint)) * self.config.audio_bias, 6)
            for idx in range(self.config.default_dim)
        ]

    def encode_visual(self, payload: str) -> List[float]:
        rng = self._rand(f"visual::{payload}")
        return [round(rng.gauss(0, self.config.visual_bias), 6) for _ in range(self.config.default_dim)]

    def encode_radar(self, payload: str) -> List[float]:
        base = _seed(payload) % 997
        return [
            round(math.cos(base + idx) * self.config.radar_bias, 6)
            for idx in range(self.config.default_dim)
        ]


def l2(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = l2(a)
    norm_b = l2(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

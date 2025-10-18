from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .context import ContextCandidate

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[misc]


@dataclass(slots=True)
class CrossEncoderConfig:
    model_name: str
    device: Optional[str] = None
    batch_size: int = 16
    max_length: int = 512
    truncate_chars: int = 1200


class CrossEncoderReranker:
    """Обёртка над sentence-transformers CrossEncoder для rerank списка чанков."""

    def __init__(self, config: CrossEncoderConfig) -> None:
        if CrossEncoder is None:
            raise RuntimeError(
                "CrossEncoder недоступен: пакет 'sentence-transformers' не установлен."
            )
        self._config = config
        self._model = CrossEncoder(
            config.model_name,
            device=config.device,
            max_length=config.max_length,
        )
        self._lock = threading.Lock()

    def __call__(self, query: str, items: List[ContextCandidate]) -> List[ContextCandidate]:
        if not items:
            return items
        pairs: List[Sequence[str]] = []
        truncate_at = max(200, int(self._config.truncate_chars or 0)) if self._config.truncate_chars else None
        for cand in items:
            text = (cand.chunk.content or cand.preview or "").strip()
            if truncate_at and len(text) > truncate_at:
                text = text[:truncate_at]
            if not text:
                text = cand.preview or ""
            pairs.append((query, text))
        with self._lock:
            scores = self._model.predict(
                pairs,
                batch_size=max(1, int(self._config.batch_size or 1)),
                show_progress_bar=False,
            )
        enriched: List[ContextCandidate] = []
        for cand, score in zip(items, scores):
            s = float(score)
            cand.adjusted_score = s
            cand.reasoning_hint = (
                f"{cand.reasoning_hint}; rerank={s:.3f}" if cand.reasoning_hint else f"rerank={s:.3f}"
            )
            enriched.append(cand)
        enriched.sort(key=lambda item: item.adjusted_score, reverse=True)
        return enriched

    def close(self) -> None:
        try:
            if hasattr(self._model, "cpu"):
                self._model.cpu()
        except Exception:  # pragma: no cover - best effort
            pass


def load_reranker(kind: str, *, config: CrossEncoderConfig | None = None) -> Optional[CrossEncoderReranker]:
    resolved = (kind or "").strip().lower()
    if not resolved or resolved in {"none", "off"}:
        return None
    if resolved in {"cross", "cross-encoder", "cross_encoder"}:
        if config is None:
            raise ValueError("Для cross-encoder rerank требуется config")
        return CrossEncoderReranker(config)
    raise ValueError(f"Неизвестный rerank backend: {kind}")


__all__ = ["CrossEncoderConfig", "CrossEncoderReranker", "load_reranker"]

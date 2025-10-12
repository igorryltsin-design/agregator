from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass
from typing import List, Protocol, Sequence

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


class EmbeddingBackend(Protocol):
    """Простой контракт для backend'ов построения эмбеддингов."""

    name: str
    model_name: str
    model_version: str
    dim: int

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class HashEmbeddingBackend:
    """Детерминированный fallback, не требующий ML-модели.

    Использует seed на основе SHA256 текста и генерирует псевдо-случайные
    векторы заданной размерности с нормализацией. Предназначен как запасной
    вариант для дев-сред и тестов.
    """

    dim: int = 384
    normalize: bool = True
    model_name: str = "hash-fallback"
    model_version: str = "hash-v1"
    name: str = "hash"

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            seed = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
            rng = random.Random(int(seed, 16))
            vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
            if self.normalize:
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors

    def close(self) -> None:
        return None


class SentenceTransformersBackend:
    """Backend на основе sentence-transformers (HuggingFace)."""

    name = "sentence-transformers"

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers не установлен. Установите пакет 'sentence-transformers' "
                "или используйте backend=hash."
            )
        self.model_name = model_name
        self.model_version = getattr(SentenceTransformer, "__version__", "unknown")  # type: ignore[attr-defined]
        self.dim = 0
        self.normalize = normalize
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)  # type: ignore[call-arg]
        try:
            example = self._model.encode(["test"], convert_to_numpy=True)  # type: ignore[arg-type]
            if example is not None and hasattr(example, "shape"):
                self.dim = int(getattr(example, "shape")[1] or 0)
        except Exception:
            logger.exception("Failed to probe embedding dimension for model %s", model_name)
            self.dim = 0
        if self.dim <= 0:
            self.dim = int(getattr(self._model, "get_sentence_embedding_dimension", lambda: 0)() or 0)
        if self.dim <= 0:
            raise RuntimeError(f"Не удалось определить размерность эмбеддингов для модели {model_name}")

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return result.tolist()

    def close(self) -> None:
        try:
            if hasattr(self._model, "cpu"):
                self._model.cpu()  # type: ignore[attr-defined]
        except Exception:
            pass


def load_embedding_backend(
    backend: str = "auto",
    *,
    model_name: str = "intfloat/multilingual-e5-large",
    dim: int = 384,
    normalize: bool = True,
    batch_size: int = 32,
    device: str | None = None,
) -> EmbeddingBackend:
    """Создаёт backend эмбеддингов, подбирая реализацию по параметрам."""
    backend = (backend or "auto").strip().lower()
    if backend == "hash":
        return HashEmbeddingBackend(dim=dim, normalize=normalize, model_name=model_name)
    if backend in {"sentence-transformers", "st"} or (backend == "auto" and SentenceTransformer is not None):
        return SentenceTransformersBackend(
            model_name=model_name,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
    if backend != "auto":
        raise RuntimeError(f"Неизвестный backend эмбеддингов: {backend}")
    # fallback
    logger.warning("Используется hash backend для эмбеддингов (sentence-transformers не найден).")
    return HashEmbeddingBackend(dim=dim, normalize=normalize, model_name=model_name)

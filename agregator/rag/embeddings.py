from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from agregator.runtime_settings import runtime_settings_store
from agregator.services.http import http_request

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


class LmStudioEmbeddingBackend:
    """OpenAI-совместимый backend (LM Studio, OpenAI, совместимые API)."""

    name = "lm-studio"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        timeout: float = 120.0,
        model_version: Optional[str] = None,
        dim: Optional[int] = None,
    ) -> None:
        if not base_url:
            raise RuntimeError("LM Studio embeddings backend требует базовый URL (например http://localhost:1234/v1).")
        self.model_name = model_name
        self.model_version = model_version or "lm-studio"
        self.normalize = normalize
        self.batch_size = max(1, int(batch_size or 1))
        self.timeout = max(10.0, float(timeout or 120.0))
        self.dim = int(dim or 0)
        self._api_key = api_key or ""
        self._base_url = base_url.rstrip("/")
        self._endpoint = self._resolve_endpoint(self._base_url)
        self._headers = {"Content-Type": "application/json"}
        if self._api_key:
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    @staticmethod
    def _resolve_endpoint(base: str) -> str:
        if not base:
            return base
        if re.search(r"/embeddings(?:\?|$)", base, flags=re.IGNORECASE):
            return base
        if re.search(r"/v\d+(?:/|$)", base):
            return f"{base.rstrip('/')}/embeddings"
        return f"{base.rstrip('/')}/v1/embeddings"

    @staticmethod
    def _chunk(items: Sequence[str], size: int) -> Sequence[Sequence[str]]:
        size = max(1, size)
        return [items[i : i + size] for i in range(0, len(items), size)]

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        batches = self._chunk(list(texts), self.batch_size)
        for batch in batches:
            payload = {
                "model": self.model_name,
                "input": batch,
            }
            response = http_request(
                "POST",
                self._endpoint,
                json=payload,
                headers=self._headers,
                timeout=(min(10.0, self.timeout * 0.2), self.timeout),
                raise_for_status=True,
            )
            try:
                data = response.json()
            except Exception as exc:  # noqa: BLE001 - хотим любой эксепшн
                raise RuntimeError(f"Некорректный ответ embeddings API: {exc}") from exc
            if not isinstance(data, dict) or "data" not in data:
                raise RuntimeError(f"Неожиданный ответ embeddings API: {data!r}")
            embeddings = data.get("data") or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Количество эмбеддингов ({len(embeddings)}) не совпадает с размером батча ({len(batch)})"
                )
            for item in embeddings:
                vec = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(vec, (list, tuple)):
                    raise RuntimeError("Embeddings API вернул некорректный формат вектора.")
                vector = [float(v) for v in vec]
                if self.normalize:
                    norm = math.sqrt(sum(val * val for val in vector)) or 1.0
                    vector = [val / norm for val in vector]
                results.append(vector)
                if not self.dim and vector:
                    self.dim = len(vector)
        return results

    def close(self) -> None:
        return None


def load_embedding_backend(
    backend: str = "auto",
    *,
    model_name: str = "intfloat/multilingual-e5-large",
    dim: int = 384,
    normalize: bool = True,
    batch_size: int = 32,
    device: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> EmbeddingBackend:
    """Создаёт backend эмбеддингов, подбирая реализацию по параметрам."""
    backend = (backend or "auto").strip().lower()
    if backend == "hash":
        return HashEmbeddingBackend(dim=dim, normalize=normalize, model_name=model_name)
    if backend in {"lm-studio", "lmstudio", "openai"}:
        resolved_base = (base_url or _default_embeddings_base()) or ""
        resolved_key = api_key or _default_embeddings_key()
        resolved_timeout = timeout or 120.0
        return LmStudioEmbeddingBackend(
            resolved_base,
            model_name=model_name,
            api_key=resolved_key or None,
            normalize=normalize,
            batch_size=batch_size,
            timeout=resolved_timeout,
            model_version="lm-studio",
            dim=dim if dim > 0 else None,
        )
    if backend in {"sentence-transformers", "st"} or (backend == "auto" and SentenceTransformer is not None):
        return SentenceTransformersBackend(
            model_name=model_name,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
    if backend == "auto":
        preferred = _runtime_preferred_backend()
        if preferred and preferred not in {"auto", "hash"}:
            return load_embedding_backend(
                preferred,
                model_name=model_name,
                dim=dim,
                normalize=normalize,
                batch_size=batch_size,
                device=device,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
    if backend != "auto":
        raise RuntimeError(f"Неизвестный backend эмбеддингов: {backend}")
    # fallback
    logger.warning("Используется hash backend для эмбеддингов (sentence-transformers не найден).")
    return HashEmbeddingBackend(dim=dim, normalize=normalize, model_name=model_name)


def _runtime_preferred_backend() -> str | None:
    try:
        runtime = runtime_settings_store.current
    except Exception:
        runtime = None
    if not runtime:
        return os.getenv("RAG_EMBEDDING_BACKEND")
    return getattr(runtime, "rag_embedding_backend", None)


def _default_embeddings_base() -> str:
    # runtime setting preferred, fall back to env or LM Studio base
    try:
        runtime = runtime_settings_store.current
    except Exception:
        runtime = None
    if runtime:
        endpoint = getattr(runtime, "rag_embedding_endpoint", None) or runtime.lmstudio_api_base
        return endpoint or ""
    return os.getenv("RAG_EMBEDDING_ENDPOINT") or os.getenv("LMSTUDIO_API_BASE", "")


def _default_embeddings_key() -> str:
    try:
        runtime = runtime_settings_store.current
    except Exception:
        runtime = None
    if runtime:
        token = getattr(runtime, "rag_embedding_api_key", None) or runtime.lmstudio_api_key
        return token or ""
    return os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LMSTUDIO_API_KEY", "")

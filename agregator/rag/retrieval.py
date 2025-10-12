from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from models import RagChunkEmbedding, RagDocument, RagDocumentChunk, db
from .utils import bytes_to_vector


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: RagDocumentChunk
    document: Optional[RagDocument]
    score: float
    lang: Optional[str]
    keywords: Optional[str]
    preview: Optional[str]


class VectorRetriever:
    """Простейший dense-поиск по чанкам с использованием косинусного сходства."""

    def __init__(
        self,
        session: Optional[Session] = None,
        *,
        model_name: str,
        model_version: str | None = None,
        max_candidates: int = 1000,
    ) -> None:
        self.session: Session = session or db.session  # type: ignore[assignment]
        self.model_name = model_name
        self.model_version = model_version
        self.max_candidates = max_candidates

    def _query_embeddings(
        self,
        *,
        exclude_document_ids: Optional[Iterable[int]] = None,
        allowed_document_ids: Optional[Iterable[int]] = None,
    ) -> List[Tuple[RagDocumentChunk, List[float]]]:
        query = (
            self.session.query(RagDocumentChunk, RagChunkEmbedding.vector)
            .join(RagChunkEmbedding, RagChunkEmbedding.chunk_id == RagDocumentChunk.id)
        )
        query = query.filter(RagChunkEmbedding.model_name == self.model_name)
        if self.model_version:
            query = query.filter(RagChunkEmbedding.model_version == self.model_version)
        if exclude_document_ids:
            exclude_ids = list(exclude_document_ids)
            if exclude_ids:
                query = query.filter(~RagDocumentChunk.document_id.in_(exclude_ids))
        if allowed_document_ids:
            include_ids = list(allowed_document_ids)
            if include_ids:
                query = query.filter(RagDocumentChunk.document_id.in_(include_ids))
            else:
                return []
        query = query.order_by(RagDocumentChunk.id.asc())
        if self.max_candidates:
            query = query.limit(self.max_candidates)
        rows = query.all()
        return [
            (chunk, bytes_to_vector(vector_bytes))
            for chunk, vector_bytes in rows
            if vector_bytes
        ]

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 6,
        exclude_document_ids: Optional[Iterable[int]] = None,
        allowed_document_ids: Optional[Iterable[int]] = None,
    ) -> List[RetrievedChunk]:
        candidates = self._query_embeddings(
            exclude_document_ids=exclude_document_ids,
            allowed_document_ids=allowed_document_ids,
        )
        if not candidates:
            return []
        scored: List[Tuple[float, RagDocumentChunk]] = []
        for chunk, vector in candidates:
            score = _cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            scored.append((score, chunk))
        if not scored:
            return []
        top = heapq.nlargest(top_k, scored, key=lambda pair: pair[0])
        documents = {
            doc.id: doc
            for doc in self.session.query(RagDocument).filter(RagDocument.id.in_({c.document_id for _, c in top if c.document_id}))
        }
        result: List[RetrievedChunk] = []
        for score, chunk in top:
            result.append(
                RetrievedChunk(
                    chunk=chunk,
                    document=documents.get(chunk.document_id) if chunk.document_id else None,
                    score=score,
                    lang=chunk.lang_primary,
                    keywords=chunk.keywords_top,
                    preview=chunk.preview,
                )
            )
        return result

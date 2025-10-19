from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import func, literal
from sqlalchemy.orm import Session, joinedload

from models import File, RagChunkEmbedding, RagDocument, RagDocumentChunk, db
from .utils import bytes_to_vector, vector_to_bytes


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
        query_vector_bytes: Optional[bytes] = None,
    ) -> List[Tuple[RagDocumentChunk, List[float], Optional[float]]]:
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
        dialect_name = ""
        try:
            bind = self.session.get_bind()
            if bind is not None and bind.dialect and getattr(bind.dialect, "name", None):
                dialect_name = bind.dialect.name.lower()
        except Exception:
            dialect_name = ""
        score_expr = None
        if query_vector_bytes and dialect_name == "sqlite":
            score_expr = func.cosine_similarity(
                RagChunkEmbedding.vector,
                literal(query_vector_bytes),
            )
            query = query.add_columns(score_expr.label("rag_dense_score")).order_by(score_expr.desc())
        else:
            query = query.order_by(RagDocumentChunk.id.asc())
        if self.max_candidates:
            query = query.limit(self.max_candidates)
        rows = query.all()
        results: List[Tuple[RagDocumentChunk, List[float], Optional[float]]] = []
        for row in rows:
            if score_expr is not None:
                chunk, vector_bytes, pre_score = row
            else:
                chunk, vector_bytes = row
                pre_score = None
            if not vector_bytes:
                continue
            results.append((chunk, bytes_to_vector(vector_bytes), float(pre_score) if pre_score is not None else None))
        return results

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 6,
        exclude_document_ids: Optional[Iterable[int]] = None,
        allowed_document_ids: Optional[Iterable[int]] = None,
    ) -> List[RetrievedChunk]:
        query_vector_bytes = b""
        if query_vector:
            try:
                query_vector_bytes = vector_to_bytes(query_vector)
            except Exception:
                query_vector_bytes = b""
        candidates = self._query_embeddings(
            exclude_document_ids=exclude_document_ids,
            allowed_document_ids=allowed_document_ids,
            query_vector_bytes=query_vector_bytes or None,
        )
        if not candidates:
            return []
        scored: List[Tuple[float, RagDocumentChunk]] = []
        for chunk, vector, pre_score in candidates:
            score = pre_score if pre_score is not None else _cosine_similarity(query_vector, vector)
            if score <= 0:
                continue
            scored.append((score, chunk))
        if not scored:
            return []
        top = heapq.nlargest(top_k, scored, key=lambda pair: pair[0])
        doc_ids = {c.document_id for _, c in top if c.document_id}
        documents = {}
        if doc_ids:
            fetched = (
                self.session.query(RagDocument)
                .options(
                    joinedload(RagDocument.file).joinedload(File.collection),
                    joinedload(RagDocument.file).joinedload(File.tags),
                )
                .filter(RagDocument.id.in_(doc_ids))
                .all()
            )
            documents = {doc.id: doc for doc in fetched}
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

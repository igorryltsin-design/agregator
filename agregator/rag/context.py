from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from models import RagDocument, RagDocumentChunk
from .retrieval import RetrievedChunk, VectorRetriever
from .sparse import KeywordMatch, KeywordRetriever


@dataclass(slots=True)
class ContextCandidate:
    chunk: RagDocumentChunk
    document: Optional[RagDocument]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    adjusted_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    reasoning_hint: str = ""


class ContextSelector:
    """Комбинирует dense и keyword-результаты, обеспечивает разнообразие по документам."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: Optional[KeywordRetriever] = None,
        *,
        dense_weight: float = 1.0,
        sparse_weight: float = 0.6,
        doc_penalty: float = 0.1,
        max_per_document: int = 2,
        dense_top_k: int = 12,
        sparse_limit: int = 100,
        min_dense_score: float = 0.0,
        min_sparse_score: float = 0.0,
        min_combined_score: float = 0.0,
        max_total_tokens: Optional[int] = None,
        token_estimator: Optional[Callable[[RagDocumentChunk], int]] = None,
        rerank_fn: Optional[Callable[[str, List["ContextCandidate"]], List["ContextCandidate"]]] = None,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.doc_penalty = max(0.0, doc_penalty)
        self.max_per_document = max(1, max_per_document)
        self.dense_top_k = max(1, dense_top_k)
        self.sparse_limit = max(0, sparse_limit)
        self.min_dense_score = max(0.0, min_dense_score)
        self.min_sparse_score = max(0.0, min_sparse_score)
        self.min_combined_score = max(0.0, min_combined_score)
        self.max_total_tokens = max_total_tokens
        self.token_estimator = token_estimator or self._default_token_estimator
        self.rerank_fn = rerank_fn

    def select(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        top_k: int = 6,
        languages: Optional[Sequence[str]] = None,
        max_total_tokens: Optional[int] = None,
    ) -> List[ContextCandidate]:
        dense_hits = self.vector_retriever.search_by_vector(
            query_vector,
            top_k=self.dense_top_k,
        )
        if languages:
            languages_set = {lang.lower() for lang in languages}
            dense_hits = [
                hit
                for hit in dense_hits
                if (hit.chunk.lang_primary or "").lower() in languages_set
            ]
        sparse_hits: List[KeywordMatch] = []
        if self.keyword_retriever and self.sparse_limit > 0:
            sparse_hits = self.keyword_retriever.search(
                query,
                languages=languages,
                limit=self.sparse_limit,
                max_per_document=self.max_per_document * 2,
            )
        candidates = self._combine_hits(dense_hits, sparse_hits, top_k=top_k, query=query)
        token_limit = max_total_tokens if max_total_tokens is not None else self.max_total_tokens
        if token_limit:
            candidates = self._cap_tokens(candidates, token_limit)
        return candidates

    def _combine_hits(
        self,
        dense_hits: Sequence[RetrievedChunk],
        sparse_hits: Sequence[KeywordMatch],
        *,
        top_k: int,
        query: str,
    ) -> List[ContextCandidate]:
        candidates: Dict[int, ContextCandidate] = {}

        for hit in dense_hits:
            chunk = hit.chunk
            doc = hit.document or getattr(chunk, "document", None)
            cand = candidates.setdefault(
                chunk.id,
                ContextCandidate(
                    chunk=chunk,
                    document=doc,
                ),
            )
            cand.dense_score = max(cand.dense_score, hit.score)

        for match in sparse_hits:
            chunk = match.chunk
            doc = match.document or getattr(chunk, "document", None)
            cand = candidates.setdefault(
                chunk.id,
                ContextCandidate(
                    chunk=chunk,
                    document=doc,
                ),
            )
            cand.sparse_score = max(cand.sparse_score, match.score)
            if match.matched_terms:
                cand.matched_terms = match.matched_terms

        filtered: List[ContextCandidate] = []
        for cand in candidates.values():
            cand.combined_score = (
                self.dense_weight * cand.dense_score
                + self.sparse_weight * cand.sparse_score
            )
            cand.reasoning_hint = self._build_reasoning_hint(cand)
            if not self._passes_thresholds(cand):
                continue
            filtered.append(cand)

        if not filtered:
            return []

        sorted_candidates = sorted(
            filtered,
            key=lambda item: (item.combined_score, item.dense_score, item.chunk.id),
            reverse=True,
        )

        selected: List[ContextCandidate] = []
        doc_counts: Dict[Optional[int], int] = {}
        seen_ids: set[int] = set()
        for cand in sorted_candidates:
            doc = cand.document
            doc_id = doc.id if doc is not None else None
            count_for_doc = doc_counts.get(doc_id, 0)
            if count_for_doc >= self.max_per_document:
                continue
            cand.adjusted_score = cand.combined_score - self.doc_penalty * count_for_doc
            selected.append(cand)
            doc_counts[doc_id] = count_for_doc + 1
            seen_ids.add(cand.chunk.id)
            if len(selected) >= top_k:
                break

        if len(selected) < top_k:
            for cand in sorted_candidates:
                if cand.chunk.id in seen_ids:
                    continue
                cand.adjusted_score = cand.combined_score
                selected.append(cand)
                seen_ids.add(cand.chunk.id)
                if len(selected) >= top_k:
                    break
        selected = selected[:top_k]

        did_rerank = False
        if self.rerank_fn and selected:
            try:
                reranked = self.rerank_fn(query, selected.copy())
                if isinstance(reranked, list) and reranked:
                    selected = reranked
                    did_rerank = True
            except Exception:
                # Если реранкер упал, возвращаем исходный порядок
                pass

        if not did_rerank:
            selected.sort(key=lambda item: (item.adjusted_score, item.combined_score), reverse=True)
        return selected

    @staticmethod
    def _build_reasoning_hint(cand: ContextCandidate) -> str:
        parts: List[str] = []
        if cand.dense_score:
            parts.append(f"dense={cand.dense_score:.3f}")
        if cand.sparse_score:
            if cand.matched_terms:
                terms = ", ".join(cand.matched_terms[:4])
                parts.append(f"keywords={terms}")
            parts.append(f"sparse={cand.sparse_score:.3f}")
        preview = (cand.chunk.preview or "")[:120].replace("\n", " ").strip()
        if preview:
            parts.append(f"preview=\"{preview}\"")
        return "; ".join(parts)

    def _passes_thresholds(self, cand: ContextCandidate) -> bool:
        if cand.dense_score and cand.dense_score < self.min_dense_score:
            return False
        if cand.sparse_score and cand.sparse_score < self.min_sparse_score:
            return False
        if cand.combined_score < self.min_combined_score:
            return False
        return True

    def _cap_tokens(self, candidates: List[ContextCandidate], limit: int) -> List[ContextCandidate]:
        if limit <= 0 or not candidates:
            return candidates
        total = 0
        kept: List[ContextCandidate] = []
        for cand in candidates:
            tokens = max(1, self.token_estimator(cand.chunk))
            if total and total + tokens > limit:
                continue
            kept.append(cand)
            total += tokens
            if total >= limit:
                break
        return kept if kept else candidates[:1]

    @staticmethod
    def _default_token_estimator(chunk: RagDocumentChunk) -> int:
        tok = getattr(chunk, "token_count", None)
        if tok:
            try:
                return max(1, int(tok))
            except Exception:
                pass
        text = (chunk.content or "").strip()
        if not text:
            return 50
        words = len(text.split())
        return max(1, int(words * 1.3))

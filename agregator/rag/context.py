from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from models import RagDocument, RagDocumentChunk
from .retrieval import RetrievedChunk, VectorRetriever
from .sparse import KeywordMatch, KeywordRetriever

QUERY_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9ёЁ]+")


@dataclass(slots=True)
class ContextCandidate:
    chunk: RagDocumentChunk
    document: Optional[RagDocument]
    preview: str = ""
    token_estimate: int = 0
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    adjusted_score: float = 0.0
    matched_terms: List[str] = field(default_factory=list)
    reasoning_hint: str = ""
    section_path: str = ""
    structure_bonus: float = 0.0
    metadata_bonus: float = 0.0
    metadata_hits: List[str] = field(default_factory=list)
    authority_bonus: float = 0.0


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
        token_multiplier: float = 1.6,
        section_overhead: int = 80,
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
        self.token_multiplier = max(1.0, float(token_multiplier))
        self.section_overhead = max(0, int(section_overhead))

    def select(
        self,
        query: str,
        query_vector: Sequence[float],
        *,
        top_k: int = 6,
        languages: Optional[Sequence[str]] = None,
        max_total_tokens: Optional[int] = None,
        allowed_document_ids: Optional[Sequence[int]] = None,
        precomputed_dense_hits: Optional[Sequence[RetrievedChunk]] = None,
        query_terms: Optional[Sequence[str]] = None,
        authority_scores: Optional[Dict[int, float]] = None,
    ) -> List[ContextCandidate]:
        if precomputed_dense_hits is not None:
            dense_hits = list(precomputed_dense_hits)
        else:
            dense_hits = self.vector_retriever.search_by_vector(
                query_vector,
                top_k=self.dense_top_k,
                allowed_document_ids=allowed_document_ids,
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
                query_terms=query_terms,
            )
        candidates = self._combine_hits(
            dense_hits,
            sparse_hits,
            top_k=top_k,
            query=query,
            authority_scores=authority_scores,
        )
        for cand in candidates:
            self._estimate_tokens(cand)
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
        authority_scores: Optional[Dict[int, float]] = None,
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
                    preview=self._make_preview(chunk),
                    section_path=(chunk.section_path or "").strip(),
                ),
            )
            cand.dense_score = max(cand.dense_score, hit.score)
            if not cand.section_path and getattr(chunk, "section_path", None):
                cand.section_path = (chunk.section_path or "").strip()
            if doc and doc.file_id and authority_scores:
                cand.authority_bonus = max(cand.authority_bonus, float(authority_scores.get(doc.file_id, 0.0)))

        for match in sparse_hits:
            chunk = match.chunk
            doc = match.document or getattr(chunk, "document", None)
            cand = candidates.setdefault(
                chunk.id,
                ContextCandidate(
                    chunk=chunk,
                    document=doc,
                    preview=self._make_preview(chunk),
                    section_path=(chunk.section_path or "").strip(),
                ),
            )
            if not cand.preview:
                cand.preview = self._make_preview(chunk)
            if not cand.section_path and getattr(chunk, "section_path", None):
                cand.section_path = (chunk.section_path or "").strip()
            cand.sparse_score = max(cand.sparse_score, match.score)
            if match.matched_terms:
                cand.matched_terms = match.matched_terms
            if doc and doc.file_id and authority_scores:
                cand.authority_bonus = max(cand.authority_bonus, float(authority_scores.get(doc.file_id, 0.0)))

        filtered: List[ContextCandidate] = []
        query_terms = self._query_terms(query)
        for cand in candidates.values():
            self._enrich_candidate(cand, query_terms)
            cand.combined_score = (
                self.dense_weight * cand.dense_score
                + self.sparse_weight * cand.sparse_score
                + cand.structure_bonus
                + cand.metadata_bonus
                + cand.authority_bonus
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
        if cand.structure_bonus:
            parts.append(f"struct=+{cand.structure_bonus:.3f}")
        if cand.metadata_bonus:
            parts.append(f"meta=+{cand.metadata_bonus:.3f}")
            if cand.metadata_hits:
                parts.append(f"hits={', '.join(cand.metadata_hits[:3])}")
        if cand.authority_bonus:
            parts.append(f"authority=+{cand.authority_bonus:.3f}")
        if cand.section_path:
            section = cand.section_path.replace("\n", " ").strip()
            if len(section) > 80:
                section = section[:77] + "…"
            parts.append(f"section=\"{section}\"")
        preview = (cand.chunk.preview or "")[:120].replace("\n", " ").strip()
        if preview:
            parts.append(f"preview=\"{preview}\"")
        return "; ".join(parts)

    @staticmethod
    def _make_preview(chunk: RagDocumentChunk) -> str:
        raw = (chunk.preview or chunk.content or "").strip()
        if not raw:
            return ""
        return raw[:200].replace("\n", " ").strip()

    @staticmethod
    def _parse_chunk_meta(chunk: RagDocumentChunk) -> Dict[str, object]:
        meta_raw = getattr(chunk, "meta", None)
        if not meta_raw:
            return {}
        if isinstance(meta_raw, dict):
            return meta_raw
        try:
            return json.loads(meta_raw)
        except Exception:
            return {}

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        seen: set[str] = set()
        terms: List[str] = []
        for match in QUERY_TOKEN_RE.finditer(query or ""):
            token = match.group(0).lower()
            if token and token not in seen:
                seen.add(token)
                terms.append(token)
        return terms

    def _structure_bonus(self, cand: ContextCandidate, meta: Dict[str, object]) -> float:
        bonus = 0.0
        distance = meta.get("heading_distance")
        try:
            dist_val = max(0, int(distance))
            bonus += max(0.0, 0.12 - 0.02 * dist_val)
        except Exception:
            pass
        level = meta.get("heading_level")
        try:
            level_val = max(1, int(level))
            bonus += max(0.0, 0.08 * (3 - min(level_val, 3)))
        except Exception:
            pass
        if cand.section_path:
            bonus += 0.02
        return max(0.0, bonus)

    def _metadata_bonus(self, cand: ContextCandidate, query_terms: Sequence[str]) -> tuple[float, List[str]]:
        doc = cand.document or getattr(cand.chunk, "document", None)
        if doc is None:
            return 0.0, []
        file_obj = getattr(doc, "file", None)
        if file_obj is None or not query_terms:
            return 0.0, []
        haystack: List[str] = []

        def _push(value: Optional[str]) -> None:
            if not value:
                return
            haystack.append(str(value).lower())

        _push(getattr(file_obj, "title", None))
        _push(getattr(file_obj, "keywords", None))
        _push(getattr(file_obj, "material_type", None))
        collection = getattr(file_obj, "collection", None)
        if collection is not None:
            _push(getattr(collection, "name", None))
            _push(getattr(collection, "slug", None))
        tags = getattr(file_obj, "tags", None) or []
        for tag in tags:
            _push(getattr(tag, "key", None))
            _push(getattr(tag, "value", None))
        haystack = [item for item in haystack if item]
        if not haystack:
            return 0.0, []

        matched: List[str] = []
        for term in query_terms:
            if not term:
                continue
            for item in haystack:
                if term in item:
                    matched.append(term)
                    break
        if not matched:
            return 0.0, []
        unique_hits = list(dict.fromkeys(matched))
        bonus = min(0.25, 0.06 * len(unique_hits))
        return bonus, unique_hits

    def _enrich_candidate(self, cand: ContextCandidate, query_terms: Sequence[str]) -> None:
        meta = self._parse_chunk_meta(cand.chunk)
        if not cand.section_path:
            path = meta.get("heading_path")
            if isinstance(path, list):
                cand.section_path = " / ".join(str(part) for part in path if part).strip()
        cand.structure_bonus = self._structure_bonus(cand, meta)
        meta_bonus, hits = self._metadata_bonus(cand, query_terms)
        cand.metadata_bonus = meta_bonus
        cand.metadata_hits = hits

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
            tokens = cand.token_estimate or self._estimate_tokens(cand)
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
                return max(1, int(tok) * 3 // 2)
            except Exception:
                pass
        text = (chunk.content or "").strip()
        if not text:
            return 50
        words = len(text.split())
        return max(1, int(words * 1.6))

    def _estimate_tokens(self, cand: ContextCandidate) -> int:
        try:
            raw_tokens = max(1, int(self.token_estimator(cand.chunk)))
        except Exception:
            raw_tokens = max(1, len((cand.chunk.content or "").split()))
        adjusted = int(raw_tokens * self.token_multiplier) + self.section_overhead
        cand.token_estimate = max(1, adjusted)
        return cand.token_estimate

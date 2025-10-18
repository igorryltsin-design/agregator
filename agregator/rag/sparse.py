from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, TYPE_CHECKING

from sqlalchemy import or_, text
from sqlalchemy.orm import Session, joinedload

from models import RagDocument, RagDocumentChunk, db

if TYPE_CHECKING:  # pragma: no cover - только для подсказок типов
    from agregator.services.search import SearchService  # noqa: F401

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9ёЁ]+" )


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


@dataclass(slots=True)
class KeywordMatch:
    chunk: RagDocumentChunk
    document: Optional[RagDocument]
    score: float
    matched_terms: List[str] = field(default_factory=list)
    total_terms: int = 0


class KeywordRetriever:
    """Простейший поисковик по ключевым словам для fallback/комбинированной выдачи."""

    def __init__(
        self,
        session: Optional[Session] = None,
        *,
        limit: int = 200,
        min_term_length: int = 3,
        search_service: "SearchService" | None = None,
        expand_terms_fn: Optional[Callable[[Sequence[str]], Sequence[str]]] = None,
        lemma_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.session: Session = session or db.session  # type: ignore[assignment]
        self.limit = max(1, limit)
        self.min_term_length = max(1, min_term_length)
        self.search_service = search_service
        self.expand_terms_fn = expand_terms_fn
        self.lemma_fn = lemma_fn

    def search(
        self,
        query: str,
        *,
        languages: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        max_per_document: Optional[int] = None,
    ) -> List[KeywordMatch]:
        raw_terms = [t for t in _tokenize(query) if len(t) >= self.min_term_length]
        if not raw_terms:
            return []
        primary_terms: List[str] = raw_terms
        if self.lemma_fn:
            try:
                primary_terms = [self.lemma_fn(token) or token for token in primary_terms]
            except Exception:
                primary_terms = raw_terms
        primary_terms = [term for term in primary_terms if term]
        if not primary_terms:
            return []
        primary_terms = list(dict.fromkeys(primary_terms))
        original_terms = list(dict.fromkeys(raw_terms))
        expanded_terms: List[str] = []
        if self.expand_terms_fn:
            try:
                expanded_terms = [
                    term for term in self.expand_terms_fn(primary_terms) if term and len(term) >= self.min_term_length
                ]
            except Exception:
                expanded_terms = []
        weights: Dict[str, float] = {}
        combined_order: List[str] = []
        for term in primary_terms:
            if term not in weights:
                weights[term] = 1.0
                combined_order.append(term)
        for term in original_terms:
            if term not in weights:
                weights[term] = 0.85
                combined_order.append(term)
        for term in expanded_terms:
            if term not in weights:
                weights[term] = 0.6
                combined_order.append(term)
        if not combined_order:
            combined_order = primary_terms
        max_terms = 30
        if len(combined_order) > max_terms:
            combined_order = combined_order[:max_terms]
        weights = {term: weights.get(term, 0.6) for term in combined_order}
        fetch_limit = min(max(self.limit, len(combined_order) * 20), 1000)
        if limit is not None:
            fetch_limit = min(fetch_limit, max(1, int(limit) * 3))
        fts_match_expr = " ".join(f"{term}*" for term in combined_order[:8] if len(term) >= self.min_term_length)
        candidate_files: Dict[int, float] = {}
        if fts_match_expr and self.search_service:
            try:
                rows = self.session.execute(
                    text(
                        "SELECT rowid, bm25(files_fts) AS score "
                        "FROM files_fts WHERE files_fts MATCH :match LIMIT :limit"
                    ),
                    {"match": fts_match_expr, "limit": min(2000, fetch_limit * 2)},
                ).fetchall()
                for row in rows:
                    try:
                        fid = int(row[0])
                    except Exception:
                        continue
                    try:
                        raw_score = float(row[1]) if row[1] is not None else 0.0
                    except Exception:
                        raw_score = 0.0
                    score = 1.0 / (1.0 + max(raw_score, 0.0))
                    candidate_files[fid] = max(candidate_files.get(fid, 0.0), score)
            except Exception:
                candidate_files = {}
        if not candidate_files and self.search_service:
            try:
                ids = self.search_service.candidate_ids(" ".join(combined_order[:8]), limit=fetch_limit)
                if ids:
                    for fid in ids:
                        candidate_files[int(fid)] = candidate_files.get(int(fid), 0.0)
            except Exception:
                pass

        like_conditions = []
        for term in combined_order:
            pattern = f"%{term}%"
            like_conditions.append(
                or_(
                    RagDocumentChunk.content.ilike(pattern),
                    RagDocumentChunk.preview.ilike(pattern),
                    RagDocumentChunk.keywords_top.ilike(pattern),
                )
            )
        if not like_conditions:
            return []
        query_obj = self.session.query(RagDocumentChunk).options(
            joinedload(RagDocumentChunk.document).joinedload(RagDocument.file)
        )
        query_obj = query_obj.filter(or_(*like_conditions))
        if candidate_files:
            query_obj = query_obj.join(RagDocument, RagDocument.id == RagDocumentChunk.document_id)
            query_obj = query_obj.filter(RagDocument.file_id.in_(candidate_files.keys()))
        if languages:
            query_obj = query_obj.filter(RagDocumentChunk.lang_primary.in_(languages))
        rows = query_obj.limit(fetch_limit).all()
        if not rows:
            return []
        matches: List[KeywordMatch] = []
        primary_weight_total = sum(weights.get(term, 1.0) for term in primary_terms) or float(len(primary_terms))
        for chunk in rows:
            content = (chunk.content or "").lower()
            if not content:
                content = ""
            preview = (chunk.preview or "").lower()
            keywords_text = (chunk.keywords_top or "").lower()
            matched: List[str] = []
            weight_sum = 0.0
            for term in combined_order:
                if term in content or term in preview or term in keywords_text:
                    matched.append(term)
                    weight = weights.get(term, 0.6)
                    weight_sum += weight
            if not matched:
                continue
            normalized = min(1.0, weight_sum / max(1.0, primary_weight_total))
            doc = getattr(chunk, "document", None)
            file_id = getattr(doc, "file_id", None) if doc else None
            bm25_boost = 0.0
            if file_id is not None and candidate_files:
                bm25_boost = candidate_files.get(int(file_id), 0.0)
            if bm25_boost > 0:
                score = 0.7 * normalized + 0.3 * bm25_boost
            else:
                score = normalized
            matches.append(
                KeywordMatch(
                    chunk=chunk,
                    document=getattr(chunk, "document", None),
                    score=score,
                    matched_terms=matched,
                    total_terms=len(primary_terms),
                )
            )
        if not matches:
            return []
        matches.sort(key=lambda item: (item.score, item.chunk.id), reverse=True)
        limited: List[KeywordMatch] = []
        doc_counts: dict[Optional[int], int] = {}
        result_limit = max(1, limit or self.limit)
        for item in matches:
            doc = item.document
            doc_id = doc.id if doc is not None else None
            if max_per_document is not None and doc_id is not None:
                if doc_counts.get(doc_id, 0) >= max_per_document:
                    continue
            limited.append(item)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if len(limited) >= result_limit:
                break
        return limited

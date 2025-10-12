from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from models import RagDocument, RagDocumentChunk, db

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
    ) -> None:
        self.session: Session = session or db.session  # type: ignore[assignment]
        self.limit = max(1, limit)
        self.min_term_length = max(1, min_term_length)

    def search(
        self,
        query: str,
        *,
        languages: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
        max_per_document: Optional[int] = None,
    ) -> List[KeywordMatch]:
        terms = [t for t in _tokenize(query) if len(t) >= self.min_term_length]
        terms = sorted(set(terms))
        if not terms:
            return []
        conditions = [RagDocumentChunk.content.ilike(f"%{term}%") for term in terms]
        if not conditions:
            return []
        fetch_limit = min(max(self.limit, len(terms) * 20), 1000)
        if limit is not None:
            fetch_limit = min(fetch_limit, max(1, int(limit) * 3))
        query_obj = (
            self.session.query(RagDocumentChunk)
            .options(joinedload(RagDocumentChunk.document).joinedload(RagDocument.file))
            .filter(or_(*conditions))
        )
        if languages:
            query_obj = query_obj.filter(RagDocumentChunk.lang_primary.in_(languages))
        rows = query_obj.limit(fetch_limit).all()
        if not rows:
            return []
        matches: List[KeywordMatch] = []
        total_terms = len(terms)
        for chunk in rows:
            content = (chunk.content or "").lower()
            if not content:
                continue
            matched = [term for term in terms if term in content]
            if not matched:
                continue
            score = len(matched) / total_terms
            matches.append(
                KeywordMatch(
                    chunk=chunk,
                    document=getattr(chunk, "document", None),
                    score=score,
                    matched_terms=matched,
                    total_terms=total_terms,
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

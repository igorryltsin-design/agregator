from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from models import (
    File,
    RagDocument,
    RagDocumentChunk,
    RagDocumentVersion,
    RagIngestFailure,
    db,
)

logger = logging.getLogger(__name__)

RU_STOPWORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
}

EN_STOPWORDS = {
    "the",
    "and",
    "that",
    "have",
    "for",
    "not",
    "with",
    "you",
    "this",
    "but",
    "his",
    "from",
    "they",
    "say",
    "her",
    "she",
    "will",
    "one",
    "all",
    "would",
    "there",
    "their",
    "what",
    "about",
    "which",
    "when",
    "make",
    "can",
    "like",
    "time",
    "just",
    "him",
    "know",
}

GENERIC_STOPWORDS = {
    "—",
    "–",
    "-",
    "—",
    "…",
    "©",
    "®",
}


@dataclass(slots=True)
class ChunkConfig:
    max_tokens: int = 700
    overlap: int = 120
    min_tokens: int = 80
    preview_chars: int = 280
    keyword_count: int = 8


@dataclass(slots=True)
class ChunkData:
    ordinal: int
    text: str
    token_count: int
    char_count: int
    preview: str
    keywords: List[str]
    lang_primary: str
    content_hash: str
    section_path: Optional[str]
    meta: Dict[str, int | str | bool | float]


def normalize_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))
    while "\n\n\n" in normalized:
        normalized = normalized.replace("\n\n\n", "\n\n")
    return normalized.strip()


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    sample = text[:2000]
    cyr = sum("а" <= ch.lower() <= "я" or ch == "ё" for ch in sample)
    lat = sum("a" <= ch.lower() <= "z" for ch in sample)
    if cyr == 0 and lat == 0:
        return "unknown"
    if cyr > lat * 1.2:
        return "ru"
    if lat > cyr * 1.2:
        return "en"
    return "mixed"


def _tokenize_for_keywords(text: str) -> List[str]:
    tokens = []
    current = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            current.append(ch.lower())
        else:
            if current:
                tokens.append("".join(current))
                current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def extract_keywords(text: str, lang: str, limit: int = 8) -> List[str]:
    tokens = _tokenize_for_keywords(text)
    if not tokens:
        return []
    stopwords: set[str] = set(GENERIC_STOPWORDS)
    if lang == "ru":
        stopwords |= RU_STOPWORDS
    elif lang == "en":
        stopwords |= EN_STOPWORDS
    else:
        stopwords |= RU_STOPWORDS | EN_STOPWORDS
    counts: Dict[str, int] = {}
    for token in tokens:
        if len(token) < 3 or token in stopwords or token.isnumeric():
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:limit]]


def chunk_text(text: str, *, config: Optional[ChunkConfig] = None) -> List[ChunkData]:
    if not text:
        return []
    cfg = config or ChunkConfig()
    tokens = text.split()
    if not tokens:
        return []
    max_tokens = max(cfg.max_tokens, 1)
    overlap = min(cfg.overlap, max_tokens - 1) if max_tokens > 1 else 0
    step = max(max_tokens - overlap, 1)
    chunks: List[ChunkData] = []
    total_tokens = len(tokens)
    ordinal = 0
    for start in range(0, total_tokens, step):
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        text_chunk = " ".join(chunk_tokens).strip()
        if not text_chunk:
            continue
        token_count = len(chunk_tokens)
        if token_count < cfg.min_tokens and ordinal > 0:
            prev = chunks[-1]
            merged = f"{prev.text} {text_chunk}".strip()
            chunks[-1] = ChunkData(
                ordinal=prev.ordinal,
                text=merged,
                token_count=prev.token_count + token_count,
                char_count=len(merged),
                preview=(merged[: cfg.preview_chars]).strip(),
                keywords=extract_keywords(merged, detect_language(merged), cfg.keyword_count),
                lang_primary=detect_language(merged),
                content_hash=sha256(merged.encode("utf-8")).hexdigest(),
                section_path=None,
                meta={
                    **prev.meta,
                    "token_end": end,
                    "merged_with_next": True,
                },
            )
            continue
        ordinal += 1
        lang = detect_language(text_chunk)
        keywords = extract_keywords(text_chunk, lang, cfg.keyword_count)
        chunks.append(
            ChunkData(
                ordinal=ordinal,
                text=text_chunk,
                token_count=token_count,
                char_count=len(text_chunk),
                preview=(text_chunk[: cfg.preview_chars]).strip(),
                keywords=keywords,
                lang_primary=lang,
                content_hash=sha256(text_chunk.encode("utf-8")).hexdigest(),
                section_path=None,
                meta={
                    "token_start": start,
                    "token_end": end,
                    "overlap": overlap,
                    "total_tokens": total_tokens,
                },
            )
        )
    return chunks


def _safe_json_dumps(data: Dict) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return json.dumps({"__fallback__": repr(data)}, ensure_ascii=False)


class RagIndexer:
    def __init__(
        self,
        session: Optional[Session] = None,
        *,
        chunk_config: Optional[ChunkConfig] = None,
        normalizer_version: str = "v1",
    ) -> None:
        self.session: Session = session or db.session  # type: ignore[assignment]
        self.chunk_config = chunk_config or ChunkConfig()
        self.normalizer_version = normalizer_version

    def ingest_document(
        self,
        file_obj: File,
        raw_text: str,
        *,
        metadata: Optional[Dict[str, object]] = None,
        skip_if_unchanged: bool = True,
        commit: bool = True,
    ) -> Dict[str, object]:
        if not isinstance(file_obj, File):
            raise TypeError("file_obj must be models.File instance")
        metadata = metadata or {}

        clean_text = normalize_text(raw_text or "")
        raw_hash = sha256((raw_text or "").encode("utf-8")).hexdigest() if raw_text is not None else None
        dedupe_hash = sha256(clean_text.encode("utf-8")).hexdigest() if clean_text else None
        lang = detect_language(clean_text)

        document = file_obj.rag_document
        if document is None:
            document = RagDocument(file_id=file_obj.id)
            self.session.add(document)

        last_version = document.versions[-1] if document.versions else None
        if skip_if_unchanged and last_version and last_version.dedupe_hash and last_version.dedupe_hash == dedupe_hash:
            document.import_status = "up_to_date"
            document.last_indexed_at = datetime.utcnow()
            if commit:
                self.session.commit()
            else:
                self.session.flush()
            return {
                "skipped": True,
                "reason": "unchanged",
                "document_id": document.id,
                "version": last_version.version,
            }

        next_version = (document.latest_version or 0) + 1
        version = RagDocumentVersion(
            document=document,
            version=next_version,
            sha256=raw_hash,
            dedupe_hash=dedupe_hash,
            normalizer_version=self.normalizer_version,
            metadata_json=_safe_json_dumps(metadata),
            raw_text=raw_text,
            clean_text=clean_text,
            lang_primary=lang,
        )
        self.session.add(version)

        chunks = chunk_text(clean_text, config=self.chunk_config)
        version.chunk_count = len(chunks)

        document.latest_version = next_version
        document.lang_primary = lang
        document.import_status = "ready" if chunks else "empty"
        document.is_ready_for_rag = bool(chunks)
        document.last_indexed_at = datetime.utcnow()

        for chunk in chunks:
            chunk_record = RagDocumentChunk(
                document=document,
                version=version,
                ordinal=chunk.ordinal,
                section_path=chunk.section_path,
                token_count=chunk.token_count,
                char_count=chunk.char_count,
                content=chunk.text,
                content_hash=chunk.content_hash,
                preview=chunk.preview,
                keywords_top=", ".join(chunk.keywords),
                lang_primary=chunk.lang_primary,
                meta=_safe_json_dumps(chunk.meta),
            )
            self.session.add(chunk_record)

        try:
            if commit:
                self.session.commit()
            else:
                self.session.flush()
        except Exception as exc:
            logger.exception("Failed to ingest document %s: %s", file_obj.id, exc)
            self.session.rollback()
            failure = RagIngestFailure(
                file_id=file_obj.id,
                stage="ingest",
                error=str(exc),
                meta=_safe_json_dumps(
                    {
                        "normalizer_version": self.normalizer_version,
                        "document_id": document.id if document.id else None,
                    }
                ),
            )
            self.session.add(failure)
            self.session.commit()
            raise

        return {
            "skipped": False,
            "document_id": document.id,
            "version_id": version.id,
            "version": version.version,
            "chunks": len(chunks),
            "language": lang,
            "dedupe_hash": dedupe_hash,
        }

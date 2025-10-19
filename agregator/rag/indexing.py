from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_HEADING_NUMERIC_RE = re.compile(r"^(\d+(?:\.\d+)*)[\s).:-]")

try:  # pragma: no cover - optional dependency
    from razdel import sentenize as _sentenize  # type: ignore
except Exception:  # pragma: no cover
    _sentenize = None


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


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    if _sentenize:
        try:
            return [item.text.strip() for item in _sentenize(text) if item.text.strip()]
        except Exception:
            pass
    return [sent.strip() for sent in _SENTENCE_SPLIT_RE.split(text) if sent.strip()]


@dataclass(slots=True)
class SegmentInfo:
    text: str
    tokens: List[str]
    path: Tuple[str, ...]
    heading_level: Optional[int]
    is_heading: bool
    distance_from_heading: int = 0


def _guess_heading_level(line: str) -> Optional[int]:
    stripped = line.strip()
    if not stripped:
        return None
    numeric = _HEADING_NUMERIC_RE.match(stripped)
    if numeric:
        parts = numeric.group(1).split(".")
        return max(1, min(len(parts), 5))
    alpha_total = sum(1 for ch in stripped if ch.isalpha())
    if stripped.endswith(":") and alpha_total >= 3:
        return 2
    if alpha_total and alpha_total <= 120:
        upper_total = sum(1 for ch in stripped if ch.isalpha() and ch.isupper())
        if upper_total / max(alpha_total, 1) >= 0.65:
            return 1
    return None


def _segment_text(text: str) -> List[SegmentInfo]:
    """Разбивает текст на сегменты с учётом заголовков и их иерархии."""
    if not text:
        return []

    segments: List[SegmentInfo] = []
    path: List[str] = []
    pending_lines: List[str] = []

    def _flush_pending() -> None:
        if not pending_lines:
            return
        joined = " ".join(pending_lines).strip()
        pending_lines.clear()
        if not joined:
            return
        for sentence in _split_sentences(joined):
            tokens = sentence.split()
            if not tokens:
                continue
            segments.append(
                SegmentInfo(
                    text=sentence,
                    tokens=tokens,
                    path=tuple(path),
                    heading_level=len(path) if path else None,
                    is_heading=False,
                )
            )

    paragraphs = re.split(r"\n{2,}", text)
    for paragraph in paragraphs:
        block = paragraph.strip()
        if not block:
            _flush_pending()
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            _flush_pending()
            continue
        for line in lines:
            level = _guess_heading_level(line)
            if level is not None:
                _flush_pending()
                normalized = line.strip().rstrip(":").strip()
                if not normalized:
                    continue
                while len(path) < level:
                    path.append("")
                path[level - 1] = normalized
                del path[level:]
                tokens = normalized.split() or [normalized]
                segments.append(
                    SegmentInfo(
                        text=normalized,
                        tokens=tokens,
                        path=tuple(path),
                        heading_level=level,
                        is_heading=True,
                    )
                )
                continue
            if line.lstrip().startswith(("-", "*", "•")):
                _flush_pending()
                tokens = line.split() or [line]
                segments.append(
                    SegmentInfo(
                        text=line,
                        tokens=tokens,
                        path=tuple(path),
                        heading_level=len(path) if path else None,
                        is_heading=False,
                    )
                )
                continue
            pending_lines.append(line)
        _flush_pending()
    _flush_pending()

    last_heading_idx: Optional[int] = None
    for idx, segment in enumerate(segments):
        if segment.is_heading:
            last_heading_idx = idx
            segment.distance_from_heading = 0
        else:
            segment.distance_from_heading = (
                0 if last_heading_idx is None else idx - last_heading_idx
            )
    return segments


def chunk_text(text: str, *, config: Optional[ChunkConfig] = None) -> List[ChunkData]:
    if not text:
        return []
    cfg = config or ChunkConfig()
    segment_infos = _segment_text(text)
    if not segment_infos:
        fallback = text.strip()
        if not fallback:
            return []
        tokens = fallback.split()
        if not tokens:
            return []
        segment_infos = [
            SegmentInfo(
                text=fallback,
                tokens=tokens,
                path=(),
                heading_level=None,
                is_heading=False,
                distance_from_heading=0,
            )
        ]

    tokens: List[str] = []
    segment_ranges: List[Tuple[int, int, SegmentInfo]] = []
    cursor = 0
    for info in segment_infos:
        seg_tokens = info.tokens or [tok for tok in info.text.split() if tok]
        start_idx = cursor
        if seg_tokens:
            tokens.extend(seg_tokens)
            cursor += len(seg_tokens)
        segment_ranges.append((start_idx, cursor, info))
    total_tokens = len(tokens)
    if total_tokens == 0:
        return []
    breakpoints = sorted({end for _, end, _ in segment_ranges if end > 0} | {total_tokens})

    max_tokens = max(cfg.max_tokens, 1)
    overlap = min(cfg.overlap, max_tokens - 1) if max_tokens > 1 else 0
    step = max(max_tokens - overlap, 1)

    def _segment_span(start_token: int, end_token: int) -> Tuple[int, int, int]:
        if not segment_ranges:
            return (0, 0, 0)
        seg_start = None
        seg_end = None
        count = 0
        for idx, (seg_start_token, seg_end_token, _info) in enumerate(segment_ranges):
            if seg_end_token <= start_token:
                continue
            if seg_start_token >= end_token:
                break
            if seg_start is None:
                seg_start = idx
            seg_end = idx
            count += 1
        if seg_start is None:
            seg_start = 0
        if seg_end is None:
            seg_end = seg_start
        return seg_start, seg_end, count or 1

    def _align_breakpoint(start_token: int, raw_end: int) -> int:
        candidate = raw_end
        within = [bp for bp in breakpoints if start_token < bp <= raw_end]
        if within:
            candidate = within[-1]
        else:
            for bp in breakpoints:
                if bp <= raw_end:
                    continue
                span = bp - start_token
                if span <= max_tokens + max(cfg.min_tokens, 40) or bp == total_tokens:
                    candidate = bp
                    break
        if candidate <= start_token:
            candidate = min(raw_end, total_tokens)
        return min(candidate, total_tokens)

    chunks: List[ChunkData] = []
    ordinal = 0
    start = 0
    while start < total_tokens:
        raw_end = min(start + max_tokens, total_tokens)
        end = _align_breakpoint(start, raw_end)
        if end <= start:
            end = min(raw_end if raw_end > start else total_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        text_chunk = " ".join(chunk_tokens).strip()
        if not text_chunk:
            start = end if end > start else end + 1
            continue

        segment_indices = [
            idx
            for idx, (seg_start_token, seg_end_token, _info) in enumerate(segment_ranges)
            if seg_end_token > start and seg_start_token < end
        ]
        if not segment_indices:
            segment_indices = [len(segment_ranges) - 1]

        if len(chunk_tokens) < cfg.min_tokens and chunks:
            prev = chunks[-1]
            prev_start = int(prev.meta.get("token_start", 0))
            merged_tokens = tokens[prev_start:end]
            merged_text = " ".join(merged_tokens).strip()
            if not merged_text:
                start = end if end > start else end + 1
                continue
            seg_start_idx, seg_end_idx, seg_count = _segment_span(prev_start, end)
            span_infos = [info for _, _, info in segment_ranges[seg_start_idx : seg_end_idx + 1]]
            heading_path: Tuple[str, ...] = ()
            heading_level = None
            heading_distance = None
            for info in reversed(span_infos):
                if info.path:
                    heading_path = info.path
                    heading_level = info.heading_level or len(info.path)
                    break
            if span_infos:
                heading_distance = min(info.distance_from_heading for info in span_infos)
            lang = detect_language(merged_text)
            chunks[-1] = ChunkData(
                ordinal=prev.ordinal,
                text=merged_text,
                token_count=len(merged_tokens),
                char_count=len(merged_text),
                preview=(merged_text[: cfg.preview_chars]).strip(),
                keywords=extract_keywords(merged_text, lang, cfg.keyword_count),
                lang_primary=lang,
                content_hash=sha256(merged_text.encode("utf-8")).hexdigest(),
                section_path=" / ".join(heading_path) if heading_path else prev.section_path,
                meta={
                    **prev.meta,
                    "token_end": end,
                    "merged_with_next": True,
                    "segments": seg_count,
                    "segment_start": seg_start_idx,
                    "segment_end": seg_end_idx,
                    "heading_path": list(heading_path),
                    "heading_level": heading_level,
                    "heading_distance": heading_distance,
                },
            )
            next_start = max(prev_start, end - overlap)
            if next_start <= prev_start:
                next_start = end
            start = min(next_start, total_tokens)
            continue

        ordinal += 1
        seg_start_idx, seg_end_idx, seg_count = _segment_span(start, end)
        span_infos = [info for _, _, info in segment_ranges[seg_start_idx : seg_end_idx + 1]]
        heading_path: Tuple[str, ...] = ()
        heading_level = None
        heading_distance = None
        for info in reversed(span_infos):
            if info.path:
                heading_path = info.path
                heading_level = info.heading_level or len(info.path)
                break
        if span_infos:
            heading_distance = min(info.distance_from_heading for info in span_infos)
        section_path = " / ".join(heading_path) if heading_path else None

        lang = detect_language(text_chunk)
        keywords = extract_keywords(text_chunk, lang, cfg.keyword_count)
        chunks.append(
            ChunkData(
                ordinal=ordinal,
                text=text_chunk,
                token_count=len(chunk_tokens),
                char_count=len(text_chunk),
                preview=(text_chunk[: cfg.preview_chars]).strip(),
                keywords=keywords,
                lang_primary=lang,
                content_hash=sha256(text_chunk.encode("utf-8")).hexdigest(),
                section_path=section_path,
                meta={
                    "token_start": start,
                    "token_end": end,
                    "overlap": overlap,
                    "total_tokens": total_tokens,
                    "segments": seg_count,
                    "segment_start": seg_start_idx,
                    "segment_end": seg_end_idx,
                    "heading_path": list(heading_path),
                    "heading_level": heading_level,
                    "heading_distance": heading_distance,
                },
            )
        )
        if end >= total_tokens:
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = max(0, min(next_start, total_tokens))
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

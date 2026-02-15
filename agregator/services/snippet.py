"""Text snippet extraction and chunking utilities.

Extracted from ``app.py`` – the functions here operate on plain text and
``Path`` objects, so they have no Flask dependency and are easy to test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator, List, Tuple

# ---------------------------------------------------------------------------
# Optional document-format libraries (same fallback pattern as app.py)
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    import docx as _docx
except ImportError:  # pragma: no cover
    _docx = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:  # pragma: no cover
    rtf_to_text = None

try:
    from ebooklib import epub
except ImportError:  # pragma: no cover
    epub = None

try:
    import djvu.decode  # type: ignore
except ImportError:  # pragma: no cover
    djvu = None  # type: ignore

SNIPPET_WINDOW_RADIUS: int = 220


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def collect_snippets(
    text: str,
    terms: List[str],
    max_snips: int = 2,
) -> List[str]:
    """Locate short text windows around *terms* inside *text*.

    Returns up to *max_snips* non-overlapping snippets sorted by position.
    """
    t = text or ""
    if not t:
        return []
    tl = t.lower()
    windows: List[Tuple[int, int]] = []
    for raw in terms:
        term = (raw or "").strip()
        if not term:
            continue
        ql = term.lower()
        pos = 0
        found_any = False
        for _i in range(3):
            idx = tl.find(ql, pos)
            if idx < 0:
                break
            found_any = True
            start = max(0, idx - SNIPPET_WINDOW_RADIUS)
            end = min(len(t), idx + len(term) + SNIPPET_WINDOW_RADIUS)
            windows.append((start, end))
            pos = idx + len(term)
        if not found_any and len(ql) >= 3:
            for part in re.split(r"[\s\-_/]+", ql):
                if len(part) < 3:
                    continue
                idx = tl.find(part)
                if idx >= 0:
                    start = max(0, idx - SNIPPET_WINDOW_RADIUS)
                    end = min(len(t), idx + len(part) + SNIPPET_WINDOW_RADIUS)
                    windows.append((start, end))

    # merge overlapping windows
    windows.sort()
    merged: List[List[int]] = []
    for w in windows:
        if not merged or w[0] > merged[-1][1] + 40:
            merged.append(list(w))
        else:
            merged[-1][1] = max(merged[-1][1], w[1])

    outs: List[Tuple[int, str]] = []
    for a, b in merged[:max_snips]:
        snip = t[a:b]
        snip = re.sub(r"\s+", " ", snip).strip()
        outs.append((a, snip))
    outs.sort(key=lambda x: x[0])
    return [s for _pos, s in outs]


def split_text_chunks(
    text: str, chunk_chars: int, max_chunks: int
) -> List[str]:
    """Split *text* into fixed-size chunks."""
    if not text:
        return []
    chunk_chars = max(256, int(chunk_chars or 1024))
    max_chunks = max(1, int(max_chunks or 1))
    limit = min(len(text), chunk_chars * max_chunks)
    trimmed = text[:limit]
    return [trimmed[i : i + chunk_chars] for i in range(0, len(trimmed), chunk_chars)]


def iter_document_chunks(
    path: Path,
    chunk_chars: int = 6000,
    max_chunks: int = 30,
    *,
    # Optional extraction fallbacks – callers may pass these from app.py
    extract_text_pdf=None,
    extract_text_docx=None,
    extract_text_rtf=None,
    extract_text_epub=None,
    extract_text_djvu=None,
) -> Generator[str, None, None]:
    """Yield text chunks from a document file at *path*.

    Supports PDF, DOCX, RTF, EPUB, DJVU, and plain-text formats.
    """
    if not path or not path.exists() or not path.is_file():
        return
    chunk_chars = max(512, int(chunk_chars or 2000))
    max_chunks = max(1, int(max_chunks or 1))
    ext = path.suffix.lower()
    yielded = 0

    try:
        # ---- plain text ----
        if ext in {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log"}:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                while yielded < max_chunks:
                    chunk = fh.read(chunk_chars)
                    if not chunk:
                        break
                    yielded += 1
                    yield chunk
            return

        # ---- PDF ----
        if ext == ".pdf" and fitz is not None:
            with fitz.open(path) as doc:
                buffer = ""
                for page in doc:
                    buffer += page.get_text("text") or ""
                    while len(buffer) >= chunk_chars and yielded < max_chunks:
                        yield buffer[:chunk_chars]
                        buffer = buffer[chunk_chars:]
                        yielded += 1
                        if yielded >= max_chunks:
                            return
                if buffer and yielded < max_chunks:
                    yield buffer[:chunk_chars]
            return

        # ---- DOCX ----
        if ext == ".docx" and _docx is not None:
            document = _docx.Document(str(path))
            buffer = ""
            for para in document.paragraphs:
                buffer += (para.text or "") + "\n"
                while len(buffer) >= chunk_chars and yielded < max_chunks:
                    yield buffer[:chunk_chars]
                    buffer = buffer[chunk_chars:]
                    yielded += 1
                    if yielded >= max_chunks:
                        return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return

        # ---- RTF ----
        if ext == ".rtf" and rtf_to_text is not None:
            text_content = rtf_to_text(
                path.read_text(encoding="utf-8", errors="ignore")
            )
            for chunk in split_text_chunks(text_content, chunk_chars, max_chunks):
                yield chunk
            return

        # ---- EPUB ----
        if ext == ".epub" and epub is not None:
            book = epub.read_epub(str(path))
            buffer = ""
            for item in book.get_items():
                if item.get_type() != epub.ITEM_DOCUMENT:
                    continue
                buffer += item.get_content().decode(errors="ignore")
                while len(buffer) >= chunk_chars and yielded < max_chunks:
                    yield buffer[:chunk_chars]
                    buffer = buffer[chunk_chars:]
                    yielded += 1
                    if yielded >= max_chunks:
                        return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return

        # ---- DJVU ----
        if ext == ".djvu" and djvu is not None:
            with djvu.decode.open(str(path)) as d:
                buffer = ""
                for page in d.pages:
                    buffer += page.get_text()
                    while len(buffer) >= chunk_chars and yielded < max_chunks:
                        yield buffer[:chunk_chars]
                        buffer = buffer[chunk_chars:]
                        yielded += 1
                        if yielded >= max_chunks:
                            return
            if buffer and yielded < max_chunks:
                yield buffer[:chunk_chars]
            return

    except Exception:
        pass

    # ---- fallback: use caller-provided extraction functions ----
    text_content = ""
    try:
        extractors = {
            ".pdf": extract_text_pdf,
            ".docx": extract_text_docx,
            ".rtf": extract_text_rtf,
            ".epub": extract_text_epub,
            ".djvu": extract_text_djvu,
        }
        extractor = extractors.get(ext)
        if extractor:
            text_content = extractor(path, limit_chars=chunk_chars * max_chunks) or ""
        else:
            text_content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text_content = ""
    for chunk in split_text_chunks(text_content, chunk_chars, max_chunks):
        yield chunk

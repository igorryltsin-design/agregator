from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple

CITATION_RE = re.compile(r"\[(\d+):(\d+)\]")


@dataclass(slots=True)
class ValidationResult:
    is_empty: bool = False
    missing_citations: bool = False
    unknown_citations: List[Tuple[int, int]] = field(default_factory=list)
    extra_citations: List[Tuple[int, int]] = field(default_factory=list)
    hallucination_warning: bool = False
    facts_with_issues: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "is_empty": self.is_empty,
            "missing_citations": self.missing_citations,
            "unknown_citations": [list(item) for item in self.unknown_citations],
            "extra_citations": [list(item) for item in self.extra_citations],
            "hallucination_warning": self.hallucination_warning,
            "facts_with_issues": list(self.facts_with_issues),
        }


def extract_citations(text: str) -> List[Tuple[int, int]]:
    return [(int(doc), int(chunk)) for doc, chunk in CITATION_RE.findall(text)]


def validate_answer(
    answer_text: str,
    allowed_references: Sequence[Tuple[int, int]],
    *,
    require_facts: bool = True,
) -> ValidationResult:
    """Проверяет ответ на наличие цитат и соответствие контексту."""
    allowed_set: Set[Tuple[int, int]] = {(int(doc), int(chunk)) for doc, chunk in allowed_references}
    result = ValidationResult()
    normalized = (answer_text or "").strip()
    if not normalized:
        result.is_empty = True
        result.hallucination_warning = True
        return result

    facts_block = _extract_block(normalized, "факты")
    if not facts_block:
        if require_facts:
            result.missing_citations = True
            result.hallucination_warning = True
        return result

    facts_lines = [line.strip() for line in facts_block.splitlines() if line.strip()]

    missing_citations = False
    unknown_refs: List[Tuple[int, int]] = []
    for line in facts_lines:
        if not line.startswith("-"):
            continue
        citations = extract_citations(line)
        if not citations:
            missing_citations = True
            result.facts_with_issues.append(line)
            continue
        for citation in citations:
            if citation not in allowed_set:
                unknown_refs.append(citation)
    result.missing_citations = missing_citations
    result.unknown_citations = sorted(set(unknown_refs))
    if missing_citations or unknown_refs:
        result.hallucination_warning = True
    result.extra_citations = sorted(set(extract_citations(normalized)) - allowed_set)
    if result.extra_citations:
        result.hallucination_warning = True
    return result


def _extract_block(text: str, header_keyword: str) -> Optional[str]:
    lines = text.splitlines()
    header_keyword = header_keyword.lower()
    capturing = False
    buffer: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if capturing:
                buffer.append("")
            continue
        lower = stripped.lower()
        if lower.startswith("факты:") and header_keyword == "факты":
            capturing = True
            continue
        if lower.startswith("источники:") and header_keyword != "источники":
            if capturing:
                break
        if capturing:
            buffer.append(stripped)
    return "\n".join(buffer).strip() if buffer else None

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


SYSTEM_PROMPT_TEMPLATE = (
    "Ты аналитик, который отвечает по корпоративной базе знаний. "
    "Используй только переданный контекст. Если фактов недостаточно — явно сообщи об этом. "
    "Формат ответа:\n"
    "Факты:\n"
    "- <краткое утверждение> [doc_id:chunk]\n"
    "Источники:\n"
    "- [doc_id:chunk] Название документа (язык)\n"
    "Если источников нет, верни:\n"
    "Факты:\n"
    "- Источников не найдено\n"
    "Источники:\n"
    "- Источников не найдено\n"
    "Никаких других разделов или пояснений добавлять нельзя."
)


@dataclass(slots=True)
class ContextSection:
    doc_id: int
    chunk_id: int
    title: str
    language: str
    score_dense: float = 0.0
    score_sparse: float = 0.0
    combined_score: float = 0.0
    reasoning_hint: str = ""
    preview: str = ""
    content: str = ""
    url: Optional[str] = None
    extra: dict = field(default_factory=dict)
    translation_hint: str = ""


def build_system_prompt(custom_template: Optional[str] = None) -> str:
    """Возвращает системный промпт для RAG-ответа."""
    template = (custom_template or "").strip()
    return template or SYSTEM_PROMPT_TEMPLATE


def build_user_prompt(
    query: str,
    sections: Iterable[ContextSection],
    *,
    include_scores: bool = True,
) -> str:
    """Формирует пользовательский промпт с перечислением чанков и запросом."""
    cleaned_query = query.strip()
    section_lines: List[str] = []
    for idx, section in enumerate(sections, start=1):
        header = _format_section_header(section, idx, include_scores=include_scores)
        body = _format_section_body(section)
        section_lines.append(f"{header}\n{body}")
    context_block = "\n\n".join(section_lines) if section_lines else "Контекст отсутствует."
    return (
        f"Вопрос пользователя:\n{cleaned_query or 'нет запроса'}\n\n"
        f"Контекст:\n{context_block}\n\n"
        "Сформулируй ответ в требуемом формате."
    )


def _format_section_header(section: ContextSection, idx: int, *, include_scores: bool) -> str:
    parts = [f"[{idx}] doc_id={section.doc_id} chunk_id={section.chunk_id}"]
    if section.language:
        parts.append(f"lang={section.language}")
    if include_scores:
        parts.append(f"dense={section.score_dense:.3f}")
        parts.append(f"sparse={section.score_sparse:.3f}")
        parts.append(f"combined={section.combined_score:.3f}")
    if section.reasoning_hint:
        parts.append(f"hint={section.reasoning_hint}")
    return "; ".join(parts)


def _format_section_body(section: ContextSection) -> str:
    title = section.title or "Без названия"
    preview = (section.preview or "").strip()
    url_line = f"URL: {section.url}" if section.url else ""
    quoted_content = (section.content or "").strip()
    if quoted_content:
        quoted_content = f"\"\"\"\n{quoted_content}\n\"\"\""
    lines = [
        f"Заголовок: {title}",
    ]
    if preview:
        lines.append(f"Сниппет: {preview}")
    if url_line:
        lines.append(url_line)
    if section.translation_hint:
        lines.append(f"Подсказка: {section.translation_hint}")
    if section.extra:
        for key, value in section.extra.items():
            lines.append(f"{key}: {value}")
    if quoted_content:
        lines.append(f"Цитата:\n{quoted_content}")
    return "\n".join(lines)


def fallback_answer() -> str:
    """Возвращает текст fallback-ответа без источников."""
    return (
        "Факты:\n"
        "- Источников не найдено\n"
        "Источники:\n"
        "- Источников не найдено"
    )

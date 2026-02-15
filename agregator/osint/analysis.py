"""LLM analysis helpers for OSINT search results.

Extracted from ``osint/service.py`` to isolate the analysis context building,
message preparation and fallback logic from the main orchestration class.
"""

from __future__ import annotations

import json
from typing import Any, Iterable


def build_analysis_context(
    snapshot: dict, *, max_items: int = 3, snippet_limit: int = 220
) -> list[dict[str, Any]]:
    """Build structured analysis context from a job snapshot's sources."""
    contexts: list[dict[str, Any]] = []
    sources = snapshot.get("sources") or []
    for source in sources:
        metadata = source.get("metadata") or {}
        label = (
            metadata.get("label")
            or source.get("source")
            or source.get("engine")
            or "Источник"
        )
        entries: list[dict[str, str]] = []
        for item in (source.get("results") or [])[:max_items]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet_raw = str(item.get("snippet") or "").strip()
            if not title:
                continue
            snippet = snippet_raw
            if snippet_limit and len(snippet) > snippet_limit:
                snippet = snippet[: snippet_limit - 1].rstrip() + "\u2026"
            entries.append({"title": title, "url": url, "snippet": snippet})
        contexts.append(
            {
                "label": label,
                "fallback": bool(metadata.get("fallback")),
                "alerts": analysis_source_alerts(source),
                "notes": analysis_llm_notes(source),
                "entries": entries,
            }
        )
    return contexts


def analysis_source_alerts(source: dict) -> list[str]:
    """Collect warning alerts for a single OSINT source."""
    alerts: list[str] = []
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    if metadata.get("fallback"):
        alerts.append("Режим fallback — требуется ручная проверка.")
    if metadata.get("from_cache") or source.get("from_cache"):
        alerts.append("Ответ из кэша — информация может быть устаревшей.")
    if metadata.get("blocked") or source.get("blocked"):
        alerts.append("Источник заблокирован или обрезан — проверьте оригинал.")
    error_message = source.get("error")
    if error_message:
        alerts.append(f"Ошибка источника: {str(error_message).strip()}")
    llm_error = source.get("llm_error")
    if llm_error:
        alerts.append(f"LLM-разбор не выполнен: {str(llm_error).strip()}")
    return alerts


def analysis_llm_notes(source: dict, *, max_items: int = 3) -> list[str]:
    """Extract LLM-generated notes from a source payload."""
    payload = source.get("llm_payload")
    notes: list[str] = []
    if not isinstance(payload, str):
        return notes
    raw = payload.strip()
    if not raw:
        return notes
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    entries: Iterable[dict[str, Any]] = []
    if isinstance(parsed, dict):
        maybe_items = parsed.get("items")
        entries = maybe_items if isinstance(maybe_items, list) else []
    elif isinstance(parsed, list):
        entries = parsed
    count = 0
    for item in entries:
        if count >= max_items:
            break
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        url = str(item.get("url") or "").strip()
        if not title and not snippet:
            continue
        parts = []
        if title:
            parts.append(title)
        if snippet:
            parts.append(snippet)
        note = " — ".join(parts)
        if url:
            note += f" (URL: {url})"
        notes.append(note)
        count += 1
    if notes:
        return notes
    fallback_lines = [line.strip("•*- \t") for line in raw.splitlines() if line.strip()]
    for line in fallback_lines:
        if len(notes) >= max_items:
            break
        notes.append(line[:220])
    return notes


def build_structured_payload(contexts: list[dict[str, Any]]) -> str | None:
    """Serialize analysis contexts to a JSON string for embedding in prompts."""
    if not contexts:
        return None
    payload: list[dict[str, Any]] = []
    for ctx in contexts:
        entries = []
        for entry in ctx.get("entries") or []:
            if len(entries) >= 4:
                break
            if not isinstance(entry, dict):
                continue
            entries.append(
                {
                    "title": entry.get("title"),
                    "snippet": entry.get("snippet"),
                    "url": entry.get("url"),
                }
            )
        payload.append(
            {
                "label": ctx.get("label"),
                "fallback": bool(ctx.get("fallback")),
                "alerts": ctx.get("alerts") or [],
                "notes": ctx.get("notes") or [],
                "entries": entries,
            }
        )
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception:
        return None


def build_analysis_messages(
    query: str, contexts: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Construct chat messages for the OSINT analysis LLM call."""
    if not contexts:
        return []
    lines: list[str] = []
    for ctx in contexts:
        label = ctx.get("label") or "Источник"
        fallback = bool(ctx.get("fallback"))
        header_suffix = " [проверка вручную]" if fallback else ""
        lines.append(f"{label}{header_suffix}:")
        for alert in ctx.get("alerts") or []:
            lines.append(f"  ! {alert}")
        entries = ctx.get("entries") or []
        if not entries:
            lines.append("  - нет результатов")
        for entry in entries:
            title = entry.get("title") or ""
            snippet = entry.get("snippet") or ""
            url = entry.get("url") or ""
            fallback_note = " [проверка вручную]" if fallback else ""
            snippet_suffix = f" — {snippet}" if snippet else ""
            url_suffix = f" (URL: {url})" if url else ""
            lines.append(f"  - {title}{snippet_suffix}{url_suffix}{fallback_note}".rstrip())
        notes = ctx.get("notes") or []
        if notes:
            for idx, note in enumerate(notes[:3], start=1):
                lines.append(f"  * Наблюдение {idx}: {note}")
        if not entries and not notes:
            lines.append("  - сведений нет")

    user_content = (
        f"Запрос: {query or 'нет запроса'}\n\n"
        "Контекст по источникам:\n" + "\n".join(lines)
    )
    structured_payload = build_structured_payload(contexts)
    if structured_payload:
        user_content += "\n\nСтруктурированные данные (JSON):\n" + structured_payload
    system_prompt = (
        "Ты аналитик OSINT. Используй переданные данные, чтобы подготовить "
        "структурированный отчёт на русском языке.\n"
        "Формат ответа строго следующий:\n"
        "Наблюдения:\n- …\n"
        "Риски:\n- …\n"
        "Следующие шаги:\n- …\n"
        "Если раздел пуст, укажи «- нет данных». "
        "Не выдумывай факты — только из предоставленного контекста."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def build_analysis_fallback(
    query: str, contexts: list[dict[str, Any]], *, reason: str | None = None
) -> str:
    """Produce a plain-text fallback analysis when the LLM is unavailable."""
    lines: list[str] = []
    header_query = query.strip() if isinstance(query, str) else ""
    if header_query:
        lines.append(f"Краткий обзор по запросу «{header_query}».")
    else:
        lines.append("Краткий обзор результатов поиска.")

    if not contexts:
        lines.append("Источники не вернули релевантных результатов.")
    else:
        for ctx in contexts:
            label = ctx.get("label") or "Источник"
            entries = ctx.get("entries") or []
            if not entries:
                lines.append(f"\n{label}: результатов нет.")
                continue
            lines.append(f"\n{label}:")
            for alert in ctx.get("alerts") or []:
                lines.append(f"  ⚠ {alert}")
            notes = ctx.get("notes") or []
            for note in notes:
                lines.append(f"  • {note}")
            for entry in entries:
                title = entry.get("title") or ""
                snippet = entry.get("snippet") or ""
                url = entry.get("url") or ""
                line = f"  - {title}"
                if snippet:
                    line += f" — {snippet}"
                if url:
                    line += f" ({url})"
                lines.append(line)

    if reason:
        lines.append(f"\n(Примечание: {reason})")
    return "\n".join(lines)

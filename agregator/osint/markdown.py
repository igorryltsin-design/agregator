"""Lightweight Markdown renderer for OSINT summaries."""

from __future__ import annotations

import html
import re
from typing import List


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_ULIST_RE = re.compile(r"^[-*+]\s+")
_OLIST_RE = re.compile(r"^\d+\.\s+")
_BLOCKQUOTE_RE = re.compile(r"^>\s?")
_HR_RE = re.compile(r"^(-{3,}|_{3,}|\*{3,})$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _escape(text: str) -> str:
    return html.escape(text, quote=False)


def _apply_inline_formatting(text: str) -> str:
    escaped = _escape(text)

    def _replace_code(match: re.Match[str]) -> str:
        return f"<code>{match.group(1).strip()}</code>"

    def _replace_bold(match: re.Match[str]) -> str:
        return f"<strong>{match.group(1)}</strong>"

    def _replace_italic(match: re.Match[str]) -> str:
        return f"<em>{match.group(1)}</em>"

    def _replace_strike(match: re.Match[str]) -> str:
        return f"<del>{match.group(1)}</del>"

    def _replace_link(match: re.Match[str]) -> str:
        text_part = match.group(1)
        href = match.group(2).strip()
        if re.match(r"^(javascript|data):", href, flags=re.IGNORECASE):
            return text_part
        safe_href = html.escape(href, quote=True)
        return f'<a href="{safe_href}" target="_blank" rel="noopener">{text_part}</a>'

    escaped = re.sub(r"`([^`]+)`", _replace_code, escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", _replace_bold, escaped)
    escaped = re.sub(r"__([^_]+)__", _replace_bold, escaped)
    escaped = re.sub(r"\*(?!\s)([^*]+)\*", _replace_italic, escaped)
    escaped = re.sub(r"_(?!\s)([^_]+)_", _replace_italic, escaped)
    escaped = re.sub(r"~~([^~]+)~~", _replace_strike, escaped)
    escaped = _LINK_RE.sub(_replace_link, escaped)
    return escaped


def _close_lists(state: dict[str, bool], buffer: List[str]) -> None:
    if state.get("in_ul"):
        buffer.append("</ul>")
        state["in_ul"] = False
    if state.get("in_ol"):
        buffer.append("</ol>")
        state["in_ol"] = False


def _handle_list_item(
    line: str,
    *,
    state: dict[str, bool],
    buffer: List[str],
    ordered: bool,
) -> None:
    content = re.sub(_OLIST_RE if ordered else _ULIST_RE, "", line, count=1)
    if ordered:
        if not state.get("in_ol"):
            _close_lists(state, buffer)
            buffer.append("<ol>")
            state["in_ol"] = True
    else:
        if not state.get("in_ul"):
            _close_lists(state, buffer)
            buffer.append("<ul>")
            state["in_ul"] = True
    buffer.append(f"<li>{_apply_inline_formatting(content)}</li>")


def _handle_heading(line: str, *, state: dict[str, bool], buffer: List[str]) -> None:
    match = _HEADING_RE.match(line)
    if not match:
        _handle_paragraph(line, state=state, buffer=buffer)
        return
    level = min(max(len(match.group(1)), 1), 6)
    content = match.group(2)
    _close_lists(state, buffer)
    buffer.append(f"<h{level}>{_apply_inline_formatting(content)}</h{level}>")


def _handle_paragraph(line: str, *, state: dict[str, bool], buffer: List[str]) -> None:
    _close_lists(state, buffer)
    buffer.append(f"<p>{_apply_inline_formatting(line)}</p>")


def _handle_blockquote(line: str, *, state: dict[str, bool], buffer: List[str]) -> None:
    _close_lists(state, buffer)
    content = _BLOCKQUOTE_RE.sub("", line, count=1)
    buffer.append(f"<blockquote>{_apply_inline_formatting(content)}</blockquote>")


def render_markdown(markdown: str | None) -> str:
    """Render a limited Markdown subset to sanitized HTML."""
    if not markdown:
        return ""
    lines = markdown.splitlines()
    buffer: List[str] = []
    state = {"in_ul": False, "in_ol": False, "in_code": False}
    code_buffer: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n\r")
        stripped = line.strip()

        if state["in_code"]:
            if stripped.startswith("```"):
                code_html = html.escape("\n".join(code_buffer))
                buffer.append(f"<pre><code>{code_html}</code></pre>")
                code_buffer = []
                state["in_code"] = False
            else:
                code_buffer.append(raw_line)
            continue

        if stripped.startswith("```"):
            _close_lists(state, buffer)
            state["in_code"] = True
            code_buffer = []
            continue

        if not stripped:
            _close_lists(state, buffer)
            continue

        if _HEADING_RE.match(stripped):
            _handle_heading(stripped, state=state, buffer=buffer)
            continue

        if _ULIST_RE.match(stripped):
            _handle_list_item(stripped, state=state, buffer=buffer, ordered=False)
            continue

        if _OLIST_RE.match(stripped):
            _handle_list_item(stripped, state=state, buffer=buffer, ordered=True)
            continue

        if _BLOCKQUOTE_RE.match(stripped):
            _handle_blockquote(stripped, state=state, buffer=buffer)
            continue

        if _HR_RE.match(stripped):
            _close_lists(state, buffer)
            buffer.append("<hr />")
            continue

        _handle_paragraph(stripped, state=state, buffer=buffer)

    _close_lists(state, buffer)
    if state["in_code"] and code_buffer:
        code_html = html.escape("\n".join(code_buffer))
        buffer.append(f"<pre><code>{code_html}</code></pre>")

    return "\n".join(buffer)


__all__ = ["render_markdown"]

"""LLM-assisted parsing for SERP HTML snapshots."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html import unescape
from typing import Callable, Iterable, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from .serp import SerpResult

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
_LINK_RE = re.compile(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<body>.*?)</a>', re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(slots=True)
class ParsedSerpItem:
    rank: int
    title: str
    url: str
    snippet: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedSerpPayload:
    engine: str
    query: str
    items: list[ParsedSerpItem]
    raw_response: str | None = None
    llm_model: str | None = None
    llm_error: str | None = None


LlmChatCallable = Callable[[list[dict[str, str]]], object]


class SerpParser:
    """Convert SERP HTML into structured entries using an optional LLM bridge."""

    def __init__(
        self,
        *,
        llm_chat: LlmChatCallable | None = None,
        max_items: int = 10,
        max_html_chars: int = 20000,
        logger: logging.Logger | None = None,
    ) -> None:
        self.llm_chat = llm_chat
        self.max_items = max(1, max_items)
        self.max_html_chars = max(1000, max_html_chars)
        self.logger = logger or logging.getLogger("agregator.osint.parser")

    def parse(self, serp_result: SerpResult) -> ParsedSerpPayload:
        api_items = serp_result.metadata.get("api_items")
        if isinstance(api_items, list):
            return self._parse_from_api_items(serp_result, api_items)

        if self.llm_chat is not None:
            try:
                return self._parse_with_llm(serp_result)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("LLM SERP parse failed (%s): %s", serp_result.engine, exc)
                return self._fallback_payload(serp_result, llm_error=str(exc))
        return self._fallback_payload(serp_result)

    # ------------------------------------------------------------------
    # LLM parsing
    # ------------------------------------------------------------------
    def _parse_with_llm(self, serp_result: SerpResult) -> ParsedSerpPayload:
        truncated_html = serp_result.html[: self.max_html_chars]
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты помощник аналитика OSINT. На входе HTML страницы результатов поиска.\n"
                    "Верни JSON со списком полей `items`, где каждый элемент содержит ключи:\n"
                    "`title` (строка), `url` (绝对 URL), `snippet` (краткий текст), `score` (0..1, по желанию).\n"
                    "Не добавляй пояснений, только JSON без форматированного текста."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Поисковая система: {serp_result.engine}\n"
                    f"Запрос: {serp_result.query}\n"
                    f"HTML:\n{truncated_html}"
                ),
            },
        ]
        content, model, raw = self._invoke_llm(messages)
        cleaned = self._strip_code_fence(content)
        data = self._load_json(cleaned)
        items_data = self._extract_items(data)
        parsed_items = self._normalise_items(items_data)
        return ParsedSerpPayload(
            engine=serp_result.engine,
            query=serp_result.query,
            items=parsed_items[: self.max_items],
            raw_response=raw or content,
            llm_model=model,
            llm_error=None,
        )

    def _invoke_llm(self, messages: list[dict[str, str]]) -> tuple[str, str | None, str | None]:
        response = self.llm_chat(messages)
        content: str | None = None
        model: str | None = None
        raw: str | None = None
        if isinstance(response, str):
            content = response
            raw = response
        elif isinstance(response, (list, tuple)) and response:
            primary = response[0]
            secondary = response[1] if len(response) > 1 else None
            content = str(primary) if primary is not None else ""
            raw = str(secondary) if secondary is not None else content
        elif isinstance(response, dict):
            for key in ("content", "text", "message", "output"):
                if key in response and response[key]:
                    content = str(response[key])
                    break
            if not content:
                content = str(response)
            raw = str(
                response.get("raw")
                or response.get("raw_text")
                or response.get("response")
                or content
            )
            model = str(response.get("model") or response.get("provider") or "") or None
        else:
            content = str(response)
            raw = content
        return content or "", model, raw

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        trimmed = text.strip()
        if not trimmed:
            return trimmed
        match = _CODE_FENCE_RE.match(trimmed)
        if match:
            return match.group(1).strip()
        if trimmed.startswith("```"):
            trimmed = trimmed.lstrip("`")
        if trimmed.endswith("```"):
            trimmed = trimmed.rstrip("`")
        return trimmed.strip()

    @staticmethod
    def _load_json(text: str) -> object:
        if not text:
            raise ValueError("empty LLM response")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON from LLM: {exc}") from exc

    @staticmethod
    def _extract_items(data: object) -> Sequence[dict]:
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list):
                return items
        if isinstance(data, list):
            return data  # pragma: no cover - fallback format
        raise ValueError("JSON does not contain 'items' list")

    def _normalise_items(self, entries: Iterable[dict]) -> list[ParsedSerpItem]:
        results: list[ParsedSerpItem] = []
        seen_urls: set[str] = set()
        for idx, entry in enumerate(entries, start=1):
            if len(results) >= self.max_items:
                break
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url") or "").strip()
            title = self._clean_text(entry.get("title"))
            snippet = self._clean_text(entry.get("snippet"))
            if not url or not title:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            metadata = {}
            score = entry.get("score")
            if isinstance(score, (int, float)):
                metadata["score"] = float(score)
            for extra_key in ("source", "language", "source_type"):
                if extra_key in entry:
                    metadata[extra_key] = entry[extra_key]
            results.append(
                ParsedSerpItem(
                    rank=idx,
                    title=title,
                    url=url,
                    snippet=snippet,
                    metadata=metadata,
                )
            )
        return results

    def _parse_from_api_items(
        self,
        serp_result: SerpResult,
        items: Iterable[dict[str, object]],
    ) -> ParsedSerpPayload:
        parsed_items = self._normalise_items(items)
        return ParsedSerpPayload(
            engine=serp_result.engine,
            query=serp_result.query,
            items=parsed_items[: self.max_items],
            raw_response=None,
            llm_model=None,
            llm_error=None,
        )

    # ------------------------------------------------------------------
    # Fallback parsing
    # ------------------------------------------------------------------
    def _fallback_payload(self, serp_result: SerpResult, *, llm_error: str | None = None) -> ParsedSerpPayload:
        items = self._fallback_items(serp_result.html)
        return ParsedSerpPayload(
            engine=serp_result.engine,
            query=serp_result.query,
            items=items,
            raw_response=None,
            llm_model=None,
            llm_error=llm_error,
        )

    def _fallback_items(self, html: str) -> list[ParsedSerpItem]:
        items: list[ParsedSerpItem] = []
        seen: set[str] = set()
        for match_idx, match in enumerate(_LINK_RE.finditer(html), start=1):
            href = match.group("href")
            body = match.group("body")
            if not href:
                continue
            href = unescape(href)
            if href.startswith("/url"):
                parsed = urlparse(href)
                query = parse_qs(parsed.query)
                candidate = query.get("q") or query.get("url")
                if candidate:
                    href = unquote(candidate[0])
            if href.startswith("/"):
                continue  # relative links are typically navigation items
            if href in seen:
                continue
            seen.add(href)
            text = self._clean_text(body)
            if not text or len(text) < 4:
                continue
            items.append(
                ParsedSerpItem(
                    rank=len(items) + 1,
                    title=text,
                    url=href,
                    snippet="",
                    metadata={"fallback": True},
                )
            )
            if len(items) >= self.max_items:
                break
        return items

    @staticmethod
    def _clean_text(value: object) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = _TAG_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text)
        return unescape(text).strip()

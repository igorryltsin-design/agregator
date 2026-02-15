"""SERP connectors orchestrated via Playwright."""

from __future__ import annotations

import html as html_module
import json
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Mapping, Optional, Sequence
from urllib.parse import urlencode

from agregator.services import BrowserManager, BrowserSettings, get_browser_manager, get_task_queue
from agregator.services.http import http_request
from agregator.services.tasks import TaskQueue

from .artifacts import artifact_subdir, relative_artifact_path
from .cache import configure_osint_cache, osint_cache_get, osint_cache_set

SUPPORTED_ENGINES = ("google", "yandex")
SerpEngine = Literal["google", "yandex"]


def _trim_locale(locale: str) -> str:
    token = (locale or "ru-RU").split(".")[0]
    return token.replace("_", "-")


@dataclass(slots=True)
class SerpRequest:
    query: str
    engine: SerpEngine
    locale: str = "ru-RU"
    region: str | None = None
    max_results: int | None = None
    safe: bool = False
    force_refresh: bool = False
    extra_params: Mapping[str, str] | None = None
    retry: bool = False
    user_agent_override: str | None = None
    proxy_override: str | None = None

    def cache_key(self) -> str:
        base = [
            f"query={self.query.strip()}",
            f"engine={self.engine}",
            f"locale={_trim_locale(self.locale).lower()}",
        ]
        if self.region:
            base.append(f"region={self.region}")
        if self.max_results is not None:
            base.append(f"max={self.max_results}")
        if self.safe:
            base.append("safe=1")
        if self.extra_params:
            extra = ",".join(f"{k}={v}" for k, v in sorted(self.extra_params.items()))
            base.append(f"extra={extra}")
        if self.proxy_override:
            base.append(f"proxy={self.proxy_override}")
        return "osint-serp::" + "|".join(base)


@dataclass(slots=True)
class SerpResult:
    engine: SerpEngine
    query: str
    requested_url: str
    final_url: str
    html: str
    fetched_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)
    text: str | None = None
    screenshot_path: str | None = None

    @property
    def blocked(self) -> bool:
        return bool(self.metadata.get("blocked"))


@dataclass(slots=True)
class SerpSettings:
    cache_enabled: bool = True
    cache_ttl_seconds: int = 900
    cache_max_items: int = 128
    wait_after_load_ms: int = 1200
    wait_selector: str | None = None
    navigation_timeout_ms: int = 45000
    reuse_cache_on_block: bool = True
    user_agent_override: str | None = None
    capture_screenshots: bool = True
    store_text_content: bool = True
    max_text_chars: int = 20000
    retry_user_agents: tuple[str, ...] = ()
    retry_proxies: tuple[str, ...] = ()
    search_api_url: str | None = None
    search_api_method: str = "POST"
    search_api_key: str | None = None


class SerpFetcher:
    """Coordinates SERP fetching via headless browser and background queue."""

    def __init__(
        self,
        *,
        browser: BrowserManager | None = None,
        settings: SerpSettings | None = None,
        task_queue: TaskQueue | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.browser = browser or get_browser_manager()
        self.settings = settings or SerpSettings()
        self.queue = task_queue or get_task_queue()
        self.logger = logger or logging.getLogger("agregator.osint.serp")
        configure_osint_cache(
            enabled=self.settings.cache_enabled,
            max_items=self.settings.cache_max_items,
            ttl_seconds=self.settings.cache_ttl_seconds,
        )

    def configure_browser(self, settings: BrowserSettings) -> None:
        self.browser.configure(settings)

    def fetch(self, request: SerpRequest) -> SerpResult:
        cache_key = request.cache_key()
        if self.settings.search_api_url:
            return self._fetch_via_search_api(request)
        if self.settings.cache_enabled and not request.force_refresh and not request.retry:
            cached = osint_cache_get(cache_key)
            if isinstance(cached, SerpResult):
                cached.metadata.setdefault("from_cache", True)
                return cached

        user_agent_override = request.user_agent_override
        proxy_override = request.proxy_override
        if request.retry:
            retry_user_agent, retry_proxy = self._pick_retry_overrides()
            if user_agent_override is None and retry_user_agent:
                user_agent_override = retry_user_agent
            if proxy_override is None and retry_proxy:
                proxy_override = retry_proxy

        result = self._fetch_live(request, user_agent_override=user_agent_override, proxy_override=proxy_override)

        if self.settings.cache_enabled and not result.blocked:
            osint_cache_set(cache_key, result)
        elif result.blocked and self.settings.cache_enabled and self.settings.reuse_cache_on_block:
            cached = osint_cache_get(cache_key)
            if isinstance(cached, SerpResult):
                cached.metadata.setdefault("blocked_fallback", True)
                return cached

        return result

    def submit(
        self,
        request: SerpRequest,
        *,
        on_complete: Optional[Callable[[SerpResult], None]] = None,
    ) -> str:
        description = f"osint-serp:{request.engine}:{request.query[:48]}"
        return self.queue.submit(
            self._execute_job,
            request,
            on_complete=on_complete,
            description=description,
        )

    def choose_retry_overrides(self) -> tuple[str | None, str | None]:
        return self._pick_retry_overrides()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _execute_job(
        self,
        request: SerpRequest,
        on_complete: Optional[Callable[[SerpResult], None]] = None,
    ) -> None:
        try:
            result = self.fetch(request)
        except Exception as exc:
            self.logger.exception("SERP fetch failed (%s): %s", request.engine, exc)
            raise
        if on_complete is None:
            return
        try:
            on_complete(result)
        except Exception as exc:
            self.logger.exception("SERP callback failed: %s", exc)

    def _fetch_live(
        self,
        request: SerpRequest,
        *,
        user_agent_override: str | None = None,
        proxy_override: str | None = None,
    ) -> SerpResult:
        engine = request.engine
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(f"Unsupported search engine: {engine}")
        target_url = self._build_url(request)
        extra_headers = {"Accept-Language": _trim_locale(request.locale)}
        metadata: dict[str, object] = {
            "engine": engine,
            "locale": request.locale,
            "region": request.region,
            "requested_url": target_url,
        }
        user_agent = user_agent_override or self.settings.user_agent_override
        screenshot_path: str | None = None
        blocked = False
        try:
            with self.browser.page(user_agent=user_agent, headers=extra_headers, proxy=proxy_override) as page:
                page.goto(
                    target_url,
                    wait_until="networkidle",
                    timeout=self.settings.navigation_timeout_ms,
                )
                if self.settings.wait_selector:
                    page.wait_for_selector(self.settings.wait_selector, timeout=self.settings.navigation_timeout_ms)
                extra_wait = self.settings.wait_after_load_ms
                if extra_wait > 0:
                    jitter = random.randint(150, 600)
                    page.wait_for_timeout(extra_wait + jitter)
                html = page.content()
                final_url = page.url
                if user_agent is None:
                    try:
                        user_agent = page.evaluate("navigator.userAgent")
                    except Exception:
                        user_agent = None
                blocked = self._looks_blocked(html.lower(), engine)
                if self.settings.capture_screenshots and not blocked:
                    screenshot_path = self._capture_screenshot(page)
        except Exception as exc:
            metadata["error"] = str(exc)
            raise
        metadata["from_cache"] = False
        metadata["blocked"] = blocked
        metadata["fallback"] = blocked
        metadata["final_url"] = final_url
        if blocked:
            metadata["fallback_url"] = final_url
        if user_agent:
            metadata["user_agent"] = user_agent
        if proxy_override:
            metadata["proxy"] = proxy_override
        metadata["retry"] = bool(request.retry)
        metadata["fetched_at"] = datetime.now(timezone.utc).isoformat()
        text_content: str | None = None
        text_excerpt: str | None = None
        if self.settings.store_text_content:
            text_content, text_excerpt, extra_meta = self._extract_text_and_meta(
                html,
                limit=self.settings.max_text_chars,
            )
            if extra_meta.get("meta"):
                metadata["document_meta"] = extra_meta["meta"]
            if extra_meta.get("word_count") is not None:
                metadata["word_count"] = extra_meta["word_count"]
            if text_excerpt:
                metadata["text_excerpt"] = text_excerpt
        if screenshot_path:
            metadata["screenshot_path"] = screenshot_path
        html_artifact = self._store_artifact("html", ".html", html)
        if html_artifact:
            metadata["html_artifact"] = html_artifact
        text_artifact = self._store_artifact("text", ".txt", text_content)
        if text_artifact:
            metadata["text_artifact"] = text_artifact
        metadata_snapshot = None
        try:
            metadata_snapshot = json.dumps(metadata, ensure_ascii=False, indent=2)
        except Exception:
            metadata_snapshot = None
        if metadata_snapshot:
            metadata_artifact = self._store_artifact("metadata", ".json", metadata_snapshot)
            if metadata_artifact:
                metadata["metadata_artifact"] = metadata_artifact
        return SerpResult(
            engine=engine,
            query=request.query,
            requested_url=target_url,
            final_url=final_url,
            html=html,
            fetched_at=datetime.now(timezone.utc),
            metadata=metadata,
            text=text_content,
            screenshot_path=screenshot_path,
        )

    def _fetch_via_search_api(self, request: SerpRequest) -> SerpResult:
        url = (self.settings.search_api_url or '').strip()
        if not url:
            raise ValueError("search_api_url_not_configured")
        method = (self.settings.search_api_method or "POST").strip().upper()
        payload: dict[str, object | None] = {
            "query": request.query,
            "engine": request.engine,
            "locale": request.locale,
            "region": request.region,
            "max_results": request.max_results,
            "safe": request.safe,
            "retry": request.retry,
        }
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.search_api_key:
            headers["Authorization"] = f"Bearer {self.settings.search_api_key}"
        try:
            if method == "GET":
                response = http_request(
                    method,
                    url,
                    params={k: v for k, v in payload.items() if v is not None},
                    headers=headers,
                    raise_for_status=True,
                )
            else:
                response = http_request(
                    method,
                    url,
                    json=payload,
                    headers=headers,
                    raise_for_status=True,
                )
        except Exception as exc:
            self.logger.warning("Search API request failed (%s): %s", request.engine, exc)
            raise
        try:
            data = response.json()
        except Exception as exc:
            raise ValueError("invalid_search_api_response") from exc
        items_raw = data.get("items") or data.get("results")
        if not isinstance(items_raw, list):
            raise ValueError("search_api_no_items")
        items: list[dict[str, object]] = [entry for entry in items_raw if isinstance(entry, dict)]
        preview_items = items[: min(len(items), 50)]
        html_fragments: list[str] = []
        text_parts: list[str] = []
        for idx, entry in enumerate(preview_items, start=1):
            url_value = str(entry.get("url") or "").strip()
            if not url_value:
                continue
            title_value = str(entry.get("title") or url_value).strip()
            snippet_value = str(entry.get("snippet") or "").strip()
            html_fragments.append(
                f'<div data-rank="{idx}"><a href="{html_module.escape(url_value)}">{html_module.escape(title_value)}</a>'
            )
            if snippet_value:
                html_fragments.append(f'<p>{html_module.escape(snippet_value)}</p>')
                text_parts.append(snippet_value)
            html_fragments.append("</div>")
        html_body = "".join(html_fragments)
        html_content = f"<html><body>{html_body}</body></html>" if html_body else "<html><body></body></html>"
        metadata: dict[str, object] = {
            "api_items": items,
            "api_url": url,
            "api_response_code": getattr(response, "status_code", 0),
            "from_cache": False,
            "blocked": bool(data.get("blocked")),
            "fallback": False,
            "final_url": str(data.get("final_url") or data.get("url") or url),
        }
        requested_url = str(data.get("requested_url") or url)
        text_content = "\n".join(text_parts) if text_parts else None
        return SerpResult(
            engine=request.engine,
            query=request.query,
            requested_url=requested_url,
            final_url=metadata["final_url"] or url,
            html=html_content,
            fetched_at=datetime.now(timezone.utc),
            metadata=metadata,
            text=text_content,
            screenshot_path=None,
        )

    def _capture_screenshot(self, page) -> str | None:
        try:
            directory = artifact_subdir("screenshots")
            filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}.png"
            file_path = directory / filename
            page.screenshot(path=str(file_path), full_page=True)
            return relative_artifact_path(file_path)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("SERP screenshot failed: %s", exc)
            return None

    def _store_artifact(self, kind: str, suffix: str, payload: str | bytes | None) -> str | None:
        if payload is None:
            return None
        data: bytes
        if isinstance(payload, bytes):
            data = payload
        else:
            text = payload.strip()
            if not text:
                return None
            data = text.encode("utf-8", "ignore")
        try:
            directory = artifact_subdir(kind)
            filename = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:12]}{suffix}"
            path = directory / filename
            path.write_bytes(data)
            return relative_artifact_path(path)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("SERP artifact store failed (%s): %s", kind, exc)
            return None

    def _pick_retry_overrides(self) -> tuple[str | None, str | None]:
        user_agent: str | None = None
        proxy: str | None = None
        if self.settings.retry_user_agents:
            user_agent = random.choice(self.settings.retry_user_agents)
        elif not self.settings.user_agent_override:
            try:
                pool = tuple(self.browser.settings.user_agents)
                if pool:
                    user_agent = random.choice(pool)
            except Exception:
                user_agent = None
        if self.settings.retry_proxies:
            proxy = random.choice(self.settings.retry_proxies)
        return user_agent, proxy

    @staticmethod
    def _extract_text_and_meta(html: str, *, limit: int) -> tuple[str | None, str | None, dict[str, object]]:
        if not html:
            return None, None, {}
        cleaned = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<!--.*?-->", " ", cleaned, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", cleaned)
        text = html_module.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return None, None, {}
        if limit > 0 and len(text) > limit:
            text = text[:limit].strip()
        words = text.split()
        word_count = len(words)
        excerpt_limit = min(len(text), 1200)
        excerpt = text[:excerpt_limit].strip()
        meta_info = SerpFetcher._extract_meta_tags(html)
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            meta_info.setdefault("title", html_module.unescape(title_match.group(1).strip()))
        summary = {
            "meta": meta_info,
            "word_count": word_count,
        }
        return text, excerpt, summary

    @staticmethod
    def _extract_meta_tags(html: str) -> dict[str, str]:
        meta: dict[str, str] = {}
        for match in re.finditer(r"<meta\s+[^>]*>", html, flags=re.IGNORECASE):
            tag = match.group(0)
            name = SerpFetcher._attr_value(tag, "name") or SerpFetcher._attr_value(tag, "property")
            content = SerpFetcher._attr_value(tag, "content")
            if not name or not content:
                continue
            key = name.lower()
            if key in {"description", "keywords", "og:title", "og:description"}:
                meta[key] = html_module.unescape(content.strip())
        return meta

    @staticmethod
    def _attr_value(tag: str, attribute: str) -> str | None:
        quoted = re.search(fr"{attribute}\s*=\s*\"([^\"]*)\"", tag, flags=re.IGNORECASE)
        if quoted:
            return quoted.group(1)
        single = re.search(fr"{attribute}\s*=\s*'([^']*)'", tag, flags=re.IGNORECASE)
        if single:
            return single.group(1)
        unquoted = re.search(fr"{attribute}\s*=\s*([^\s>]+)", tag, flags=re.IGNORECASE)
        if unquoted:
            return unquoted.group(1)
        return None

    def _build_url(self, request: SerpRequest) -> str:
        locale = _trim_locale(request.locale)
        stem = locale.split("-")[0]
        max_results = request.max_results
        params: dict[str, str] = {}
        extra = dict(request.extra_params or {})
        if request.engine == "google":
            params = {
                "q": request.query,
                "hl": stem,
                "num": str(max_results or 10),
                "source": "hp",
            }
            if request.safe:
                params["safe"] = "active"
        elif request.engine == "yandex":
            params = {
                "text": request.query,
                "lr": request.region or "213",  # Moscow default
            }
            if stem:
                params["lang"] = stem
        else:
            raise ValueError(f"Unsupported search engine: {request.engine}")
        params.update(extra)
        query = urlencode(params, safe=":+")
        if request.engine == "google":
            return f"https://www.google.com/search?{query}"
        return f"https://yandex.ru/search/?{query}"

    @staticmethod
    def _looks_blocked(content_lower: str, engine: SerpEngine) -> bool:
        needles: Sequence[str]
        if engine == "google":
            needles = (
                "unusual traffic",
                "our systems have detected",
                "before we continue",
                "sorry, but we're having trouble",
                "i'm not a robot",
                "я не робот",
                "подозрительный трафик",
                "почему это могло произойти",
                "about this page",
                "this page appears when",
            )
        elif engine == "yandex":
            needles = (
                "похоже, ваш запрос похож",
                "превышено количество обращений",
                "captcha",
                "необходимо подтвердить",
            )
        else:
            raise ValueError(f"Unsupported search engine: {engine}")
        return any(token in content_lower for token in needles)

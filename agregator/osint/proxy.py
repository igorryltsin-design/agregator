"""Simple HTTP proxy helpers for interactive OSINT browsing."""

from __future__ import annotations

import html as html_module
import re
from typing import Callable
from urllib.parse import quote, urljoin, urlparse

import requests

_PROXY_SESSIONS: dict[str, requests.Session] = {}
_LINK_ATTRS = re.compile(r'(href|src|action)=("|\')([^"\']+)("|\')', re.IGNORECASE)


def _build_proxy_url(source_id: str, target: str) -> str:
    return f"/api/osint/browser/{quote(source_id, safe='')}?next={quote(target, safe='')}"


def _rewrite_attribute(source_id: str, base_url: str, match: re.Match) -> str:
    attr, quote_char, raw_value, _ = match.groups()
    if not raw_value:
        return match.group(0)
    candidate = raw_value.strip()
    parsed = urlparse(candidate)
    absolute = candidate
    if not parsed.scheme:
        absolute = urljoin(base_url, candidate)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return match.group(0)
    if absolute.startswith("http"):
        proxied = _build_proxy_url(source_id, absolute)
        return f'{attr}={quote_char}{proxied}{quote_char}'
    if candidate.startswith("//"):
        proxied = _build_proxy_url(source_id, f"https:{candidate}")
        return f'{attr}={quote_char}{proxied}{quote_char}'
    return match.group(0)


def rewrite_html_for_proxy(source_id: str, base_url: str, html: str) -> str:
    return _LINK_ATTRS.sub(lambda match: _rewrite_attribute(source_id, base_url, match), html)


def get_proxy_session(source_id: str) -> requests.Session:
    session = _PROXY_SESSIONS.get(source_id)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        )
        _PROXY_SESSIONS[source_id] = session
    return session


def fetch_proxied_page(source_id: str, url: str, method: str, **kwargs) -> requests.Response:
    session = get_proxy_session(source_id)
    return session.request(method, url, **kwargs)

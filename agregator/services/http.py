"""HTTP utility layer with retry-aware client for Agregator."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class HttpSettings:
    """Runtime configuration for HTTP requests."""

    timeout: float  # суммарное время ожидания чтения в секундах
    connect_timeout: float  # время установления соединения в секундах
    retries: int
    backoff_factor: float
    status_forcelist: Iterable[int] = (429, 500, 502, 503, 504)


_LOGGER = logging.getLogger("agregator.http")
_SETTINGS = HttpSettings(
    timeout=float(os.getenv("AG_HTTP_TIMEOUT", "120") or 120),
    connect_timeout=float(os.getenv("AG_HTTP_CONNECT_TIMEOUT", "10") or 10),
    retries=int(os.getenv("AG_HTTP_RETRIES", "3") or 3),
    backoff_factor=float(os.getenv("AG_HTTP_BACKOFF", "0.5") or 0.5),
)
_SESSION: Session | None = None
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"})


def _build_retry(settings: HttpSettings) -> Retry:
    return Retry(
        total=max(0, settings.retries),
        connect=max(0, settings.retries),
        read=max(0, settings.retries),
        backoff_factor=max(0.0, settings.backoff_factor),
        status_forcelist=tuple(settings.status_forcelist),
        allowed_methods=_ALLOWED_METHODS,
        raise_on_status=False,
    )


def _create_session(settings: HttpSettings) -> Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_build_retry(settings))
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def configure_http(settings: HttpSettings) -> None:
    """Update HTTP defaults and rebuild the shared session."""
    global _SETTINGS, _SESSION
    _SETTINGS = settings
    _SESSION = _create_session(settings)
    _LOGGER.info(
        "HTTP client configured: timeout=%ss connect=%ss retries=%s backoff=%s",
        settings.timeout,
        settings.connect_timeout,
        settings.retries,
        settings.backoff_factor,
    )


def get_http_session() -> Session:
    """Return a lazily initialised shared requests session."""
    global _SESSION
    if _SESSION is None:
        _SESSION = _create_session(_SETTINGS)
    return _SESSION


def get_http_settings() -> HttpSettings:
    """Возвращает текущие настройки HTTP-клиента."""
    return _SETTINGS


def _default_timeout() -> tuple[float, float]:
    connect = max(0.1, float(_SETTINGS.connect_timeout))
    read = max(connect + 1.0, float(_SETTINGS.timeout))
    return connect, read


def http_request(
    method: str,
    url: str,
    *,
    timeout: Any | None = None,
    session: Session | None = None,
    logger: Optional[logging.Logger] = None,
    fallback: Any | None = None,
    raise_for_status: bool = False,
    **kwargs: Any,
) -> Response | Any:
    """Perform an HTTP request with shared session and logging.

    Parameters
    ----------
    method: HTTP verb (GET/POST/...)
    url: target URL
    timeout: custom timeout or tuple (connect, read). Falls back to defaults if omitted.
    session: optional `requests.Session` to use instead of the shared one.
    logger: logger for error messages (defaults to `agregator.http`).
    fallback: value to return instead of raising on failure.
    raise_for_status: if True, call `response.raise_for_status()` before returning.
    kwargs: forwarded to `session.request`.
    """

    sess = session or get_http_session()
    timer = timeout if timeout is not None else _default_timeout()
    log = logger or _LOGGER
    verb = method.upper()
    try:
        response = sess.request(verb, url, timeout=timer, **kwargs)
        if raise_for_status:
            response.raise_for_status()
        return response
    except requests.RequestException as exc:
        log.warning("HTTP %s %s failed: %s", verb, url, exc)
        if fallback is not None:
            return fallback
        raise

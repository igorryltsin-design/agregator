"""Playwright-based headless browser manager for external search connectors."""

from __future__ import annotations

import atexit
import os
import random
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Mapping, MutableMapping, Sequence

from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    return token in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_tuple(name: str, default: tuple[int, int]) -> tuple[int, int]:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = raw.replace(";", ",").split(",")
    if len(parts) != 2:
        return default
    try:
        width = int(parts[0].strip())
        height = int(parts[1].strip())
    except ValueError:
        return default
    return max(320, width), max(320, height)


DEFAULT_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_1) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.6367.118 Safari/537.36",
)


@dataclass(slots=True)
class BrowserSettings:
    """Launch configuration for Playwright browser sessions."""

    headless: bool = field(default_factory=lambda: _env_bool("AG_BROWSER_HEADLESS", True))
    slow_mo_ms: int = field(default_factory=lambda: _env_int("AG_BROWSER_SLOWMO_MS", 0, minimum=0, maximum=2000))
    default_viewport: tuple[int, int] = field(
        default_factory=lambda: _env_tuple("AG_BROWSER_VIEWPORT", (1366, 768))
    )
    user_agents: Sequence[str] = field(default_factory=lambda: DEFAULT_USER_AGENTS)
    proxy: str | None = field(default_factory=lambda: (os.getenv("AG_BROWSER_PROXY") or None))
    context_timeout_ms: int = field(
        default_factory=lambda: _env_int("AG_BROWSER_CONTEXT_TIMEOUT_MS", 45000, minimum=5000, maximum=180000)
    )
    navigation_timeout_ms: int = field(
        default_factory=lambda: _env_int("AG_BROWSER_NAV_TIMEOUT_MS", 45000, minimum=5000, maximum=180000)
    )


class BrowserManager:
    """Lazy singleton around Playwright with safe shutdown semantics."""

    def __init__(self, settings: BrowserSettings | None = None) -> None:
        self._settings = settings or BrowserSettings()
        self._lock = threading.Lock()
        self._browser = None
        self._playwright = None
        self._states: dict[int, dict[str, object | None]] = {}
        atexit.register(self.stop)

    @property
    def settings(self) -> BrowserSettings:
        return self._settings

    def _get_state(self) -> tuple[dict[str, object | None], int]:
        ident = threading.get_ident()
        with self._lock:
            state = self._states.get(ident)
            if state is None:
                state = {
                    "playwright": None,
                    "browser": None,
                    "started": False,
                }
                self._states[ident] = state
        return state, ident

    def configure(self, settings: BrowserSettings) -> None:
        old_states: list[dict[str, object | None]] = []
        with self._lock:
            old_states = list(self._states.values())
            self._states = {}
            self._settings = settings
        for state in old_states:
            browser = state.get("browser")
            playwright = state.get("playwright")
            try:
                if isinstance(browser, Browser):
                    browser.close()
            except Exception:
                pass
            try:
                if isinstance(playwright, Playwright):
                    playwright.stop()
            except Exception:
                pass

    def _start_for_state(self, state: dict[str, object | None]) -> None:
        if state.get("browser") is not None:
            return
        playwright = sync_playwright().start()
        launch_kwargs: MutableMapping[str, object] = {
            "headless": self._settings.headless,
            "slow_mo": self._settings.slow_mo_ms,
        }
        if self._settings.proxy:
            launch_kwargs["proxy"] = {"server": self._settings.proxy}
        browser = playwright.chromium.launch(**launch_kwargs)
        state["playwright"] = playwright
        state["browser"] = browser
        state["started"] = True

    def start(self) -> None:
        state, ident = self._get_state()
        if state.get("browser") is not None:
            return
        with self._lock:
            state = self._states.get(ident)
            if state is None:
                state = {
                    "playwright": None,
                    "browser": None,
                    "started": False,
                }
                self._states[ident] = state
            if state.get("browser") is None:
                self._start_for_state(state)

    def stop(self) -> None:
        with self._lock:
            states = list(self._states.values())
            self._states = {}
        for state in states:
            browser = state.get("browser")
            playwright = state.get("playwright")
            try:
                if isinstance(browser, Browser):
                    browser.close()
            except Exception:
                pass
            try:
                if isinstance(playwright, Playwright):
                    playwright.stop()
            except Exception:
                pass

    def _resolve_user_agent(self, user_agent: str | None) -> str | None:
        if user_agent:
            return user_agent
        pool = self._settings.user_agents
        if not pool:
            return None
        return random.choice(tuple(pool))

    def new_context(
        self,
        *,
        user_agent: str | None = None,
        headers: Mapping[str, str] | None = None,
        proxy: str | None = None,
    ) -> BrowserContext:
        self.start()
        context_kwargs: MutableMapping[str, object] = {
            "locale": "ru-RU",
            "timezone_id": "Europe/Moscow",
            "viewport": {"width": self._settings.default_viewport[0], "height": self._settings.default_viewport[1]},
        }
        resolved_user_agent = self._resolve_user_agent(user_agent)
        if resolved_user_agent:
            context_kwargs["user_agent"] = resolved_user_agent
        if headers:
            context_kwargs["extra_http_headers"] = headers
        if proxy:
            context_kwargs["proxy"] = {"server": proxy}
        state, ident = self._get_state()
        browser = state.get("browser")
        if browser is None:
            with self._lock:
                state = self._states.get(ident)
                if state is None:
                    state = {
                        "playwright": None,
                        "browser": None,
                        "started": False,
                    }
                    self._states[ident] = state
                if state.get("browser") is None:
                    self._start_for_state(state)
                browser = state.get("browser")
        if not isinstance(browser, Browser):
            raise RuntimeError("browser_not_initialized")
        context = browser.new_context(**context_kwargs)
        stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU', 'ru']});
            Object.defineProperty(navigator, 'language', {get: () => 'ru-RU'});
            Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
            window.chrome = window.chrome || { runtime: {} };
        """
        try:
            context.add_init_script(stealth_script)
        except Exception:
            pass
        context.set_default_timeout(self._settings.context_timeout_ms)
        context.set_default_navigation_timeout(self._settings.navigation_timeout_ms)
        return context

    @contextmanager
    def page(
        self,
        *,
        user_agent: str | None = None,
        headers: Mapping[str, str] | None = None,
        proxy: str | None = None,
    ) -> Iterator[Page]:
        context = self.new_context(user_agent=user_agent, headers=headers, proxy=proxy)
        page = context.new_page()
        try:
            yield page
        finally:
            try:
                context.close()
            except Exception:
                pass


_DEFAULT_MANAGER = BrowserManager()


def get_browser_manager() -> BrowserManager:
    return _DEFAULT_MANAGER


def configure_browser(settings: BrowserSettings) -> None:
    _DEFAULT_MANAGER.configure(settings)

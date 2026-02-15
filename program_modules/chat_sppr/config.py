"""Configuration primitives for the SPPR chat module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value is not None else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = _env(key, str(default)).lower()
    if raw in {"1", "true", "yes", "y"}:
        return True
    if raw in {"0", "false", "no", "n"}:
        return False
    return default


@dataclass(slots=True)
class RetrievalConfig:
    top_k: int = 5
    decay: float = 0.92
    freshness_hours: int = 48
    fallback_answers: List[str] = field(
        default_factory=lambda: [
            "Я уточню детали и вернусь с ответом.",
            "Пока собираю данные, попробуйте сформулировать вопрос иначе.",
        ]
    )


@dataclass(slots=True)
class IntentConfig:
    threshold: float = 0.55
    suggestions: int = 3
    cooldown_sec: int = 30


@dataclass(slots=True)
class SessionConfig:
    idle_timeout: timedelta = timedelta(minutes=20)
    max_turns: int = 50
    autosave_interval: timedelta = timedelta(minutes=5)


@dataclass(slots=True)
class TelemetryConfig:
    enable: bool = True
    interval: timedelta = timedelta(seconds=30)
    window: int = 1000


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    database_url: str
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    intents: IntentConfig = field(default_factory=IntentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    extras: Dict[str, Any] = field(default_factory=dict)


def load_config(base_dir: Path | None = None) -> AppConfig:
    base = base_dir or Path(os.getenv("CHAT_SPPR_HOME", Path.cwd()))
    data_dir = base / "chat-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "chat_sppr.db"

    retrieval = RetrievalConfig(
        top_k=_env_int("CHAT_RETRIEVAL_TOPK", 5),
        decay=_env_float("CHAT_RETRIEVAL_DECAY", 0.92),
        freshness_hours=_env_int("CHAT_RETRIEVAL_FRESHNESS_H", 48),
    )
    intents = IntentConfig(
        threshold=_env_float("CHAT_INTENT_THRESHOLD", 0.55),
        suggestions=_env_int("CHAT_INTENT_SUGGESTIONS", 3),
        cooldown_sec=_env_int("CHAT_INTENT_COOLDOWN", 30),
    )
    session = SessionConfig(
        idle_timeout=timedelta(minutes=_env_int("CHAT_SESSION_IDLE_MIN", 20)),
        max_turns=_env_int("CHAT_SESSION_MAX_TURNS", 50),
        autosave_interval=timedelta(minutes=_env_int("CHAT_SESSION_AUTOSAVE_MIN", 5)),
    )
    telemetry = TelemetryConfig(
        enable=_env_bool("CHAT_TELEMETRY", True),
        interval=timedelta(seconds=_env_int("CHAT_TELEMETRY_INTERVAL", 30)),
        window=_env_int("CHAT_TELEMETRY_WINDOW", 1000),
    )
    extras = {
        "instance_id": _env("CHAT_INSTANCE_ID", "chat-sppr-local"),
        "profile": _env("CHAT_PROFILE", "default"),
    }

    return AppConfig(
        base_dir=base,
        data_dir=data_dir,
        database_url=f"sqlite:///{db_path}",
        retrieval=retrieval,
        intents=intents,
        session=session,
        telemetry=telemetry,
        extras=extras,
    )

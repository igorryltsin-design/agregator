"""Configuration primitives for Contextual Analytics SPPR."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any


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
class VectorizerConfig:
    default_dim: int = 48
    audio_bias: float = 0.6
    text_bias: float = 0.7
    radar_bias: float = 0.8
    visual_bias: float = 0.65


@dataclass(slots=True)
class ScoringConfig:
    similarity_threshold: float = 0.45
    prominence_decay: float = 0.92
    hot_window_minutes: int = 15


@dataclass(slots=True)
class TimelineConfig:
    max_events: int = 500
    aggregation_window: timedelta = timedelta(minutes=10)


@dataclass(slots=True)
class EvidenceConfig:
    narrative_max_len: int = 2048
    confidence_floor: float = 0.4
    refresh_interval: timedelta = timedelta(minutes=5)


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    database_url: str
    profile: str
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    extras: Dict[str, Any] = field(default_factory=dict)


def load_config(base_dir: Path | None = None) -> AppConfig:
    root = base_dir or Path(os.getenv("CONTEXT_ANALYTIC_HOME", Path.cwd()))
    data_dir = root / "context_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "context_analytic.db"

    vectorizer = VectorizerConfig(
        default_dim=_env_int("CONTEXT_VEC_DIM", 48),
        audio_bias=_env_float("CONTEXT_VEC_AUDIO", 0.6),
        text_bias=_env_float("CONTEXT_VEC_TEXT", 0.7),
        radar_bias=_env_float("CONTEXT_VEC_RADAR", 0.8),
        visual_bias=_env_float("CONTEXT_VEC_VISUAL", 0.65),
    )
    scoring = ScoringConfig(
        similarity_threshold=_env_float("CONTEXT_SIM_THRESHOLD", 0.45),
        prominence_decay=_env_float("CONTEXT_PROMINENCE_DECAY", 0.92),
        hot_window_minutes=_env_int("CONTEXT_HOT_WINDOW_MIN", 15),
    )
    timeline = TimelineConfig(
        max_events=_env_int("CONTEXT_TIMELINE_MAX", 500),
        aggregation_window=timedelta(
            minutes=_env_int("CONTEXT_TIMELINE_WINDOW_MIN", 10)
        ),
    )
    evidence = EvidenceConfig(
        narrative_max_len=_env_int("CONTEXT_EVIDENCE_LEN", 2048),
        confidence_floor=_env_float("CONTEXT_EVIDENCE_FLOOR", 0.4),
        refresh_interval=timedelta(
            minutes=_env_int("CONTEXT_EVIDENCE_REFRESH_MIN", 5)
        ),
    )
    extras = {
        "instance_id": _env("CONTEXT_INSTANCE_ID", "context-sppr-local"),
        "environment": _env("CONTEXT_ENV", "development"),
    }
    return AppConfig(
        base_dir=root,
        data_dir=data_dir,
        database_url=f"sqlite:///{db_path}",
        profile=_env("CONTEXT_PROFILE", "default"),
        vectorizer=vectorizer,
        scoring=scoring,
        timeline=timeline,
        evidence=evidence,
        extras=extras,
    )

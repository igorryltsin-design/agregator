"""Configuration helpers for Integrator SPPR."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict


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
class StreamConfig:
    max_bandwidth: float = 1000.0
    priority_levels: int = 5
    default_priority: int = 3
    smoothing_window: int = 60


@dataclass(slots=True)
class TopologyConfig:
    max_edges: int = 200
    heartbeat_interval: timedelta = timedelta(seconds=15)
    retry_policy: Dict[str, Any] = field(
        default_factory=lambda: {"base_delay": 2.0, "max_delay": 30.0, "factor": 2.0}
    )


@dataclass(slots=True)
class ReliabilityConfig:
    breach_threshold: float = 0.85
    latency_budget_ms: int = 500
    backpressure_limit: float = 0.9


@dataclass(slots=True)
class ReportingConfig:
    snapshot_interval: timedelta = timedelta(minutes=5)
    retention: timedelta = timedelta(days=7)


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    database_url: str
    profile: str
    stream: StreamConfig = field(default_factory=StreamConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    extras: Dict[str, Any] = field(default_factory=dict)


def load_config(base_dir: Path | None = None) -> AppConfig:
    base = base_dir or Path(os.getenv("INTEGRATOR_HOME", Path.cwd()))
    data_dir = base / "integrator-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "integrator_sppr.db"

    stream = StreamConfig(
        max_bandwidth=_env_float("INTEGRATOR_MAX_BW", 1000.0),
        priority_levels=_env_int("INTEGRATOR_PRI_LEVELS", 5),
        default_priority=_env_int("INTEGRATOR_PRI_DEFAULT", 3),
        smoothing_window=_env_int("INTEGRATOR_SMOOTH_WINDOW", 60),
    )
    topology = TopologyConfig(
        max_edges=_env_int("INTEGRATOR_MAX_EDGES", 200),
        heartbeat_interval=timedelta(
            seconds=_env_int("INTEGRATOR_HEARTBEAT_SEC", 15)
        ),
        retry_policy={
            "base_delay": _env_float("INTEGRATOR_RETRY_BASE", 2.0),
            "max_delay": _env_float("INTEGRATOR_RETRY_MAX", 30.0),
            "factor": _env_float("INTEGRATOR_RETRY_FACTOR", 2.0),
        },
    )
    reliability = ReliabilityConfig(
        breach_threshold=_env_float("INTEGRATOR_RELIABILITY_THRESHOLD", 0.85),
        latency_budget_ms=_env_int("INTEGRATOR_LATENCY_BUDGET_MS", 500),
        backpressure_limit=_env_float("INTEGRATOR_BACKPRESSURE_LIMIT", 0.9),
    )
    reporting = ReportingConfig(
        snapshot_interval=timedelta(
            minutes=_env_int("INTEGRATOR_REPORT_INTERVAL_MIN", 5)
        ),
        retention=timedelta(days=_env_int("INTEGRATOR_REPORT_RETENTION_DAYS", 7)),
    )
    extras = {
        "instance_id": _env("INTEGRATOR_INSTANCE_ID", "integrator-local"),
        "environment": _env("INTEGRATOR_ENVIRONMENT", "development"),
    }

    return AppConfig(
        base_dir=base,
        data_dir=data_dir,
        database_url=f"sqlite:///{db_path}",
        profile=_env("INTEGRATOR_PROFILE", "default"),
        stream=stream,
        topology=topology,
        reliability=reliability,
        reporting=reporting,
        extras=extras,
    )

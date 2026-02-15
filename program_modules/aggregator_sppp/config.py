"""Configuration helpers for the Aggregator SPPR module."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict


def _env(key: str, default: str) -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else default


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
    value = _env(key, str(default)).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return default


@dataclass(slots=True)
class TelemetryConfig:
    enable_metrics: bool = True
    export_interval: timedelta = timedelta(seconds=30)
    window_size: int = 5000


@dataclass(slots=True)
class IngestionConfig:
    batch_size: int = 128
    default_confidence: float = 0.5
    default_intensity: float = 0.7
    deduplicate_window: timedelta = timedelta(seconds=15)


@dataclass(slots=True)
class FusionConfig:
    alignment_threshold: float = 0.65
    coherence_threshold: float = 0.5
    max_group_span: timedelta = timedelta(minutes=5)
    replay_window: timedelta = timedelta(hours=48)


@dataclass(slots=True)
class GraphConfig:
    similarity_threshold: float = 0.6
    max_neighbors: int = 20
    decay_factor: float = 0.92


@dataclass(slots=True)
class HistoryConfig:
    retention_days: int = 30
    auto_snapshot_interval: timedelta = timedelta(minutes=10)


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    database_url: str
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def db_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url[10:]).expanduser()
        return Path(self.base_dir / "aggregator_sppr.db")


def load_config(base_dir: Path | None = None) -> AppConfig:
    base_dir = base_dir or Path(os.getenv("AGGREGATOR_HOME", Path.cwd()))
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = _env("AGGREGATOR_DB_PATH", str(data_dir / "aggregator_sppr.db"))
    telemetry = TelemetryConfig(
        enable_metrics=_env_bool("AGGREGATOR_METRICS", True),
        export_interval=timedelta(seconds=_env_int("AGGREGATOR_METRICS_INTERVAL", 30)),
        window_size=_env_int("AGGREGATOR_METRICS_WINDOW", 5000),
    )
    ingestion = IngestionConfig(
        batch_size=_env_int("AGGREGATOR_BATCH_SIZE", 128),
        default_confidence=_env_float("AGGREGATOR_DEFAULT_CONFIDENCE", 0.5),
        default_intensity=_env_float("AGGREGATOR_DEFAULT_INTENSITY", 0.7),
        deduplicate_window=timedelta(
            seconds=_env_int("AGGREGATOR_DEDUP_WINDOW", 15)
        ),
    )
    fusion = FusionConfig(
        alignment_threshold=_env_float("AGGREGATOR_ALIGN_THRESHOLD", 0.65),
        coherence_threshold=_env_float("AGGREGATOR_COHERENCE_THRESHOLD", 0.5),
        max_group_span=timedelta(
            minutes=_env_int("AGGREGATOR_MAX_GROUP_SPAN_MIN", 5)
        ),
        replay_window=timedelta(hours=_env_int("AGGREGATOR_REPLAY_WINDOW_H", 48)),
    )
    graph = GraphConfig(
        similarity_threshold=_env_float("AGGREGATOR_GRAPH_THRESHOLD", 0.6),
        max_neighbors=_env_int("AGGREGATOR_GRAPH_NEIGHBORS", 20),
        decay_factor=_env_float("AGGREGATOR_GRAPH_DECAY", 0.92),
    )
    history = HistoryConfig(
        retention_days=_env_int("AGGREGATOR_HISTORY_DAYS", 30),
        auto_snapshot_interval=timedelta(
            minutes=_env_int("AGGREGATOR_HISTORY_SNAPSHOT_MIN", 10)
        ),
    )
    extra = {
        "profile": _env("AGGREGATOR_PROFILE", "default"),
        "instance_id": _env("AGGREGATOR_INSTANCE_ID", "agg-sppr-local"),
    }

    return AppConfig(
        base_dir=base_dir,
        database_url=f"sqlite:///{db_path}",
        telemetry=telemetry,
        ingestion=ingestion,
        fusion=fusion,
        graph=graph,
        history=history,
        extra=extra,
    )

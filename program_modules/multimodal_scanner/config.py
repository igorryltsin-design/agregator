"""Configuration for the multimodal scanner module."""

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
    value = _env(key, str(default)).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return default


@dataclass(slots=True)
class SourceConfig:
    root_paths: list[str]
    max_depth: int = 4
    follow_symlinks: bool = False
    include_patterns: list[str] = field(default_factory=lambda: ["*.pdf", "*.docx", "*.wav", "*.png", "*.mp3"])


@dataclass(slots=True)
class FingerprintConfig:
    block_size: int = 8192
    rolling_window: int = 256
    hash_alg: str = "sha256"


@dataclass(slots=True)
class ClassifierConfig:
    text_threshold: float = 0.6
    audio_threshold: float = 0.5
    image_threshold: float = 0.55


@dataclass(slots=True)
class StorageConfig:
    database_url: str
    data_dir: Path


@dataclass(slots=True)
class AppConfig:
    base_dir: Path
    storage: StorageConfig
    source: SourceConfig
    fingerprint: FingerprintConfig = field(default_factory=FingerprintConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    graph_threshold: float = 0.65
    stats_window: timedelta = timedelta(hours=6)
    extras: Dict[str, Any] = field(default_factory=dict)


def load_config(base_dir: Path | None = None) -> AppConfig:
    base = base_dir or Path(os.getenv("SCANNER_HOME", Path.cwd()))
    data_dir = base / "scanner-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "scanner.db"

    root_paths = _env("SCANNER_ROOTS", str(base / "library")).split(os.pathsep)
    source = SourceConfig(
        root_paths=[path for path in root_paths if path],
        max_depth=_env_int("SCANNER_MAX_DEPTH", 4),
        follow_symlinks=_env_bool("SCANNER_FOLLOW_SYMLINKS", False),
        include_patterns=_env("SCANNER_PATTERNS", "*.pdf;*.docx;*.png;*.wav").split(";"),
    )
    fingerprint = FingerprintConfig(
        block_size=_env_int("SCANNER_BLOCK_SIZE", 8192),
        rolling_window=_env_int("SCANNER_ROLLING_WINDOW", 256),
        hash_alg=_env("SCANNER_HASH", "sha256"),
    )
    classifier = ClassifierConfig(
        text_threshold=_env_float("SCANNER_TEXT_THRESHOLD", 0.6),
        audio_threshold=_env_float("SCANNER_AUDIO_THRESHOLD", 0.5),
        image_threshold=_env_float("SCANNER_IMAGE_THRESHOLD", 0.55),
    )
    storage = StorageConfig(database_url=f"sqlite:///{db_path}", data_dir=data_dir)
    extras = {
        "instance_id": _env("SCANNER_INSTANCE_ID", "scanner-local"),
        "profile": _env("SCANNER_PROFILE", "default"),
    }
    return AppConfig(
        base_dir=base,
        storage=storage,
        source=source,
        fingerprint=fingerprint,
        classifier=classifier,
        graph_threshold=_env_float("SCANNER_GRAPH_THRESHOLD", 0.65),
        stats_window=timedelta(hours=_env_int("SCANNER_STATS_WINDOW_H", 6)),
        extras=extras,
    )

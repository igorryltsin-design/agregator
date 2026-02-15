"""Logging utilities for Contextual Analytics SPPR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class LoggingConfig:
    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(config: LoggingConfig | None = None) -> Dict[str, logging.Logger]:
    cfg = config or LoggingConfig()
    logging.basicConfig(level=cfg.level, format=cfg.fmt)
    return {
        "context": logging.getLogger("context"),
        "vectorizer": logging.getLogger("context.vectorizer"),
        "scoring": logging.getLogger("context.scoring"),
        "timeline": logging.getLogger("context.timeline"),
        "evidence": logging.getLogger("context.evidence"),
        "api": logging.getLogger("context.api"),
    }

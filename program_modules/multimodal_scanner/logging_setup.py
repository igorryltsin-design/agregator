"""Logging utilities for the scanner module."""

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
        "scanner": logging.getLogger("scanner"),
        "walker": logging.getLogger("scanner.walker"),
        "fingerprint": logging.getLogger("scanner.fingerprint"),
        "classifier": logging.getLogger("scanner.classifier"),
        "graph": logging.getLogger("scanner.graph"),
    }

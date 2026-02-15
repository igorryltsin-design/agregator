"""Logging configuration for Integrator SPPR."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict


@dataclass(slots=True)
class LoggerConfig:
    level: int = logging.INFO
    fmt: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(config: LoggerConfig | None = None) -> Dict[str, logging.Logger]:
    cfg = config or LoggerConfig()
    logging.basicConfig(level=cfg.level, format=cfg.fmt)
    return {
        "integrator": logging.getLogger("integrator"),
        "streams": logging.getLogger("integrator.streams"),
        "topology": logging.getLogger("integrator.topology"),
        "monitor": logging.getLogger("integrator.monitor"),
        "report": logging.getLogger("integrator.report"),
    }

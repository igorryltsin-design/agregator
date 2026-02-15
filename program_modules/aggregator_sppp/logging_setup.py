"""Custom logging configuration for the Aggregator SPPR module."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from logging import Logger
from typing import Dict

DEFAULT_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


@dataclass(slots=True)
class LoggerConfig:
    level: int = logging.INFO
    fmt: str = DEFAULT_FORMAT
    enable_console: bool = True


def configure_logging(config: LoggerConfig | None = None) -> Dict[str, Logger]:
    config = config or LoggerConfig()
    logging.basicConfig(
        level=config.level,
        format=config.fmt,
        stream=sys.stdout,
    )
    logging.debug("Logging initialized with level %s", config.level)
    return {
        "aggregator": logging.getLogger("aggregator"),
        "pipeline": logging.getLogger("aggregator.pipeline"),
        "database": logging.getLogger("aggregator.database"),
        "telemetry": logging.getLogger("aggregator.telemetry"),
    }

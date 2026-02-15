"""Logging helpers for chat module."""

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
        "chat": logging.getLogger("chat"),
        "session": logging.getLogger("chat.session"),
        "intent": logging.getLogger("chat.intent"),
        "retrieval": logging.getLogger("chat.retrieval"),
        "api": logging.getLogger("chat.api"),
    }

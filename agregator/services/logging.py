"""Logging helpers for Agregator."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, List, Optional

from flask import Flask


class SensitiveDataFilter(logging.Filter):
    """Удаляет токены и пароли из строк логов."""

    _PATTERNS: Iterable[tuple[re.Pattern[str], str]] = (
        (re.compile(r"(authorization=)([^\s]+)", re.I), r"\1***"),
        (re.compile(r"(api[_-]?key=)([^&\s]+)", re.I), r"\1***"),
        (re.compile(r"(access[_-]?token=)([^&\s]+)", re.I), r"\1***"),
        (re.compile(r"(password=)([^&\s]+)", re.I), r"\1***"),
        (re.compile(r"Bearer\s+[A-Za-z0-9._-]+"), "Bearer ***"),
    )

    def __init__(self) -> None:
        super().__init__(name="SensitiveDataFilter")

    @staticmethod
    def _sanitize_value(value: object) -> object:
        if isinstance(value, str):
            sanitized = value
            for pattern, repl in SensitiveDataFilter._PATTERNS:
                sanitized = pattern.sub(repl, sanitized)
            return sanitized
        if isinstance(value, (list, tuple)):
            return type(value)(SensitiveDataFilter._sanitize_value(v) for v in value)
        if isinstance(value, dict):
            return {k: SensitiveDataFilter._sanitize_value(v) for k, v in value.items()}
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._sanitize_value(record.msg)
        if record.args:
            record.args = self._sanitize_value(record.args)
        return True


class JsonFormatter(logging.Formatter):
    """Форматтер, выводящий JSON-строки."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        data = {
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        if record.__dict__.get("extra"):
            data["extra"] = record.__dict__["extra"]
        return json.dumps(data, ensure_ascii=False)


def configure_logging(
    app: Flask,
    log_file_path: Path,
    *,
    level: str | int | None = None,
    sentry_dsn: str | None = None,
    sentry_environment: str | None = None,
) -> None:
    """Attach a rotating file handler to the Flask logger with structured output."""

    log_level = level or app.config.get("LOG_LEVEL") or "INFO"
    resolved_level = logging.getLevelName(str(log_level).upper())
    if isinstance(resolved_level, str):  # unknown name returns string
        resolved_level = logging.INFO

    app.logger.setLevel(resolved_level)
    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    handler = get_rotating_log_handler(app, log_file_path)
    if handler is None:
        handler = RotatingFileHandler(
            log_file_path,
            maxBytes=100 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        app.logger.addHandler(handler)

    handler.setLevel(resolved_level)
    if not any(isinstance(f, SensitiveDataFilter) for f in handler.filters):
        handler.addFilter(SensitiveDataFilter())
    handler.setFormatter(JsonFormatter())

    if sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.logging import LoggingIntegration

            sentry_logging = LoggingIntegration(
                level=resolved_level,
                event_level=logging.ERROR,
            )
            sentry_sdk.init(
                dsn=sentry_dsn,
                environment=sentry_environment,
                integrations=[sentry_logging],
                traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0")),
            )
            app.logger.info("Sentry инициализирован")
        except Exception as exc:  # noqa: BLE001
            app.logger.warning("Не удалось инициализировать Sentry: %s", exc)


def get_rotating_log_handler(app: Flask, log_file_path: Path) -> Optional[RotatingFileHandler]:
    """Return the configured rotating handler for the app logger, if any."""
    for handler in app.logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            base_filename = getattr(handler, "baseFilename", "")
            if Path(base_filename).resolve() == log_file_path.resolve():
                return handler
    return None


def list_system_log_files(log_dir: Path, log_file_path: Path) -> List[dict]:
    """Enumerate log files with metadata for diagnostics endpoints."""
    files: List[dict] = []
    try:
        base = log_file_path.name
        for entry in sorted(log_dir.glob(f"{base}*")):
            if not entry.is_file():
                continue
            try:
                stat = entry.stat()
                files.append(
                    {
                        "name": entry.name,
                        "size": stat.st_size,
                        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "rotated": entry.name != base,
                    }
                )
            except Exception:
                continue
    except Exception:
        pass
    return files


def tail_log_file(path: Path, max_lines: int = 200) -> List[str]:
    """Read the last `max_lines` from the given log file."""
    if max_lines <= 0 or not path.exists() or not path.is_file():
        return []
    max_lines = min(max_lines, 2000)
    chunk_size = 8192
    buffer = b""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        file_size = fh.tell()
        remaining = file_size
        newlines = 0
        while remaining > 0 and newlines <= max_lines:
            read_size = min(chunk_size, remaining)
            remaining -= read_size
            fh.seek(remaining)
            chunk = fh.read(read_size)
            buffer = chunk + buffer
            newlines = buffer.count(b"\n")
        text = buffer.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return lines[-max_lines:]


def resolve_log_name(log_dir: Path, log_file_path: Path, candidate: Optional[str]) -> Optional[Path]:
    """Validate and resolve a log filename inside the log directory."""
    name = (candidate or "").strip() or log_file_path.name
    if name == "current":
        name = log_file_path.name
    if not name.startswith(log_file_path.name):
        return None
    path = (log_dir / name).resolve()
    try:
        path.relative_to(log_dir.resolve())
    except Exception:
        return None
    return path if path.exists() and path.is_file() else None

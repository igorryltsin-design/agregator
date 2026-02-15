"""Helpers for storing OSINT artifacts on disk."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from flask import current_app

_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _instance_path() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(current_app.instance_path))
    except RuntimeError:
        pass
    for raw in (
        os.getenv("AGREGATOR_INSTANCE_PATH"),
        os.getenv("INSTANCE_PATH"),
    ):
        if raw:
            candidates.append(Path(raw))
    candidates.extend(
        [
            Path(os.getcwd()) / "instance",
            Path.home() / ".agregator" / "instance",
            Path(tempfile.gettempdir()) / "agregator-instance",
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve(strict=False))
        if normalized in seen:
            continue
        seen.add(normalized)
        if _is_writable_dir(candidate):
            return candidate
    raise OSError("Unable to find writable instance directory for OSINT artifacts")


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def artifact_root() -> Path:
    """Return the root directory for OSINT artifacts, ensuring it exists."""
    root = _instance_path() / "osint_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sanitize_segment(segment: str) -> str:
    cleaned = _SEGMENT_RE.sub("_", segment.strip())
    return cleaned or "artifact"


def artifact_subdir(kind: str) -> Path:
    """Return a subdirectory under the artifact root for the given kind."""
    safe_kind = _sanitize_segment(kind)
    subdir = artifact_root() / safe_kind
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def relative_artifact_path(path: Path) -> str:
    """Return a path relative to the artifact root."""
    root = artifact_root()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return path.name


def resolve_artifact_path(relative_path: str) -> Path:
    """Resolve a relative artifact path to an absolute path under the root."""
    root = artifact_root()
    candidate = root / relative_path
    resolved = candidate.resolve()
    if not str(resolved).startswith(str(root.resolve())):
        raise ValueError("Invalid artifact path")
    return resolved

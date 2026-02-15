"""Response envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict


def iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.utcnow()
    return dt.replace(microsecond=0).isoformat() + "Z"


@dataclass
class Envelope:
    ok: bool
    payload: Dict[str, Any]
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "payload": self.payload, "error": self.error, "timestamp": iso()}


def success(payload: Dict[str, Any]) -> Envelope:
    return Envelope(True, payload)


def failure(message: str) -> Envelope:
    return Envelope(False, {}, message)

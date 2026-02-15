"""Input validators for ingestion and API payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from .. import utils


class ValidationError(Exception):
    pass


@dataclass(slots=True)
class SourcePayload:
    name: str
    modality: str
    reliability: float
    priority: int
    description: Optional[str]

    @classmethod
    def from_dict(cls, data: dict) -> "SourcePayload":
        required = {"name", "modality"}
        missing = required - data.keys()
        if missing:
            raise ValidationError(f"Missing fields: {', '.join(sorted(missing))}")
        return cls(
            name=data["name"],
            modality=data["modality"],
            reliability=float(data.get("reliability", 0.5)),
            priority=int(data.get("priority", 5)),
            description=data.get("description"),
        )


@dataclass(slots=True)
class SignalPayload:
    source_name: str
    signal_type: str
    payload_ref: str
    recorded_at: datetime
    metadata: dict
    intensity: float

    @classmethod
    def from_dict(cls, data: dict) -> "SignalPayload":
        required = {"source_name", "signal_type", "payload_ref", "recorded_at"}
        missing = required - data.keys()
        if missing:
            raise ValidationError(f"Missing fields: {', '.join(sorted(missing))}")
        return cls(
            source_name=data["source_name"],
            signal_type=data["signal_type"],
            payload_ref=data["payload_ref"],
            recorded_at=utils.normalize_timestamp(data["recorded_at"]),
            metadata=data.get("metadata", {}),
            intensity=float(data.get("intensity", 0.5)),
        )


@dataclass(slots=True)
class FusionWindowPayload:
    window_start: datetime
    window_end: datetime
    required_modalities: List[str]

    @classmethod
    def from_dict(cls, data: dict) -> "FusionWindowPayload":
        now = utils.utc_now()
        default_start = now - timedelta(minutes=5)
        window_start = utils.normalize_timestamp(
            data.get("window_start", default_start)
        )
        window_end = utils.normalize_timestamp(data.get("window_end", now))
        if window_end <= window_start:
            window_end = window_start + timedelta(minutes=1)
        return cls(
            window_start=window_start,
            window_end=window_end,
            required_modalities=list(data.get("required_modalities", [])),
        )

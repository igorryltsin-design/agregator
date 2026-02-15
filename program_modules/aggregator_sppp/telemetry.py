"""Telemetry utilities for collecting runtime metrics."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .config import TelemetryConfig


@dataclass
class MetricSample:
    timestamp: float
    value: float


@dataclass
class RollingMetric:
    name: str
    samples: List[MetricSample] = field(default_factory=list)
    max_samples: int = 5000

    def add(self, value: float) -> None:
        self.samples.append(MetricSample(time.time(), value))
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]

    def average(self) -> float:
        if not self.samples:
            return 0.0
        return sum(sample.value for sample in self.samples) / len(self.samples)

    def latest(self) -> float:
        return self.samples[-1].value if self.samples else 0.0


class Telemetry:
    """Lightweight metric collector with periodic export callback."""

    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.metrics: Dict[str, RollingMetric] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._export_callback = None

    def configure_export(self, callback) -> None:
        self._export_callback = callback

    def start(self) -> None:
        if not self.config.enable_metrics or self._thread:
            return

        def _loop():
            while not self._stop_event.wait(self.config.export_interval.total_seconds()):
                self._export()
            self._export()

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop_event.set()
            self._thread.join(timeout=2)
            self._thread = None

    def observe(self, name: str, value: float) -> None:
        if not self.config.enable_metrics:
            return
        with self._lock:
            metric = self.metrics.setdefault(
                name, RollingMetric(name=name, max_samples=self.config.window_size)
            )
            metric.add(value)

    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {
                name: {
                    "average": metric.average(),
                    "latest": metric.latest(),
                    "sample_count": len(metric.samples),
                }
                for name, metric in self.metrics.items()
            }

    def _export(self) -> None:
        if self._export_callback:
            try:
                self._export_callback(self.snapshot())
            except Exception:  # pragma: no cover
                pass

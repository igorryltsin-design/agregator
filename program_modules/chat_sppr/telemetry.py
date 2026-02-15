"""Telemetry collector for chat module."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class MetricSample:
    timestamp: float
    value: float


@dataclass
class Metric:
    name: str
    max_samples: int
    samples: List[MetricSample] = field(default_factory=list)

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
    def __init__(self, max_samples: int = 1000):
        self.metrics: Dict[str, Metric] = {}
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._callback: Callable[[Dict[str, Dict[str, float]]], None] | None = None
        self.interval = 30.0

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            metric = self.metrics.setdefault(name, Metric(name, self.max_samples))
            metric.add(value)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                name: {
                    "average": metric.average(),
                    "latest": metric.latest(),
                    "samples": len(metric.samples),
                }
                for name, metric in self.metrics.items()
            }

    def configure_export(self, interval: float, callback: Callable[[Dict[str, Dict[str, float]]], None]) -> None:
        self.interval = interval
        self._callback = callback

    def start(self) -> None:
        if self._thread or not self._callback:
            return

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._callback(self.snapshot())
        self._callback(self.snapshot())

    def ensure_running(self) -> None:
        if self._thread or not self._callback:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None
            self._stop.clear()

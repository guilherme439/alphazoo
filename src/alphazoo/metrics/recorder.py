from __future__ import annotations

import threading

from .types import MetricEntry, MetricType


class MetricsRecorder:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics: dict[str, MetricEntry] = {}

    def scalar(self, key: str, value: float) -> None:
        with self._lock:
            self._metrics[key] = MetricEntry(MetricType.SCALAR, value)

    def mean(self, key: str, value: float) -> None:
        with self._lock:
            existing = self._metrics.get(key)
            if existing is not None and existing.type == MetricType.MEAN:
                existing.value += value
                existing.count += 1
            else:
                self._metrics[key] = MetricEntry(MetricType.MEAN, value)

    def counter(self, key: str, value: float = 1) -> None:
        with self._lock:
            existing = self._metrics.get(key)
            if existing is not None and existing.type == MetricType.COUNTER:
                existing.value += value
            else:
                self._metrics[key] = MetricEntry(MetricType.COUNTER, value)

    def lifetime_counter(self, key: str, value: float = 1) -> None:
        with self._lock:
            existing = self._metrics.get(key)
            if existing is not None and existing.type == MetricType.LIFETIME_COUNTER:
                existing.value += value
            else:
                self._metrics[key] = MetricEntry(MetricType.LIFETIME_COUNTER, value)

    def lifetime_scalar(self, key: str, value: float) -> None:
        with self._lock:
            self._metrics[key] = MetricEntry(MetricType.LIFETIME_SCALAR, value)

    def drain(self) -> dict[str, MetricEntry]:
        with self._lock:
            snapshot = dict(self._metrics)

            # Keep lifetime metrics, clear the rest
            self._metrics = {
                k: MetricEntry(v.type, v.value)
                for k, v in self._metrics.items()
                if v.type in (MetricType.LIFETIME_COUNTER, MetricType.LIFETIME_SCALAR)
            }

            return snapshot

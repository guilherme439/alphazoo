from __future__ import annotations

from typing import Optional

from .types import MetricEntry, MetricType


class MetricsStore:

    def __init__(self, public_keys: set[str]) -> None:
        self._metrics: dict[str, MetricEntry] = {}
        self._public_keys = public_keys

    def ingest(self, batches: list[dict[str, MetricEntry]]) -> None:
        for batch in batches:
            for key, entry in batch.items():
                existing = self._metrics.get(key)
                if existing is None:
                    self._metrics[key] = MetricEntry(
                        entry.type, entry.value, entry.count,
                    )
                    continue

                match entry.type:
                    case MetricType.SCALAR | MetricType.LIFETIME_SCALAR:
                        existing.value = entry.value
                    case MetricType.COUNTER | MetricType.LIFETIME_COUNTER:
                        existing.value += entry.value
                    case MetricType.MEAN:
                        existing.value += entry.value
                        existing.count += entry.count

    def get_public(self) -> dict[str, float]:
        return self._resolve(public=True)

    def get_internal(self) -> dict[str, float]:
        return self._resolve(public=False)

    def get_all(self) -> dict[str, float]:
        return self._resolve(public=None)

    def add(self, key: str, value: float) -> None:
        self._metrics[key] = MetricEntry(MetricType.SCALAR, value)

    def clear(self) -> None:
        self._metrics = {
            k: MetricEntry(v.type, v.value)
            for k, v in self._metrics.items()
            if v.type in (MetricType.LIFETIME_COUNTER, MetricType.LIFETIME_SCALAR)
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve(self, public: Optional[bool]) -> dict[str, float]:
        result: dict[str, float] = {}
        for key, entry in self._metrics.items():
            if public is True and key not in self._public_keys:
                continue
            if public is False and key in self._public_keys:
                continue
            if entry.type == MetricType.MEAN:
                result[key] = entry.value / entry.count if entry.count > 0 else 0.0
            else:
                result[key] = entry.value
        return result

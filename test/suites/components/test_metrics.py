import threading

import pytest

from alphazoo.metrics import MetricEntry, MetricType, MetricsRecorder, MetricsStore


# ---------------------------------------------------------------------------
# MetricsRecorder
# ---------------------------------------------------------------------------


class TestRecorderScalar:
    def test_last_value_wins(self) -> None:
        r = MetricsRecorder()
        r.scalar("x", 1.0)
        r.scalar("x", 2.0)
        result = r.drain()
        assert result["x"].value == 2.0

    def test_cleared_after_drain(self) -> None:
        r = MetricsRecorder()
        r.scalar("x", 5.0)
        r.drain()
        result = r.drain()
        assert "x" not in result


class TestRecorderMean:
    def test_sum_and_count(self) -> None:
        r = MetricsRecorder()
        r.mean("x", 2.0)
        r.mean("x", 4.0)
        r.mean("x", 6.0)
        result = r.drain()
        assert result["x"].value == pytest.approx(12.0)
        assert result["x"].count == 3

    def test_cleared_after_drain(self) -> None:
        r = MetricsRecorder()
        r.mean("x", 10.0)
        r.drain()
        result = r.drain()
        assert "x" not in result


class TestRecorderCounter:
    def test_accumulation(self) -> None:
        r = MetricsRecorder()
        r.counter("x", 3)
        r.counter("x", 7)
        result = r.drain()
        assert result["x"].value == 10

    def test_default_increment(self) -> None:
        r = MetricsRecorder()
        r.counter("x")
        r.counter("x")
        r.counter("x")
        result = r.drain()
        assert result["x"].value == 3

    def test_cleared_after_drain(self) -> None:
        r = MetricsRecorder()
        r.counter("x", 5)
        r.drain()
        result = r.drain()
        assert "x" not in result


class TestRecorderLifetimeCounter:
    def test_survives_drain(self) -> None:
        r = MetricsRecorder()
        r.lifetime_counter("x", 5)
        r.drain()
        r.lifetime_counter("x", 3)
        result = r.drain()
        assert result["x"].value == 8

    def test_present_in_every_drain(self) -> None:
        r = MetricsRecorder()
        r.lifetime_counter("x", 1)
        r.drain()
        result = r.drain()
        assert "x" in result
        assert result["x"].value == 1


class TestRecorderLifetimeScalar:
    def test_survives_drain(self) -> None:
        r = MetricsRecorder()
        r.lifetime_scalar("x", 10.0)
        r.drain()
        r.lifetime_scalar("x", 20.0)
        result = r.drain()
        assert result["x"].value == 20.0

    def test_present_in_every_drain(self) -> None:
        r = MetricsRecorder()
        r.lifetime_scalar("x", 42.0)
        r.drain()
        result = r.drain()
        assert result["x"].value == 42.0


class TestRecorderThreadSafety:
    def test_concurrent_writes(self) -> None:
        r = MetricsRecorder()
        n = 1000

        def writer() -> None:
            for _ in range(n):
                r.counter("x")

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        result = r.drain()
        assert result["x"].value == 4 * n


# ---------------------------------------------------------------------------
# MetricsStore
# ---------------------------------------------------------------------------


class TestStoreIngest:
    def test_scalar_last_wins(self) -> None:
        s = MetricsStore(set())
        s.ingest([
            {"x": MetricEntry(MetricType.SCALAR, 1.0)},
            {"x": MetricEntry(MetricType.SCALAR, 2.0)},
        ])
        assert s.get_all()["x"] == 2.0

    def test_counter_sums(self) -> None:
        s = MetricsStore(set())
        s.ingest([
            {"x": MetricEntry(MetricType.COUNTER, 3.0)},
            {"x": MetricEntry(MetricType.COUNTER, 7.0)},
        ])
        assert s.get_all()["x"] == 10.0

    def test_mean_weighted(self) -> None:
        # Gamer A: sum=12.0 over 3 samples (avg 4.0), Gamer B: sum=10.0 over 1 sample
        s = MetricsStore(set())
        s.ingest([
            {"x": MetricEntry(MetricType.MEAN, 12.0, count=3)},
            {"x": MetricEntry(MetricType.MEAN, 10.0, count=1)},
        ])
        assert s.get_all()["x"] == pytest.approx(5.5)  # (12 + 10) / 4

    def test_multiple_keys(self) -> None:
        s = MetricsStore(set())
        s.ingest([
            {
                "a": MetricEntry(MetricType.SCALAR, 1.0),
                "b": MetricEntry(MetricType.COUNTER, 5.0),
            },
        ])
        assert s.get_all()["a"] == 1.0
        assert s.get_all()["b"] == 5.0


class TestStoreVisibility:
    def test_public_only(self) -> None:
        s = MetricsStore(public_keys={"pub"})
        s.ingest([{
            "pub": MetricEntry(MetricType.SCALAR, 1.0),
            "priv": MetricEntry(MetricType.SCALAR, 2.0),
        }])
        public = s.get_public()
        assert "pub" in public
        assert "priv" not in public

    def test_internal_only(self) -> None:
        s = MetricsStore(public_keys={"pub"})
        s.ingest([{
            "pub": MetricEntry(MetricType.SCALAR, 1.0),
            "priv": MetricEntry(MetricType.SCALAR, 2.0),
        }])
        internal = s.get_internal()
        assert "priv" in internal
        assert "pub" not in internal


class TestStoreClear:
    def test_clears_regular_metrics(self) -> None:
        s = MetricsStore(set())
        s.ingest([{
            "g": MetricEntry(MetricType.SCALAR, 1.0),
            "c": MetricEntry(MetricType.COUNTER, 5.0),
            "m": MetricEntry(MetricType.MEAN, 3.0),
        }])
        s.clear()
        assert s.get_all() == {}

    def test_preserves_lifetime_metrics(self) -> None:
        s = MetricsStore(set())
        s.ingest([{
            "g": MetricEntry(MetricType.SCALAR, 1.0),
            "lc": MetricEntry(MetricType.LIFETIME_COUNTER, 10.0),
            "lg": MetricEntry(MetricType.LIFETIME_SCALAR, 42.0),
        }])
        s.clear()
        result = s.get_all()
        assert "g" not in result
        assert result["lc"] == 10.0
        assert result["lg"] == 42.0

    def test_lifetime_counter_accumulates_across_clears(self) -> None:
        s = MetricsStore(set())
        s.ingest([{"x": MetricEntry(MetricType.LIFETIME_COUNTER, 5.0)}])
        s.clear()
        s.ingest([{"x": MetricEntry(MetricType.LIFETIME_COUNTER, 3.0)}])
        assert s.get_all()["x"] == 8.0


class TestStoreRecorderIntegration:
    def test_recorder_to_store_flow(self) -> None:
        r1 = MetricsRecorder()
        r2 = MetricsRecorder()

        r1.counter("rollout/games", 5)
        r1.mean("rollout/tree_size", 100.0)
        r1.mean("rollout/tree_size", 200.0)

        r2.counter("rollout/games", 3)
        r2.mean("rollout/tree_size", 300.0)

        store = MetricsStore(public_keys={"rollout/games"})
        store.ingest([r1.drain(), r2.drain()])

        public = store.get_public()
        assert public["rollout/games"] == 8
        assert "rollout/tree_size" not in public

        internal = store.get_internal()
        assert internal["rollout/tree_size"] == pytest.approx(200.0)  # (100+200+300) / 3
        assert "rollout/games" not in internal

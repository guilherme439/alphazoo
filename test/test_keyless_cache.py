import threading

import torch

from alphazoo.inference.caches.keyless_cache import KeylessCache


# ---------------------------------------------------------------------------
# hash_state
# ---------------------------------------------------------------------------


class TestHashState:
    def test_equal_tensors_hash_equal(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert cache.hash_state(a) == cache.hash_state(b)

    def test_different_tensors_hash_differently(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 4.0])
        assert cache.hash_state(a) != cache.hash_state(b)


# ---------------------------------------------------------------------------
# get / put / hashed variants
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_put_then_get(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        key = torch.tensor([1.0, 2.0, 3.0])
        cache.put((key, "value"))
        assert cache.get(key) == "value"

    def test_absent_key_misses(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        assert cache.get(torch.tensor([9.0])) is None

    def test_hashed_round_trip(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        key = torch.tensor([1.0, 2.0, 3.0])
        h = cache.hash_state(key)
        cache.hashed_put(h, "value")
        assert cache.hashed_get(h) == "value"

    def test_wrapper_matches_hashed(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        key = torch.tensor([1.0, 2.0, 3.0])
        cache.put((key, 42))
        assert cache.get(key) == cache.hashed_get(cache.hash_state(key))

    def test_hashed_put_visible_via_wrapper(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        key = torch.tensor([5.0, 6.0])
        cache.hashed_put(cache.hash_state(key), 99)
        assert cache.get(key) == 99


# ---------------------------------------------------------------------------
# invalidate
# ---------------------------------------------------------------------------


class TestInvalidate:
    def test_invalidate_hides_entries(self) -> None:
        cache = KeylessCache(max_size=64, num_clients=1)
        key = torch.tensor([1.0, 2.0, 3.0])
        cache.put((key, "value"))
        assert cache.get(key) == "value"
        cache.invalidate()
        assert cache.get(key) is None


# ---------------------------------------------------------------------------
# striped locking under concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_access_no_corruption(self) -> None:
        # Few stripes (16) over many slots (256) so slots share locks, and more
        # keys than slots so they collide on slots: maximizes stripe contention.
        # Each key i stores value i, so the direct-mapped invariant is that a
        # hit for key i can only ever return i - any other value is a torn read.
        cache = KeylessCache(max_size=256, num_clients=1)
        num_keys = 512
        keys = [torch.tensor([float(i)]) for i in range(num_keys)]
        num_threads = 8
        iterations = 3000
        violations: list[tuple[int, int]] = []

        def worker(offset: int) -> None:
            for step in range(iterations):
                i = (offset + step) % num_keys
                cache.put((keys[i], i))
                j = (offset + step * 7) % num_keys
                got = cache.get(keys[j])
                if got is not None and got != j:
                    violations.append((j, got))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert violations == []

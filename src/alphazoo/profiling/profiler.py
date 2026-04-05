from __future__ import annotations

import marshal
import os
import pstats
from collections import defaultdict

import yappi


class Profiler:
    """Lightweight profiling data collector and merger.

    Wraps yappi for wall-clock profiling. Converts and merges
    profiling data entirely in memory (no temp files) using the
    marshal-based pstat format.
    """

    class PStatHolder:
        """Adapter so pstats.Stats can load from an in-memory dict.

        pstats.Stats only accepts file paths or objects with a
        ``create_stats()`` method and a ``.stats`` dict attribute.
        This wrapper satisfies that protocol.
        """

        def __init__(self, stats_dict: dict) -> None:
            self.stats = stats_dict

        def create_stats(self) -> None:
            pass

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._accumulated: list[bytes] = []

    def start(self) -> None:
        yappi.clear_stats()
        yappi.set_clock_type("wall")
        # profile_threads=true bugs snakeviz async mode profile visualization
        yappi.start(profile_threads=False, builtins=True)

    def stop(self) -> bytes:
        """Stop yappi and return the current profile as marshaled pstat bytes."""
        yappi.stop()
        data = Profiler.yappi_stats_to_bytes(yappi.get_func_stats())
        yappi.clear_stats()
        return data

    def accumulate(self, data: bytes) -> None:
        self._accumulated.append(data)

    def get_accumulated(self) -> bytes:
        """Get all accumulated data as a single marshaled pstat blob."""
        return Profiler.merge(self._accumulated)

    def save_data_to_file(self, stats: pstats.Stats, filename: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        stats.dump_stats(path)
        return path

    def save_metrics_to_file(self, metrics: dict, running_mode: str | None = None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "summary.txt")

        total_run_time = metrics.get("time/total", 0.0)
        with open(path, "w") as f:
            if running_mode:
                f.write(f"Running mode: {running_mode}\n")
            f.write(f"Total run time: {total_run_time:.2f}s ({total_run_time / 60:.2f}m)\n")

        return path

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def merge(stats_list: list[bytes]) -> bytes:
        """Merge multiple marshaled pstat byte blobs into single marshaled bytes."""
        if not stats_list:
            raise ValueError("stats_list must not be empty")
        if len(stats_list) == 1:
            return stats_list[0]
        
        merged = Profiler.bytes_to_pstats(stats_list[0])
        for b in stats_list[1:]:
            merged.add(Profiler.bytes_to_pstats(b))
        return marshal.dumps(merged.stats)

    @staticmethod
    def bytes_to_pstats(data: bytes) -> pstats.Stats:
        """Deserialize marshaled pstat bytes into a :pyclass:`pstats.Stats`."""
        return pstats.Stats(Profiler.PStatHolder(marshal.loads(data)))
    
    @staticmethod
    def yappi_stats_to_bytes(func_stats: yappi.YFuncStats) -> bytes:
        """Convert yappi function stats to marshaled pstat-format bytes.

        Builds the same dict structure that :pyfunc:`pstats.Stats.load_stats`
        expects, but entirely in memory.
        """
        pdict: dict = {}
        callers: dict = defaultdict(dict)

        for fs in func_stats:
            for ct in fs.children:
                callers[ct][(fs.module, fs.lineno, fs.name)] = (
                    ct.ncall, ct.nactualcall, ct.tsub, ct.ttot,
                )

        for fs in func_stats:
            pdict[(fs.module, fs.lineno, fs.name)] = (
                fs.ncall, fs.nactualcall, fs.tsub, fs.ttot, callers[fs],
            )

        return marshal.dumps(pdict)

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
from collections import defaultdict
from typing import Optional

import ray
from ray.actor import ActorHandle

logger = logging.getLogger("alphazoo")

_THREAD_NAME_RE = re.compile(r'^Thread \d+ "(.*)"$')
_THREAD_FRAME_RE = re.compile(r'^thread \(\d+\)(.*)$')


class Profiler:
    """Spawns and manages py-spy subprocesses attached to the main process and Ray actors.

    Each target PID gets its own py-spy `record` subprocess that writes a speedscope JSON file into `output_dir`.
    Subprocesses are stopped gracefully with SIGINT so py-spy flushes the captured profile to disk.

    `finish()` stops every subprocess and then writes a per-group aggregate file
    (`{group}_all.speedscope.json`) that merges thread profiles by their Python
    thread name and sums samples across processes of the same group.
    """

    _SAMPLE_RATE_HZ = 100
    _STOP_TIMEOUT_S = 30.0
    _FAILURE_DETECT_S = 0.5

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self._processes: list[tuple[subprocess.Popen, str]] = []
        self._group_files: dict[str, list[str]] = defaultdict(list)
        os.makedirs(output_dir, exist_ok=True)

    def start_main(self) -> None:
        self._spawn("main", "main", os.getpid())

    def attach(self, group_name: str, actors: list[ActorHandle]) -> None:
        if not actors:
            return
        pids: list[int] = ray.get([actor.get_pid.remote() for actor in actors])
        is_unique = (len(pids) == 1)
        for i, pid in enumerate(pids):
            name = group_name if is_unique else f"{group_name}_{i}"
            self._spawn(group_name, name, pid)

    def finish(self) -> None:
        self._stop_all()
        self._write_aggregate_results()

    def save_metrics_to_file(self, metrics: dict) -> str:
        path = os.path.join(self.output_dir, "summary.txt")
        total_run_time = metrics.get("time/total", 0.0)
        with open(path, "w") as f:
            f.write(f"Total run time: {total_run_time:.2f}s ({total_run_time / 60:.2f}m)\n")
        return path

    def _spawn(self, group: str, name: str, pid: int) -> None:
        executable = shutil.which("py-spy")
        if executable is None:
            raise RuntimeError("py-spy not found on PATH. Install with `pip install py-spy`.")

        output_path = os.path.join(self.output_dir, f"{name}.speedscope.json")
        stderr_path = os.path.join(self.output_dir, f"{name}.py-spy.log")

        cmd = [
            executable, "record",
            "--pid", str(pid),
            "--threads",
            "--rate", str(self._SAMPLE_RATE_HZ),
            "--format", "speedscope",
            "--output", output_path,
        ]
        stderr_file = open(stderr_path, "w")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=stderr_file)
        finally:
            stderr_file.close()

        # detect errors early.
        try:
            rc = proc.wait(timeout=self._FAILURE_DETECT_S)
        except subprocess.TimeoutExpired:
            self._processes.append((proc, output_path))
            self._group_files[group].append(output_path)
            logger.info("Profiler: py-spy attached to %s with pid %d", name, pid)
            return

        with open(stderr_path) as f:
            stderr_content = f.read().strip()
        raise RuntimeError(
            f"py-spy exited immediately (code {rc}) attaching to pid {pid}.\n"
            f"stderr:\n{stderr_content}\n\n"
            "On Linux, py-spy needs ptrace permissions:\n"
            "  sudo setcap cap_sys_ptrace=eip $(readlink -f $(which py-spy))\n"
            "  echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope"
        )

    def _stop_all(self) -> None:
        for proc, _ in self._processes:
            try:
                proc.send_signal(signal.SIGINT)
            except ProcessLookupError:
                continue

        for proc, output_path in self._processes:
            try:
                proc.wait(timeout=self._STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                logger.warning("py-spy did not exit within %.0fs, terminating: %s",
                               self._STOP_TIMEOUT_S, output_path)
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            logger.info("py-spy output: %s", output_path)

        self._processes.clear()

    def _write_aggregate_results(self) -> None:
        for group, paths in self._group_files.items():
            if len(paths) <= 1:
                continue

            present = [p for p in paths if os.path.exists(p)]
            if len(present) < 2:
                logger.warning(
                    "Skipping merge for %s: only %d of %d output files present",
                    group, len(present), len(paths),
                )
                continue
            
            output_path = self._merge_into_aggregate(group, present)
            logger.info("Profiler: aggregated %d files into %s", len(present), output_path)

    def _merge_into_aggregate(self, group: str, input_paths: list[str]) -> str:
        docs = []
        for path in input_paths:
            with open(path) as f:
                docs.append(json.load(f))

        groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
        for doc in docs:
            for profile in doc["profiles"]:
                groups[self._parse_thread_name(profile["name"])].append((doc, profile))

        merged_frames: list[dict] = []
        frame_index: dict[tuple, int] = {}
        merged_profiles: list[dict] = []
        for thread_name, entries in sorted(groups.items()):
            samples: list[list[int]] = []
            weights: list[float] = []
            for doc, profile in entries:
                remap = [self._intern_frame(f, merged_frames, frame_index) for f in doc["shared"]["frames"]]
                for stack in profile["samples"]:
                    samples.append([remap[i] for i in stack])
                weights.extend(profile["weights"])

            display_name = thread_name if thread_name else "(unnamed)"
            display_name = f"{display_name} (aggregated ×{len(entries)})"

            merged_profiles.append({
                "type": "sampled",
                "name": display_name,
                "unit": "seconds",
                "startValue": 0.0,
                "endValue": sum(weights),
                "samples": samples,
                "weights": weights,
            })

        output = {
            "$schema": "https://www.speedscope.app/file-format-schema.json",
            "shared": {"frames": merged_frames},
            "profiles": merged_profiles,
            "exporter": "alphazoo py-spy merger",
            "name": f"{group} (aggregated)",
            "activeProfileIndex": 0,
        }

        output_path = os.path.join(self.output_dir, f"{group}_all.speedscope.json")
        with open(output_path, "w") as f:
            json.dump(output, f)
        return output_path

    def _intern_frame(self, frame: dict, frames: list[dict], index: dict[tuple, int]) -> int:
        frame = self._normalize_thread_frame(frame)
        key = (frame.get("name"), frame.get("file"), frame.get("line"), frame.get("col"))
        idx = index.get(key)
        if idx is None:
            idx = len(frames)
            index[key] = idx
            frames.append(frame)
        return idx

    def _normalize_thread_frame(self, frame: dict) -> dict:
        match = _THREAD_FRAME_RE.match(frame.get("name", ""))
        if match is None:
            return frame
        return {**frame, "name": "thread" + match.group(1)}

    def _parse_thread_name(self, profile_name: str) -> str:
        match = _THREAD_NAME_RE.match(profile_name)
        return match.group(1) if match else profile_name

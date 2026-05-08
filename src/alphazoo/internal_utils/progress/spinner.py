from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger("alphazoo")


class Spinner:
    """
    Indeterminate progress indicator for waits of unknown length.

    Auto-detects whether stdout is a TTY. TTY renders an animated frame
    redrawn in place. Non-TTY emits a heartbeat log line every
    `heartbeat_sec` so piped/captured runs still show liveness.

    When `max_duration` is provided the spinner expresses "patience": the
    frame color shifts green → yellow → red and the tick interval slows
    down as `elapsed / max_duration` grows. With `max_duration=None`
    the spinner stays in its neutral cadence and color (current default).

    Driven by a background thread; no external updates needed.
    Only enabled when `alphazoo` is in verbose mode (logger at INFO level).
    """

    _FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _DONE_FRAME = "⠿"
    _BASE_FRAME_INTERVAL = 0.1
    _MAX_FRAME_INTERVAL = 0.35
    _ANSI_RESET = "\033[0m"

    def __init__(
        self,
        description: str,
        *,
        max_duration: Optional[float] = None,
        heartbeat_sec: float = 30.0,
    ) -> None:
        self.description = description
        self._max_duration = max_duration
        self._heartbeat_sec = heartbeat_sec
        self._is_tty = sys.stdout.isatty()
        self._use_color = self._is_tty and os.environ.get("NO_COLOR") is None
        self._enabled = logger.isEnabledFor(logging.INFO)
        self._start_time: Optional[float] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_idx = 0

    def __enter__(self) -> "Spinner":
        if not self._enabled:
            return self
        self._start_time = time.time()
        if self._is_tty:
            self._tty_render()
        else:
            logger.info(f"{self.description}: starting")
        self._thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if not self._enabled:
            return
        if self._is_tty:
            color = self._color_code()
            reset = self._ANSI_RESET if color else ""
            line = f"{color}{self._DONE_FRAME} {self.description}{reset}"
            sys.stdout.write("\r" + line + "   \n")
            sys.stdout.flush()
        else:
            elapsed = time.time() - (self._start_time or time.time())
            logger.info(f"{self.description}: done in {elapsed:.1f}s")

    def _tick_loop(self) -> None:
        last_heartbeat = self._start_time or time.time()
        while not self._stop_event.is_set():
            if self._is_tty:
                self._frame_idx = (self._frame_idx + 1) % len(self._FRAMES)
                self._tty_render()
                self._stop_event.wait(self._current_interval())
            else:
                now = time.time()
                if now - last_heartbeat >= self._heartbeat_sec:
                    elapsed = now - (self._start_time or now)
                    logger.info(f"{self.description}: still waiting ({elapsed:.0f}s)")
                    last_heartbeat = now
                self._stop_event.wait(min(1.0, self._heartbeat_sec))

    def _tty_render(self) -> None:
        frame = self._FRAMES[self._frame_idx]
        color = self._color_code()
        reset = self._ANSI_RESET if color else ""
        line = f"{color}{frame} {self.description}{reset}"
        sys.stdout.write("\r" + line + "   ")
        sys.stdout.flush()

    def _patience_t(self) -> float:
        if self._max_duration is None or self._max_duration <= 0 or self._start_time is None:
            return 0.0
        elapsed = time.time() - self._start_time
        return min(1.0, max(0.0, elapsed / self._max_duration))

    def _current_interval(self) -> float:
        if self._max_duration is None:
            return self._BASE_FRAME_INTERVAL
        t = self._patience_t()
        return self._BASE_FRAME_INTERVAL + t * (self._MAX_FRAME_INTERVAL - self._BASE_FRAME_INTERVAL)

    def _color_code(self) -> str:
        if not self._use_color or self._max_duration is None:
            return ""
        t = self._patience_t()
        # Two-segment lerp through yellow: green (50,200,80) → yellow (220,200,30) → red (220,50,50)
        if t < 0.5:
            u = t * 2
            r = int(50 + u * (220 - 50))
            g = 200
            b = int(80 + u * (30 - 80))
        else:
            u = (t - 0.5) * 2
            r = 220
            g = int(200 + u * (50 - 200))
            b = int(30 + u * (50 - 30))
        return f"\033[38;2;{r};{g};{b}m"

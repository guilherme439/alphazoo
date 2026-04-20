from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("alphazoo")


class Progress:
    """
    Progress bar indicator.

    Auto-detects whether stdout is a TTY. TTY renders a live `\r`-redrawn bar.
    Non-TTY (piped, redirected, pytest capture) emits at most one log line per
    crossed milestone, plus a heartbeat line if no milestone is crossed within
    `heartbeat_sec`.
    
    Only enabled when `alphazoo` is in verbo se mode (logger at INFO level).

    Two driving modes:
    - Active (default): caller drives updates via `update(value)`.
    - Passive: provide `poll_fn` + `poll_interval`. A background thread in the
      main process polls `poll_fn()` on that interval and issues an update when
      the returned value changes.
    """

    _BAR_WIDTH = 30

    def __init__(
        self,
        description: str,
        total: int,
        *,
        milestone_pct: int = 20,
        heartbeat_sec: float = 30.0,
        poll_fn: Optional[Callable[[], int]] = None,
        poll_interval: float = 2.0,
    ) -> None:
        self.description = description
        self.total = max(total, 0)
        self.current = 0
        self._milestone_pct = milestone_pct
        self._heartbeat_sec = heartbeat_sec
        self._poll_fn = poll_fn
        self._poll_interval = poll_interval
        self._is_tty = sys.stdout.isatty()
        self._enabled = logger.isEnabledFor(logging.INFO)
        self._start_time: Optional[float] = None
        self._last_log_time: float = 0.0
        self._last_logged_milestone: int = -1
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Progress":
        if not self._enabled:
            return self
        self._start_time = time.time()
        self._last_log_time = self._start_time
        if self._is_tty:
            self._tty_render()
        else:
            logger.info(f"{self.description}: starting (target {self.total})")
        if self._poll_fn is not None:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
        if not self._enabled:
            return
        elapsed = time.time() - (self._start_time or time.time())
        if self._is_tty:
            self.current = self.total
            self._tty_render()
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            logger.info(f"{self.description}: done in {elapsed:.1f}s")

    def update(self, current: int) -> None:
        if not self._enabled:
            return
        with self._lock:
            self.current = max(0, min(current, self.total))
            if self._is_tty:
                self._tty_render()
                return
            if self.total == 0:
                return
            pct = int(100 * self.current / self.total)
            milestone = (pct // self._milestone_pct) * self._milestone_pct
            now = time.time()
            crossed_milestone = milestone > self._last_logged_milestone
            heartbeat_due = (now - self._last_log_time) >= self._heartbeat_sec
            if crossed_milestone or heartbeat_due:
                self._last_logged_milestone = milestone
                self._last_log_time = now
                logger.info(f"{self.description}: {pct}%")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                value = self._poll_fn()
            except Exception:
                value = None
            if value is not None and value != self.current:
                self.update(value)
            self._stop_event.wait(self._poll_interval)

    def _tty_render(self) -> None:
        pct = int(100 * self.current / self.total) if self.total else 0
        fill = int(self._BAR_WIDTH * pct / 100)
        bar = "█" * fill + "░" * (self._BAR_WIDTH - fill)
        line = f"{self.description}: [{bar}] {pct}%"
        # trailing spaces clear any residue from a longer previous render
        sys.stdout.write("\r" + line + "   ")
        sys.stdout.flush()

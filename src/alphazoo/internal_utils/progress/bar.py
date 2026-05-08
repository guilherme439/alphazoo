from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger("alphazoo")


class Bar:
    """
    Progress bar indicator.

    Auto-detects whether stdout is a TTY. TTY renders a live `\r`-redrawn bar.
    Non-TTY (piped, redirected, pytest capture) emits at most one log line per
    crossed milestone, plus a heartbeat line if no milestone is crossed within
    `heartbeat_sec`.

    Only enabled when `alphazoo` is in verbose mode (logger at INFO level).

    Caller drives updates via `tick()` (advance by one) or `update(value)`
    (set the absolute value).
    """

    _BAR_WIDTH = 30

    def __init__(
        self,
        description: str,
        total: int,
        *,
        milestone_pct: int = 20,
        heartbeat_sec: float = 30.0,
    ) -> None:
        self.description = description
        self.total = max(total, 0)
        self.current = 0
        self._milestone_pct = milestone_pct
        self._heartbeat_sec = heartbeat_sec
        self._is_tty = sys.stdout.isatty()
        self._enabled = logger.isEnabledFor(logging.INFO)
        self._start_time: Optional[float] = None
        self._last_log_time: float = 0.0
        self._last_logged_milestone: int = -1
        self._lock = threading.Lock()

    def __enter__(self) -> "Bar":
        if not self._enabled:
            return self
        self._start_time = time.time()
        self._last_log_time = self._start_time
        if self._is_tty:
            self._tty_render()
        else:
            logger.info(f"{self.description}: starting (target {self.total})")
        return self

    def __exit__(self, *exc) -> None:
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

    def tick(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            self.current = max(0, min(self.current + 1, self.total))
            self._render_or_log()

    def update(self, current: int) -> None:
        if not self._enabled:
            return
        with self._lock:
            self.current = max(0, min(current, self.total))
            self._render_or_log()

    def _render_or_log(self) -> None:
        """Caller must hold self._lock."""
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

    def _tty_render(self) -> None:
        pct = int(100 * self.current / self.total) if self.total else 0
        fill = int(self._BAR_WIDTH * pct / 100)
        bar = "█" * fill + "░" * (self._BAR_WIDTH - fill)
        line = f"{self.description}: [{bar}] {pct}%"
        # trailing spaces clear any residue from a longer previous render
        sys.stdout.write("\r" + line + "   ")
        sys.stdout.flush()

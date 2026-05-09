from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import Any, Optional, TextIO

logger = logging.getLogger("alphazoo")


class _StdoutGuard:
    """Wraps a TextIO stream; reports writes that didn't originate from the spinner."""

    def __init__(self, spinner: "Spinner", wrapped: TextIO) -> None:
        self._spinner = spinner
        self._wrapped = wrapped

    def write(self, s: str) -> int:
        if self._spinner._is_internal_write():
            return self._wrapped.write(s)
        return self._spinner._handle_external_write(self._wrapped, s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)


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
    _QUIET_PERIOD = 2.0

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
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._line_dirty = False
        self._external_last_time: Optional[float] = None
        self._external_last_char = ""
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None

    def __enter__(self) -> "Spinner":
        if not self._enabled:
            return self
        self._start_time = time.time()
        if self._is_tty:
            self._install_guards()
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
            with self._lock:
                interrupted = self._external_last_time is not None
                ext_last_char = self._external_last_char
            self._restore_guards()
            if interrupted and ext_last_char != "\n":
                sys.stdout.write("\n")
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
                if self._should_resume_after_interruption():
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
        with self._lock:
            self._do_internal_write("\r" + line + "   ")
            sys.stdout.flush()
            self._line_dirty = True

    def _should_resume_after_interruption(self) -> bool:
        with self._lock:
            ext_time = self._external_last_time
            ext_char = self._external_last_char
        if ext_time is None:
            return True
        if time.time() - ext_time < self._QUIET_PERIOD:
            return False
        if ext_char != "\n":
            self._do_internal_write("\n")
        with self._lock:
            self._external_last_time = None
            self._external_last_char = ""
        return True

    def _install_guards(self) -> None:
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _StdoutGuard(self, sys.stdout)
        sys.stderr = _StdoutGuard(self, sys.stderr)

    def _restore_guards(self) -> None:
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr is not None:
            sys.stderr = self._original_stderr
            self._original_stderr = None

    def _do_internal_write(self, s: str) -> None:
        self._tls.internal = True
        try:
            sys.stdout.write(s)
        finally:
            self._tls.internal = False

    def _is_internal_write(self) -> bool:
        return getattr(self._tls, "internal", False)

    def _handle_external_write(self, wrapped: TextIO, s: str) -> int:
        with self._lock:
            if self._line_dirty:
                wrapped.write("\r" + " " * self._clear_width() + "\r")
                self._line_dirty = False
            written = wrapped.write(s)
            if s:
                self._external_last_time = time.time()
                self._external_last_char = s[-1]
            return written

    def _clear_width(self) -> int:
        return len(self.description) + 8

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

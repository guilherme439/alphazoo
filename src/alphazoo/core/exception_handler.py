import logging
from types import TracebackType
from typing import Callable, Optional

logger = logging.getLogger("alphazoo")


class ExceptionHandler:

    def __init__(self, on_exit: Callable[[], None]) -> None:
        self._on_exit = on_exit

    def __enter__(self) -> ExceptionHandler:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if exc_type is not None:
            if issubclass(exc_type, KeyboardInterrupt):
                logger.info("\nInterrupted by user — shutting down gracefully..."
                            "\nPress Ctrl+C again to force immediate shutdown (can cause memory leaks)\n")
            else:
                logger.error("\nException found — shutting down gracefully...", exc_info=(exc_type, exc, tb))
        else:
            logger.info("\nShutting down gracefully\n")

        try:
            self._on_exit()
        except Exception:
            logger.exception("Error during shutdown")

        # Swallow ordinary errors.
        # Let KeyboardInterrupt (and any other BaseException) 
        # propagate so callers up the stack see it.
        return exc_type is not None and issubclass(exc_type, Exception)

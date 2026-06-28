import functools
from typing import Any, Callable


class ConcurrencyUtils:

    @staticmethod
    def synchronized(method: Callable) -> Callable:
        """
        Run a method while holding the instance's ``self._lock``.

        The decorated class must expose a ``threading.Lock`` (or compatible context manager) as ``self._lock``.
        """
        @functools.wraps(method)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return method(self, *args, **kwargs)
        return wrapper

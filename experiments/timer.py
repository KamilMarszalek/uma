import gc
import time
from collections.abc import Callable
from typing import Any


class Timer:
    def __init__(self, func: Callable[[Any], Any], gc_disable: bool = False):
        self.func = func
        self.gc_disable = gc_disable
        self.elapsed = None
        self._gc_old = None

    def run(self, data: Any) -> Any:
        self._gc_old = gc.isenabled()
        if self.gc_disable:
            gc.disable()
        start = time.perf_counter()
        result = self.func(data)
        stop = time.perf_counter()
        if self.gc_disable and self._gc_old:
            gc.enable()
        self.elapsed = stop - start
        return result

    def get_elapsed(self) -> float:
        if self.elapsed is None:
            raise ValueError("Function has not been run yet")
        return self.elapsed

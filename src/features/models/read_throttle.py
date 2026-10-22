import threading
import time
from typing import Callable, Optional


class ReadThrottle:
    def __init__(
        self,
        bytes_per_second: Optional[int] = None,
        pauses_for_generation: bool = False,
        evict_page_cache: bool = False,
        busy: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_seconds: float = 0.5,
    ):
        self._busy = busy
        self._clock = clock
        self._sleep = sleep
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._bytes_per_second = bytes_per_second
        self._pauses_for_generation = pauses_for_generation
        self._evict_page_cache = evict_page_cache
        self._next_slot = 0.0

    def configure(
        self, bytes_per_second: Optional[int], pauses_for_generation: bool, evict_page_cache: bool
    ) -> None:
        with self._lock:
            self._bytes_per_second = bytes_per_second
            self._pauses_for_generation = pauses_for_generation
            self._evict_page_cache = evict_page_cache

    @property
    def evict_page_cache(self) -> bool:
        return self._evict_page_cache

    def generation_running(self) -> bool:
        return self._busy is not None and bool(self._busy())

    def holding(self) -> bool:
        return self._pauses_for_generation and self.generation_running()

    def _hold_for_generation(self, should_stop: Optional[Callable[[], bool]]) -> bool:
        while self.holding():
            if should_stop is not None and should_stop():
                return False
            self._sleep(self._poll_seconds)
        return True

    def _delay_for(self, nbytes: int) -> float:
        with self._lock:
            rate = self._bytes_per_second
            if not rate or nbytes <= 0:
                return 0.0
            now = self._clock()
            start = max(self._next_slot, now)
            self._next_slot = start + nbytes / rate
            return start - now

    def wait(self, nbytes: int, should_stop: Optional[Callable[[], bool]] = None) -> bool:
        if not self._hold_for_generation(should_stop):
            return False
        remaining = self._delay_for(nbytes)
        while remaining > 0:
            if should_stop is not None and should_stop():
                return False
            step = min(remaining, self._poll_seconds)
            self._sleep(step)
            remaining -= step
        return True

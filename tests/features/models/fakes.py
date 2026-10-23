import threading
import time
from unittest.mock import AsyncMock, MagicMock

from src.features.models.indexing_coordinator import ModelIndexingCoordinator
from src.features.models.indexing_governor import IndexingGovernor

MAX_FAKE_SLEEPS = 1000


class FakeClock:
    def __init__(self, wall=1000.0):
        self.now = 0.0
        self.wall = wall
        self.sleeps = []
        self.on_sleep = None

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        assert len(self.sleeps) < MAX_FAKE_SLEEPS, "a test slept in a loop that never ends"
        if self.on_sleep is not None:
            self.on_sleep()


class OpenPlugins:
    def execute_hook(self, hook, initial_data):
        return MagicMock(data={}), False


class ProgressScanner:
    digests = None

    def __init__(self, result=None, ticks=((1, 2, "first"), (2, 2, "second"))):
        self.result = result or {"indexed": 2, "skipped": 1, "failed": 0, "total": 3}
        self.ticks = ticks
        self.callback = None
        self.calls = 0
        self.governor = IndexingGovernor()

    def set_progress_callback(self, callback):
        self.callback = callback

    def index_models(self, cancel_check=None):
        self.calls += 1
        if self.callback is not None:
            for tick in self.ticks:
                self.callback(*tick)
        return dict(self.result)


class FakeIndexingCoordinator:
    def __init__(self):
        self.calls = []
        self.reprojected = []

    def rescan(self, trigger="location_change"):
        self.calls.append(trigger)
        return {"state": "idle"}

    def reproject(self, trigger="roots_change"):
        self.reprojected.append(trigger)
        return {"state": "idle"}

    def status(self):
        return {"state": "idle"}


class ThreadSpawner:
    def __init__(self):
        self.threads = []

    def __call__(self, target):
        thread = threading.Thread(target=target, daemon=True)
        self.threads.append(thread)
        thread.start()

    def join(self):
        for thread in list(self.threads):
            thread.join(timeout=5)
            assert not thread.is_alive()


class CountingScanner:
    digests = None

    def __init__(self, hold_first_run=False):
        self.calls = 0
        self.first_run_entered = threading.Event()
        self.release_first_run = threading.Event()
        self.first_run_was_cancelled = False
        self._hold_first_run = hold_first_run
        self._lock = threading.Lock()
        self.resolver = MagicMock()
        self.resolver.online_root_ids.return_value = []
        self.resolver.roots.return_value = []

    def set_progress_callback(self, callback):
        return None

    def index_models(self, max_workers=4, cancel_check=None):
        with self._lock:
            self.calls += 1
            first = self.calls == 1
        if first and self._hold_first_run:
            self.first_run_entered.set()
            deadline = time.monotonic() + 20
            while not self.release_first_run.wait(timeout=0.005):
                if time.monotonic() > deadline:
                    return {"indexed": 0, "cancelled": False}
                if cancel_check is not None and cancel_check():
                    self.first_run_was_cancelled = True
                    return {"indexed": 0, "cancelled": True}
        return {"indexed": 0, "cancelled": False}


def build_coordinator(scanner, spawner):
    reconciler = MagicMock()
    reconciler.reconcile = AsyncMock()
    return ModelIndexingCoordinator(
        model_repository=MagicMock(),
        plugin_registry=OpenPlugins(),
        scanner=scanner,
        native_availability_projector=reconciler,
        spawn=spawner,
    )


def wait_until(condition, timeout=5.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(interval)


def drain_manager(manager, max_jobs=None):
    done = 0
    while max_jobs is None or done < max_jobs:
        job = manager._claim_next(None)
        if job is None:
            break
        manager._run_job(job, None)
        done += 1
    manager._flush_applied(force=True)
    return done


def wait_for_idle(manager, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    with manager._cv:
        while manager._inflight or manager._has_runnable_locked():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            if manager.paused:
                return False
            manager._cv.wait(min(0.05, remaining) if remaining is not None else 0.05)
    manager._flush_applied(force=True)
    return True

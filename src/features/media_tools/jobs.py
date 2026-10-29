import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from src.features.media_tools.dto import TERMINAL_STATUSES, MediaToolItem, RunDTO, RunError
from src.platform.http.base_controller import PassThrough
from src.platform.http.errors import DomainError
from src.platform.media_tools import MediaToolError, ToolCancelled

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 2
DEFAULT_RETENTION_SECONDS = 600.0
CANCEL_WAIT_SECONDS = 5.0
GENERIC_FAILURE = "The tool could not finish"


@dataclass
class RunOutcome:
    items: List[MediaToolItem] = field(default_factory=list)
    replaced: bool = False


@dataclass
class _Run:
    run_id: str
    user_id: str
    work: Callable[["RunControl"], RunOutcome]
    status: str = "queued"
    progress: float = 0.0
    label: str = ""
    items: List[MediaToolItem] = field(default_factory=list)
    replaced: bool = False
    error: Optional[RunError] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    finished_at: Optional[float] = None


class RunControl:
    def __init__(self, manager: "MediaToolJobManager", run: _Run):
        self._manager = manager
        self._run = run

    def report(self, fraction: float, label: str = "") -> None:
        self._manager._report(self._run, fraction, label)

    def is_cancelled(self) -> bool:
        return self._run.cancel_event.is_set()


class MediaToolJobManager:
    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        retention_seconds: float = DEFAULT_RETENTION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        cancel_wait_seconds: float = CANCEL_WAIT_SECONDS,
    ):
        self._max = max(1, int(max_concurrent))
        self._retention = retention_seconds
        self._clock = clock
        self._cancel_wait = cancel_wait_seconds
        self._lock = threading.RLock()
        self._runs: Dict[str, _Run] = {}
        self._queue: Deque[str] = deque()
        self._running = 0

    def submit(self, user_id: str, work: Callable[[RunControl], RunOutcome]) -> str:
        run = _Run(run_id=uuid.uuid4().hex, user_id=user_id, work=work)
        with self._lock:
            self._purge()
            self._runs[run.run_id] = run
            self._queue.append(run.run_id)
        self._pump()
        return run.run_id

    def get(self, user_id: str, run_id: str) -> Optional[RunDTO]:
        with self._lock:
            self._purge()
            run = self._owned(user_id, run_id)
            return self._snapshot(run) if run else None

    def cancel(self, user_id: str, run_id: str) -> Optional[RunDTO]:
        with self._lock:
            self._purge()
            run = self._owned(user_id, run_id)
            if run is None:
                return None
            if run.status in TERMINAL_STATUSES:
                return self._snapshot(run)
            run.cancel_event.set()
            if run.status == "queued":
                run.status = "cancelled"
                run.finished_at = self._clock()
                run.done.set()
                return self._snapshot(run)
        run.done.wait(self._cancel_wait)
        with self._lock:
            return self._snapshot(run)

    def wait(self, user_id: str, run_id: str, timeout: Optional[float]) -> Optional[RunDTO]:
        with self._lock:
            run = self._owned(user_id, run_id)
        if run is None:
            return None
        run.done.wait(timeout)
        with self._lock:
            return self._snapshot(run)

    def counts(self) -> Dict[str, int]:
        with self._lock:
            tally = {"queued": 0, "running": 0, "finished": 0}
            for run in self._runs.values():
                if run.status == "queued":
                    tally["queued"] += 1
                elif run.status == "running":
                    tally["running"] += 1
                else:
                    tally["finished"] += 1
            return tally

    def join_all(self, timeout: float = CANCEL_WAIT_SECONDS) -> None:
        with self._lock:
            runs = list(self._runs.values())
        for run in runs:
            if run.status not in TERMINAL_STATUSES:
                run.cancel_event.set()
        for run in runs:
            if run.status == "running":
                run.done.wait(timeout)

    def _owned(self, user_id: str, run_id: str) -> Optional[_Run]:
        run = self._runs.get(run_id)
        return run if run is not None and run.user_id == user_id else None

    def _purge(self) -> None:
        now = self._clock()
        expired = [
            run_id
            for run_id, run in self._runs.items()
            if run.status in TERMINAL_STATUSES
            and run.finished_at is not None
            and now - run.finished_at >= self._retention
        ]
        for run_id in expired:
            del self._runs[run_id]

    def _snapshot(self, run: _Run) -> RunDTO:
        return RunDTO(
            run_id=run.run_id,
            status=run.status,
            progress=run.progress,
            label=run.label,
            items=list(run.items) if run.status == "completed" else [],
            replaced=run.replaced if run.status == "completed" else False,
            error=run.error,
        )

    def _report(self, run: _Run, fraction: float, label: str) -> None:
        with self._lock:
            if run.status != "running":
                return
            run.progress = max(run.progress, min(1.0, max(0.0, float(fraction))))
            if label:
                run.label = label

    def _pump(self) -> None:
        starters: List[_Run] = []
        with self._lock:
            while self._running < self._max and self._queue:
                run = self._runs.get(self._queue.popleft())
                if run is None or run.status != "queued":
                    continue
                run.status = "running"
                self._running += 1
                starters.append(run)
        for run in starters:
            threading.Thread(
                target=self._execute, args=(run,), name=f"media-tool-{run.run_id[:8]}", daemon=True
            ).start()

    def _execute(self, run: _Run) -> None:
        control = RunControl(self, run)
        outcome: Optional[RunOutcome] = None
        error: Optional[RunError] = None
        status = "completed"
        try:
            if run.cancel_event.is_set():
                raise ToolCancelled()
            outcome = run.work(control)
        except ToolCancelled:
            status = "cancelled"
        except MediaToolError as exc:
            status, error = "failed", RunError(code=exc.code, message=exc.message)
        except DomainError as exc:
            status, error = "failed", RunError(code=exc.code, message=exc.message)
        except PassThrough as exc:
            status = "failed"
            error = RunError(code=str(getattr(exc, "error", "refused")), message=str(exc))
        except Exception:
            logger.exception("Media tool run %s failed", run.run_id)
            status, error = "failed", RunError(code="tool_failed", message=GENERIC_FAILURE)
        self._finish(run, status, outcome, error)

    def _finish(
        self, run: _Run, status: str, outcome: Optional[RunOutcome], error: Optional[RunError]
    ) -> None:
        with self._lock:
            run.status = status
            run.error = error
            if status == "completed" and outcome is not None:
                run.items = list(outcome.items)
                run.replaced = outcome.replaced
                run.progress = 1.0
            run.finished_at = self._clock()
            self._running -= 1
            run.done.set()
        self._pump()

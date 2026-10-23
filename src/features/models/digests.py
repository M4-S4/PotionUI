import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from src.features.models.digest_queue_repository import (
    MAX_ATTEMPTS,
    PRIORITY_BACKGROUND,
    PRIORITY_CHOSEN,
    PRIORITY_DEMAND,
    PRIORITY_TWIN,
    DigestQueueRepository,
)
from src.features.models.file_digest import (
    HEAD_BYTES,
    INLINE_HASH_LIMIT,
    TAIL_BYTES,
    WINDOW_BYTES,
    WINDOW_COUNT,
    DigestCancelled,
    FileDigest,
    digest_file,
    quick_signature,
)
from src.features.models.hash_cache_repository import model_hash_cache_repo
from src.features.models.indexing_governor import PROFILES, IndexingGovernor, IndexingProfile
from src.features.models.locations_repository import (
    DIGEST_DUPLICATE,
    DIGEST_MERGED,
    DIGEST_STALE,
    ModelLocationsRepository,
)
from src.features.models.read_throttle import ReadThrottle
from src.features.models.repository import model_repo
from src.platform.database.rows import now_iso
from src.platform.runtime.thread_priority import set_current_thread_priority
from src.platform.settings.repository import SettingRepository

logger = logging.getLogger(__name__)

PAUSED_SETTING_KEY = "model_indexing_paused"
POLICY_SETTING_KEY = "model_digest_policy"
POLICY_UNSET = "unset"
POLICY_BACKGROUND = "background"
POLICY_ON_DEMAND = "on_demand"
POLICIES = (POLICY_UNSET, POLICY_BACKGROUND, POLICY_ON_DEMAND)
PLAN_MIN_FILES = 20
PLAN_MIN_BYTES = 64 * 1024 * 1024 * 1024

CANDIDATE_CHANGED_MESSAGE = "The file changed after it was found"
RETRY_DELAYS_SECONDS = (60.0, 600.0)
OFFLINE_RETRY_SECONDS = 60.0
APPLIED_FLUSH_SECONDS = 5.0
DEMAND_WAIT_SECONDS = 1800.0
DEMAND_RATE_GUESS_BYTES_PER_SECOND = 400 * 1024 * 1024
THROUGHPUT_WINDOW_SECONDS = 30.0
THROUGHPUT_WARMUP_SECONDS = 10.0
STATUS_FAILED_LIMIT = 50
MAX_INLINE_CANDIDATES = 8
MAX_BACKGROUND_CANDIDATES = 8
MAX_REMEMBERED_MERGES = 512
HASH_CANDIDATES_INLINE = "inline"
HASH_CANDIDATES_BACKGROUND = "background"
HASH_CANDIDATES_NEVER = "never"
HASH_CANDIDATES_PACED = "paced"
HASH_CANDIDATE_MODES = (
    HASH_CANDIDATES_INLINE, HASH_CANDIDATES_BACKGROUND, HASH_CANDIDATES_NEVER, HASH_CANDIDATES_PACED,
)
MAX_PACED_CANDIDATES = 2
PACED_WAIT_SECONDS = 900.0
PACED_POLL_SECONDS = 0.25
SIGNATURE_READ_BYTES = HEAD_BYTES + TAIL_BYTES + WINDOW_COUNT * WINDOW_BYTES


class ThroughputMeter:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window: float = THROUGHPUT_WINDOW_SECONDS,
        warmup: float = THROUGHPUT_WARMUP_SECONDS,
    ):
        self._clock = clock
        self._window = window
        self._warmup = warmup
        self._samples: Deque = deque()
        self._first: Optional[float] = None
        self._lock = threading.Lock()

    def add(self, nbytes: int) -> None:
        if nbytes <= 0:
            return
        with self._lock:
            now = self._clock()
            self._prune(now)
            if self._first is None:
                self._first = now
            self._samples.append((now, nbytes))

    def _prune(self, now: float) -> None:
        while self._samples and now - self._samples[0][0] > self._window:
            self._samples.popleft()
        if not self._samples:
            self._first = None

    def rate(self) -> Optional[float]:
        with self._lock:
            now = self._clock()
            self._prune(now)
            if self._first is None or now - self._first < self._warmup:
                return None
            span = min(self._window, now - self._first)
            return sum(nbytes for _, nbytes in self._samples) / span if span > 0 else None


@dataclass(frozen=True)
class DigestMatch:
    model: Optional[Any] = None
    candidates: int = 0
    waiting: int = 0


@dataclass
class DigestCandidate:
    key: str
    path: str
    size: int
    mtime_ns: Optional[int]
    priority: int = PRIORITY_TWIN
    attempts: int = 0
    payload: Any = None
    requires: Tuple[str, ...] = ()
    result: Optional[FileDigest] = None


@dataclass
class _Job:
    key: str
    kind: str
    path: Optional[Path]
    size: int
    mtime_ns: Optional[int]
    priority: int
    filename: str
    model_id: Optional[str] = None
    row: Optional[Dict[str, Any]] = None
    candidate: Optional[DigestCandidate] = None
    done: int = 0
    demanded: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Worker:
    stop: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    job: Optional[_Job] = None


class _ManagerPacing:
    def __init__(self, manager: "DigestManager", throttle: ReadThrottle, job: "_Job"):
        self._manager = manager
        self._throttle = throttle
        self._job = job

    @property
    def evict_page_cache(self) -> bool:
        return self._throttle.evict_page_cache

    def holding(self) -> bool:
        if self._job.demanded:
            return False
        return self._manager.paused or self._throttle.holding()

    def wait(self, nbytes: int, should_stop: Optional[Callable[[], bool]] = None) -> bool:
        job = self._job
        stopped = should_stop if should_stop is not None else (lambda: False)
        while self._manager.paused and not job.demanded:
            if stopped():
                return False
            self._manager._sleep(self._manager._pause_poll_seconds)
        if job.demanded:
            return not stopped()
        if self._throttle.wait(nbytes, lambda: job.demanded or stopped()):
            return True
        return job.demanded and not stopped()


class DigestManager:
    def __init__(
        self,
        queue: DigestQueueRepository,
        governor: IndexingGovernor,
        path_for: Callable[[Dict[str, Any]], Optional[Path]],
        locations_write_lock: Optional[threading.Lock] = None,
        on_applied: Optional[Callable[[Set[str]], None]] = None,
        setting_repository: Optional[SettingRepository] = None,
        models=None,
        locations: Optional[ModelLocationsRepository] = None,
        hash_cache=None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        idle_poll_seconds: float = 1.0,
        pause_poll_seconds: float = 0.2,
    ):
        self.queue = queue
        self.governor = governor
        self._path_for = path_for
        self._write_lock = locations_write_lock or threading.Lock()
        self._on_applied = on_applied
        self._settings = setting_repository
        self._models = models or model_repo
        self._locations = locations or ModelLocationsRepository()
        self._hash_cache = hash_cache or model_hash_cache_repo
        self._clock = clock
        self._monotonic = monotonic
        self._sleep = sleep
        self._idle_poll_seconds = idle_poll_seconds
        self._pause_poll_seconds = pause_poll_seconds
        self._meter = ThroughputMeter(monotonic)
        self._cv = threading.Condition()
        self._inflight: Dict[str, _Job] = {}
        self._workers: List[_Worker] = []
        self._pool_signature = None
        self._running = False
        self._closing = False
        self._candidates: Dict[str, DigestCandidate] = {}
        self._candidate_done: Dict[str, Callable[[DigestCandidate, Optional[FileDigest], Optional[str]], None]] = {}
        self._awaiting: Dict[str, DigestCandidate] = {}
        self._flusher: Optional[threading.Thread] = None
        self._applied: Set[str] = set()
        self._demand_threads: Dict[str, threading.Thread] = {}
        self._unflushed_since: Optional[float] = None
        self._merged: Dict[str, str] = {}
        self._version = 0
        self._paused = self._read_paused()
        self._policy = POLICY_UNSET
        self._backfill_lock = threading.Lock()
        self._unsignable: Set[str] = set()

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def version(self) -> int:
        return self._version

    def _read_paused(self) -> bool:
        if self._settings is None:
            return False
        try:
            setting = self._settings.get_setting_by_key(PAUSED_SETTING_KEY)
            return bool(setting.get_typed_value()) if setting is not None else False
        except Exception:
            logger.warning("Could not read the model indexing pause flag", exc_info=True)
            return False

    def _write_paused(self, paused: bool) -> None:
        if self._settings is None:
            return
        if not self._settings.update_setting_value_by_key(PAUSED_SETTING_KEY, "true" if paused else "false"):
            raise RuntimeError(f"The '{PAUSED_SETTING_KEY}' setting row is missing")

    @property
    def policy(self) -> str:
        if self._settings is None:
            return self._policy
        try:
            setting = self._settings.get_setting_by_key(POLICY_SETTING_KEY)
            stored = setting.get_typed_value() if setting is not None else None
        except Exception:
            logger.warning("Could not read the model digest policy", exc_info=True)
            stored = None
        return stored if stored in POLICIES else POLICY_UNSET

    def _store_policy(self, policy: str) -> None:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {', '.join(POLICIES)}")
        if self._settings is None:
            self._policy = policy
        elif not self._settings.update_setting_value_by_key(POLICY_SETTING_KEY, policy):
            raise RuntimeError(f"The '{POLICY_SETTING_KEY}' setting row is missing")

    def set_policy(self, policy: str) -> int:
        self._store_policy(policy)
        queued = 0
        if policy == POLICY_BACKGROUND:
            queued = self.enqueue_pending("policy")
        elif policy == POLICY_ON_DEMAND:
            self.queue.clear_background()
        self._wake()
        return queued

    def plan(self) -> Dict[str, Any]:
        return self._plan(self.queue.queue_stats(), self.queue.digest_stats())

    def _plan(self, stats: Dict[str, int], digests: Dict[str, int]) -> Dict[str, Any]:
        files = max(0, digests["pending"] - stats["queued"] - stats["failed"])
        size = max(0, digests["pending_bytes"] - stats["queued_bytes"] - stats["failed_bytes"])
        return {
            "awaiting_choice": self.policy == POLICY_UNSET and (files > PLAN_MIN_FILES or size > PLAN_MIN_BYTES),
            "files": files,
            "bytes": size,
            "min_files": PLAN_MIN_FILES,
            "min_bytes": PLAN_MIN_BYTES,
        }

    def apply_policy(self, reason: str) -> int:
        policy = self.policy
        if policy == POLICY_ON_DEMAND:
            self._wake()
            return 0
        if policy == POLICY_UNSET:
            plan = self.plan()
            if plan["awaiting_choice"] or plan["files"] == 0:
                self._wake()
                return 0
            self._store_policy(POLICY_BACKGROUND)
        return self.enqueue_pending(reason)

    def list_pending(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        root_id: Optional[str] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        totals = self.queue.count_pending(query, model_type, root_id)
        return {
            "items": self.queue.list_pending(query, model_type, root_id, offset, limit),
            "total": totals["total"],
            "total_bytes": totals["total_bytes"],
            "by_type": self.queue.pending_by_type(root_id),
        }

    def queue_all(self) -> int:
        rearmed = self.queue.queue_stats()["failed"]
        self.retry_failed()
        return rearmed + self.enqueue_pending("all")

    def queue_models(self, model_ids: List[str]) -> int:
        return self._queue_picked(self.queue.pending_location_ids_for_models(model_ids))

    def _queue_picked(self, location_ids: List[str]) -> int:
        self.retry_failed(location_ids)
        return self.enqueue_locations(location_ids, "pick", PRIORITY_CHOSEN)

    def queue_matching(
        self,
        query: str = "",
        model_type: Optional[str] = None,
        exclude_model_ids: Optional[List[str]] = None,
    ) -> int:
        ids = self.queue.pending_location_ids(query, model_type, None, exclude_model_ids or ())
        return self._queue_picked(ids)

    def _bump(self) -> None:
        self._version += 1

    def start(self) -> None:
        with self._cv:
            if self._running:
                return
            self._running = True
            self._closing = False
            self._spawn_pool_locked(self.governor.refresh())

    def stop(self, timeout: float = 5.0) -> None:
        with self._cv:
            self._running = False
            self._closing = True
            workers = list(self._workers)
            self._workers = []
            self._pool_signature = None
            for worker in workers:
                worker.stop.set()
            self._cv.notify_all()
        for worker in workers:
            if worker.thread is not None and worker.thread is not threading.current_thread():
                worker.thread.join(timeout)
        self._flush_applied(force=True)

    def _spawn_pool_locked(self, profile: IndexingProfile) -> None:
        self._pool_signature = (profile.workers, profile.priority)
        for index in range(profile.workers):
            worker = _Worker()
            worker.thread = threading.Thread(
                target=self._worker_main, args=(worker, profile), name=f"model-digest-{index}", daemon=True
            )
            self._workers.append(worker)
            worker.thread.start()

    def set_level(self, level: str) -> IndexingProfile:
        profile = self.governor.set_level(level)
        with self._cv:
            if self._running and self._pool_signature != (profile.workers, profile.priority):
                for worker in self._workers:
                    worker.stop.set()
                self._workers = []
                self._spawn_pool_locked(profile)
            self._bump()
            self._cv.notify_all()
        return profile

    def pause(self) -> None:
        self._write_paused(True)
        with self._cv:
            self._paused = True
            self._bump()
            self._cv.notify_all()

    def resume(self) -> None:
        self._write_paused(False)
        with self._cv:
            self._paused = False
            self._bump()
            self._cv.notify_all()

    def enqueue_locations(self, location_ids: List[str], reason: str, priority: int = PRIORITY_BACKGROUND) -> int:
        added = self.queue.enqueue(location_ids, priority, reason, now_iso())
        self._wake()
        return added

    def enqueue_pending(self, reason: str, priority: int = PRIORITY_BACKGROUND) -> int:
        added = self.queue.enqueue_all_pending(priority, reason, now_iso())
        self._wake()
        return added

    def submit_candidates(
        self,
        candidates: List[DigestCandidate],
        on_done: Callable[[DigestCandidate, Optional[FileDigest], Optional[str]], None],
    ) -> None:
        with self._cv:
            for candidate in candidates:
                if candidate.key in self._awaiting:
                    continue
                self._candidates[candidate.key] = candidate
                self._candidate_done[candidate.key] = on_done
            self._bump()
            self._cv.notify_all()

    def reconcile(self) -> int:
        stale = self.queue.delete_stale()
        if stale:
            with self._cv:
                for location_id in stale:
                    job = self._inflight.get(location_id)
                    if job is not None:
                        job.cancel.set()
            self._bump()
        self._release_awaiting()
        return len(stale)

    def _wake(self) -> None:
        with self._cv:
            self._bump()
            self._cv.notify_all()

    def _worker_main(self, worker: _Worker, profile: IndexingProfile) -> None:
        set_current_thread_priority(profile.priority)
        while not worker.stop.is_set():
            try:
                self._flush_if_due()
                self._release_awaiting()
                job = self._claim_next(worker)
            except Exception:
                logger.exception("The digest queue could not be read")
                job = None
            if job is None:
                if self._backfill_next_signature():
                    continue
                with self._cv:
                    if worker.stop.is_set():
                        break
                    self._cv.wait(self._idle_poll_seconds)
                continue
            self._run_job(job, worker)

    def _backfill_next_signature(self) -> bool:
        throttle = self.governor.throttle
        if self._paused or throttle.holding() or not self._backfill_lock.acquire(blocking=False):
            return False
        try:
            with self._cv:
                if self._inflight or self._closing:
                    return False
            rows = self._locations.list_unsigned(1, INLINE_HASH_LIMIT, self._unsignable)
            if not rows:
                return False
            row = rows[0]
            path = self._path_for(row)
            if path is None:
                self._unsignable.add(row["id"])
                return True
            try:
                before = os.stat(path)
            except OSError:
                self._unsignable.add(row["id"])
                return True
            if before.st_size != row["size"] or before.st_mtime_ns != row["mtime_ns"] or not path.is_file():
                self._unsignable.add(row["id"])
                return True
            if not throttle.wait(SIGNATURE_READ_BYTES, lambda: self._paused or self._closing):
                return False
            signature = quick_signature(str(path), before.st_size)
            try:
                after = os.stat(path)
            except OSError:
                after = None
            if signature is None or after is None or (after.st_size, after.st_mtime_ns) != (
                before.st_size, before.st_mtime_ns
            ):
                self._unsignable.add(row["id"])
                return True
            with self._write_lock:
                self._locations.set_quick_sig(row["id"], signature, before.st_size, before.st_mtime_ns)
            return True
        finally:
            self._backfill_lock.release()

    def _run_job(self, job: _Job, worker: Optional[_Worker]) -> None:
        try:
            self._execute(job, worker, self._pacing_for(job))
        except Exception as exc:
            logger.exception("Digest job failed unexpectedly for %s", job.key)
            self._fail_job(job, f"{type(exc).__name__}: {exc}")
        finally:
            self._release(job, worker)

    def _pacing_for(self, job: _Job) -> Optional[_ManagerPacing]:
        if job.priority >= PRIORITY_DEMAND:
            return None
        return _ManagerPacing(self, self.governor.throttle, job)

    def _claim_next(self, worker: Optional[_Worker]) -> Optional[_Job]:
        if self._paused:
            return None
        with self._cv:
            candidate = self._best_candidate_locked()
            rows = self.queue.next_jobs(1, self._clock(), exclude=self._inflight)
            row = rows[0] if rows else None
            if candidate is not None and (row is None or candidate.priority >= row["priority"]):
                job = _Job(
                    key=candidate.key, kind="candidate", path=Path(candidate.path), size=candidate.size,
                    mtime_ns=candidate.mtime_ns, priority=candidate.priority, filename=Path(candidate.path).name,
                    candidate=candidate,
                )
            elif row is not None:
                job = self._job_from_row(row)
            else:
                return None
            self._inflight[job.key] = job
            if worker is not None:
                worker.job = job
            return job

    def _best_candidate_locked(self) -> Optional[DigestCandidate]:
        free = [c for key, c in self._candidates.items() if key not in self._inflight]
        if not free:
            return None
        return sorted(free, key=lambda c: (-c.priority, c.size, c.key))[0]

    def _job_from_row(self, row: Dict[str, Any]) -> _Job:
        return _Job(
            key=row["location_id"], kind="location", path=self._path_for(row), size=row["size"] or 0,
            mtime_ns=row["mtime_ns"], priority=row.get("priority", PRIORITY_DEMAND), filename=row["filename"],
            model_id=row["model_id"], row=row,
        )

    def _release(self, job: _Job, worker: Optional[_Worker]) -> None:
        with self._cv:
            if self._inflight.get(job.key) is job:
                del self._inflight[job.key]
            job.finished.set()
            if worker is not None:
                worker.job = None
            self._cv.notify_all()

    def _cancel_check(self, job: _Job, worker: Optional[_Worker]) -> Callable[[], bool]:
        def cancelled() -> bool:
            if self._closing:
                return True
            if worker is not None and worker.stop.is_set():
                return True
            return job.cancel.is_set()

        return cancelled

    def _execute(self, job: _Job, worker: Optional[_Worker], pacing: Optional[_ManagerPacing]) -> None:
        if job.kind == "candidate":
            self._execute_candidate(job, worker, pacing)
        else:
            self._execute_location(job, worker, pacing)

    def _progress(self, job: _Job):
        last = {"done": 0}

        def on_progress(done: int) -> None:
            self._meter.add(done - last["done"])
            last["done"] = done
            job.done = done
            self._flush_if_due()

        return on_progress

    def _execute_location(self, job: _Job, worker: Optional[_Worker], pacing) -> None:
        if job.path is None:
            self.queue.defer(job.key, self._clock() + OFFLINE_RETRY_SECONDS)
            return
        try:
            before = os.stat(job.path)
        except OSError as exc:
            self._fail_location(job, str(exc))
            return
        if before.st_size != job.size or (job.mtime_ns is not None and before.st_mtime_ns != job.mtime_ns):
            self.queue.remove([job.key])
            self._bump()
            return
        try:
            digest = digest_file(
                str(job.path), before.st_size, self._cancel_check(job, worker), pacing, on_progress=self._progress(job)
            )
        except DigestCancelled:
            return
        except OSError as exc:
            self._fail_location(job, str(exc))
            return
        try:
            after = os.stat(job.path)
        except OSError as exc:
            self._fail_location(job, str(exc))
            return
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            self._fail_location(job, "The file changed while it was being read")
            return
        self._apply(job, digest, after)

    def _execute_candidate(self, job: _Job, worker: Optional[_Worker], pacing) -> None:
        candidate = job.candidate
        try:
            before = os.stat(job.path)
            if before.st_size != candidate.size or (
                candidate.mtime_ns is not None and before.st_mtime_ns != candidate.mtime_ns
            ):
                self._finish_candidate(candidate, None, CANDIDATE_CHANGED_MESSAGE)
                return
            digest = digest_file(
                str(job.path), before.st_size, self._cancel_check(job, worker), pacing,
                on_progress=self._progress(job),
            )
        except DigestCancelled:
            return
        except OSError as exc:
            self._fail_candidate(job, str(exc))
            return
        try:
            after = os.stat(job.path)
        except OSError as exc:
            self._fail_candidate(job, str(exc))
            return
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            self._fail_candidate(job, "The file changed while it was being read")
            return
        self._remember_digest(job.path, after, digest)
        self._complete_candidate(candidate, digest)

    def _complete_candidate(self, candidate: DigestCandidate, digest: FileDigest) -> None:
        if candidate.requires and self.queue.waiting_among(candidate.requires):
            with self._cv:
                candidate.result = digest
                self._candidates.pop(candidate.key, None)
                self._awaiting[candidate.key] = candidate
                self._bump()
            return
        self._finish_candidate(candidate, digest, None)

    def _release_awaiting(self) -> None:
        with self._cv:
            held = list(self._awaiting.values())
        if not held:
            return
        waiting = set(self.queue.waiting_among([key for candidate in held for key in candidate.requires]))
        for candidate in held:
            if waiting.intersection(candidate.requires):
                continue
            with self._cv:
                if self._awaiting.pop(candidate.key, None) is not candidate:
                    continue
            self._finish_candidate(candidate, candidate.result, None)

    def _finish_candidate(self, candidate: DigestCandidate, digest: Optional[FileDigest], error: Optional[str]) -> None:
        with self._cv:
            self._candidates.pop(candidate.key, None)
            callback = self._candidate_done.pop(candidate.key, None)
            self._bump()
        if callback is not None:
            callback(candidate, digest, error)

    def _fail_job(self, job: _Job, error: str) -> None:
        try:
            if job.kind == "candidate":
                self._fail_candidate(job, error)
            else:
                self._fail_location(job, error)
        except Exception:
            logger.exception("Could not record the digest failure for %s", job.key)

    def _fail_candidate(self, job: _Job, error: str) -> None:
        candidate = job.candidate
        candidate.attempts += 1
        if candidate.attempts >= MAX_ATTEMPTS:
            self._finish_candidate(candidate, None, error)

    def _fail_location(self, job: _Job, error: str) -> None:
        self.queue.record_failure(job.key, error, self._clock(), RETRY_DELAYS_SECONDS)
        self._bump()
        self._release_awaiting()

    def _apply(self, job: _Job, digest: FileDigest, stat: os.stat_result) -> None:
        with self._write_lock:
            result = self._locations.apply_digest_and_dequeue(job.key, digest.sha256, digest.quick_sig, digest.size)
        if result.outcome == DIGEST_STALE:
            return
        self._remember_digest(job.path, stat, digest)
        if result.outcome == DIGEST_DUPLICATE:
            self._note_duplicate(result)
        elif result.outcome == DIGEST_MERGED:
            self._note_merge(result)
        self._mark_applied(result.model_id)
        self._release_awaiting()

    def _note_merge(self, result) -> None:
        with self._cv:
            self._merged[result.merged_from] = result.model_id
            while len(self._merged) > MAX_REMEMBERED_MERGES:
                del self._merged[next(iter(self._merged))]
        logger.info(
            "The file '%s' is the same content as a model whose file went missing; it was joined to that model",
            result.model_filename,
        )

    def _merged_into(self, model_id: str) -> Optional[str]:
        with self._cv:
            return self._merged.get(model_id)

    def _merged_digest(self, model_id: str) -> Optional[str]:
        owner = self._merged_into(model_id)
        if owner is None:
            return None
        model = self._models.get_by_id(owner, include_providers=False, include_tags=False)
        return model.sha256 if model is not None else None

    def _remember_digest(self, path: Path, stat: os.stat_result, digest: FileDigest) -> None:
        try:
            self._hash_cache.put(str(path), stat.st_size, stat.st_mtime_ns, digest.sha256)
        except Exception:
            logger.debug("Could not seed the hash cache for %s", path, exc_info=True)

    def _note_duplicate(self, result) -> None:
        logger.warning(
            "Model '%s' has the same content as '%s'; its digest was not recorded on the model",
            result.model_filename, result.owner_filename or "another model",
        )

    def _mark_applied(self, model_id: str) -> None:
        with self._cv:
            if not self._applied:
                self._unflushed_since = self._clock()
            self._applied.add(model_id)
            self._bump()
        self._flush_applied(force=False)

    def _flush_if_due(self) -> None:
        since = self._unflushed_since
        if since is not None and self._clock() - since >= APPLIED_FLUSH_SECONDS:
            self._flush_applied(force=True)

    def _flush_if_due_in_background(self) -> None:
        since = self._unflushed_since
        if since is None or self._clock() - since < APPLIED_FLUSH_SECONDS:
            return
        with self._cv:
            if self._flusher is not None and self._flusher.is_alive():
                return
            self._flusher = threading.Thread(
                target=self._flush_applied, args=(True,), name="model-digest-flush", daemon=True
            )
            self._flusher.start()

    def _flush_applied(self, force: bool) -> None:
        with self._cv:
            if not self._applied:
                return
            due = self._unflushed_since is not None and self._clock() - self._unflushed_since >= APPLIED_FLUSH_SECONDS
            if not (force or due or not self._has_runnable_locked()):
                return
            models = set(self._applied)
            self._applied = set()
            self._unflushed_since = None
        if self._on_applied is not None:
            try:
                self._on_applied(models)
            except Exception:
                logger.exception("Digest applied callback failed")

    def _has_runnable_locked(self) -> bool:
        if self._candidates:
            return True
        return bool(self.queue.next_jobs(1, self._clock(), exclude=self._inflight))

    def ensure_digest(self, model_id: str, timeout: Optional[float] = None) -> Optional[str]:
        model = self._models.get_by_id(model_id, include_providers=False, include_tags=False)
        if model is None:
            return self._merged_digest(model_id)
        if model.sha256:
            return model.sha256
        deadline = None if timeout is None else time.monotonic() + timeout
        for row in self._hashable_locations(model_id):
            self.queue.enqueue([row["location_id"]], PRIORITY_DEMAND, "demand", now_iso())
            job = self._job_from_row({**row, "priority": PRIORITY_DEMAND, "attempts": 0})
            self._run_demand(job, deadline)
            refreshed = self._models.get_by_id(model_id, include_providers=False, include_tags=False)
            if refreshed is None:
                return self._merged_digest(model_id)
            if refreshed.sha256:
                return refreshed.sha256
        return None

    def _hashable_locations(self, model_id: str) -> List[Dict[str, Any]]:
        rows = self.queue.pending_locations_for_model(model_id)
        attempts = self.queue.attempts_among([row["location_id"] for row in rows])
        return [row for row in rows if attempts.get(row["location_id"], 0) < MAX_ATTEMPTS]

    def _run_demand(self, job: _Job, deadline: Optional[float]) -> None:
        with self._cv:
            existing = self._inflight.get(job.key)
            if existing is None:
                self._inflight[job.key] = job
            else:
                existing.demanded = True
                self._cv.notify_all()
        if existing is not None:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            existing.finished.wait(remaining)
            return
        self._run_job(job, None)

    def find_by_digest(
        self,
        sha256: str,
        size: Optional[int],
        hash_candidates: str = HASH_CANDIDATES_NEVER,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> DigestMatch:
        if hash_candidates not in HASH_CANDIDATE_MODES:
            raise ValueError(f"hash_candidates must be one of {', '.join(HASH_CANDIDATE_MODES)}")
        wanted = (sha256 or "").strip().lower()
        if not wanted:
            return DigestMatch()
        model = self._models.get_by_sha256(wanted, include_providers=False)
        if model is not None:
            return DigestMatch(model)
        if not size or size <= 0:
            return DigestMatch()
        rows = [row for row in self._locations.list_present_by_size(size) if row["sha256"] is None]
        if not rows:
            return DigestMatch()
        if hash_candidates == HASH_CANDIDATES_BACKGROUND:
            self._queue_candidates([row["id"] for row in rows])
        if hash_candidates == HASH_CANDIDATES_PACED:
            return self._verify_paced(wanted, [row["id"] for row in rows], timeout, on_progress)
        if hash_candidates != HASH_CANDIDATES_INLINE:
            waiting = len(self.queue.waiting_among([row["id"] for row in rows], self._clock()))
            return DigestMatch(None, len(rows), waiting)
        for model_id in list(dict.fromkeys(row["model_id"] for row in rows))[:MAX_INLINE_CANDIDATES]:
            self.ensure_digest(model_id, timeout)
            model = self._models.get_by_sha256(wanted, include_providers=False)
            if model is not None:
                return DigestMatch(model, len(rows))
        return DigestMatch(None, len(rows))

    def _verify_paced(
        self,
        wanted: str,
        location_ids: List[str],
        timeout: Optional[float],
        on_progress: Optional[Callable[[int, int], None]],
    ) -> DigestMatch:
        attempts = self.queue.attempts_among(location_ids)
        usable = [location_id for location_id in location_ids if attempts.get(location_id, 0) < MAX_ATTEMPTS]
        chosen = sorted(usable, key=lambda location_id: (location_id not in attempts, location_id))[:MAX_PACED_CANDIDATES]
        if chosen:
            self.queue.enqueue(chosen, PRIORITY_CHOSEN, "match", now_iso())
            self._wake()
            deadline = self._monotonic() + (PACED_WAIT_SECONDS if timeout is None else timeout)
            reported = -1
            while True:
                checked = self._checked_count(chosen)
                if on_progress is not None and checked != reported:
                    reported = checked
                    on_progress(checked, len(chosen))
                if not self._claimable(chosen):
                    break
                if self._closing or self._paused or self._monotonic() >= deadline:
                    break
                self._sleep(PACED_POLL_SECONDS)
            if on_progress is not None and self._checked_count(chosen) != reported:
                on_progress(self._checked_count(chosen), len(chosen))
        model = self._models.get_by_sha256(wanted, include_providers=False)
        return DigestMatch(model, len(location_ids))

    def _checked_count(self, location_ids: List[str]) -> int:
        checked = 0
        for location_id in location_ids:
            row = self._locations.get_by_id(location_id)
            if row is not None and row["sha256"]:
                checked += 1
        return checked

    def _claimable(self, location_ids: List[str]) -> List[str]:
        waiting = self.queue.waiting_among(location_ids, self._clock())
        return [location_id for location_id in waiting if self._reachable(location_id)]

    def _reachable(self, location_id: str) -> bool:
        row = self.queue.get(location_id)
        return row is not None and self._path_for(row) is not None

    def _queue_candidates(self, location_ids: List[str]) -> None:
        attempts = self.queue.attempts_among(location_ids)
        waiting = [location_id for location_id in location_ids if attempts.get(location_id, MAX_ATTEMPTS) < MAX_ATTEMPTS]
        fresh = [location_id for location_id in location_ids if location_id not in attempts]
        wanted = waiting + fresh
        chosen = wanted[:MAX_BACKGROUND_CANDIDATES]
        if len(wanted) > len(chosen):
            logger.info(
                "Queued %d of %d same-size files for hashing; the other %d wait for a later lookup",
                len(chosen), len(wanted), len(wanted) - len(chosen),
            )
        if chosen:
            self.queue.enqueue(chosen, PRIORITY_CHOSEN, "match", now_iso())
            self._wake()

    def pending_count(self) -> int:
        return self.queue.queue_stats()["queued"]

    def request_digest(self, model_id: str, retry: bool = False) -> Dict[str, Any]:
        model = self._models.get_by_id(model_id, include_providers=False, include_tags=False)
        if model is None:
            owner = self._merged_into(model_id)
            if owner is None:
                return {"state": "missing", "eta_seconds": None}
            return {"state": "ready", "eta_seconds": 0, "merged_into": owner}
        if model.sha256:
            return {"state": "ready", "eta_seconds": 0}
        pending = self.queue.pending_locations_for_model(model_id)
        if not pending:
            return {"state": "unavailable", "eta_seconds": None}
        if retry:
            self.retry_failed([row["location_id"] for row in pending])
        rows = self._hashable_locations(model_id)
        if not rows:
            return {"state": "failed", "eta_seconds": None}
        self.queue.enqueue([row["location_id"] for row in rows], PRIORITY_DEMAND, "demand", now_iso())
        self._start_demand_thread(model_id)
        rate = self._meter.rate() or DEMAND_RATE_GUESS_BYTES_PER_SECOND
        return {"state": "queued", "eta_seconds": int((rows[0]["size"] or 0) / rate) + 1}

    def _start_demand_thread(self, model_id: str) -> None:
        with self._cv:
            running = self._demand_threads.get(model_id)
            if running is not None and running.is_alive():
                return
            thread = threading.Thread(
                target=self._demand_main, args=(model_id,), name="model-digest-demand", daemon=True
            )
            self._demand_threads[model_id] = thread
            thread.start()

    def _demand_main(self, model_id: str) -> None:
        try:
            self.ensure_digest(model_id, timeout=DEMAND_WAIT_SECONDS)
        finally:
            with self._cv:
                if self._demand_threads.get(model_id) is threading.current_thread():
                    del self._demand_threads[model_id]

    def retry_failed(self, location_ids: Optional[List[str]] = None) -> int:
        retried = self.queue.retry_failed(location_ids)
        if retried:
            self._wake()
        return retried

    def status(self) -> Dict[str, Any]:
        self._flush_if_due_in_background()
        stats = self.queue.queue_stats()
        digests = self.queue.digest_stats()
        profile = self.governor.profile()
        with self._cv:
            workers = list(self._workers)
            candidates = len(self._candidates) + len(self._awaiting)
        current = []
        for worker in workers:
            job = worker.job
            if job is not None:
                current.append(
                    {"model_id": job.model_id, "filename": job.filename, "size": job.size, "done": job.done}
                )
        busy = self.governor.throttle.generation_running()
        has_work = stats["queued"] > 0 or candidates > 0
        held = (not self._paused) and busy and profile.pauses_for_generation and has_work
        remaining = max(0, stats["queued_bytes"] - sum(item["done"] for item in current))
        rate = self._meter.rate()
        eta = None
        if not self._paused and not held and remaining > 0 and rate:
            eta = int(remaining / rate)
        failed = [
            {"path": self._display_path(row), "error": row["last_error"] or ""}
            for row in self.queue.list_failed(STATUS_FAILED_LIMIT)
        ]
        return {
            "policy": self.policy,
            "level": profile.level,
            "levels": [
                {
                    "id": level, "workers": p.workers, "rate_bps": p.read_bytes_per_second,
                    "pauses_for_generation": p.pauses_for_generation,
                }
                for level, p in PROFILES.items()
            ],
            "paused": self._paused,
            "pause_reason": "generation" if held else None,
            "busy": busy,
            "workers": profile.workers,
            "pending_total": digests["pending"],
            "queued": stats["queued"],
            "deferred": max(0, digests["pending"] - stats["queued"] - stats["failed"]),
            "ready": digests["ready"],
            "pending_bytes": digests["pending_bytes"],
            "queued_bytes": stats["queued_bytes"],
            "ready_bytes": digests["ready_bytes"],
            "current": current,
            "throughput_bps": int(rate) if rate else None,
            "eta_seconds": eta,
            "duplicate_checks": candidates,
            "failed": failed,
            "plan": self._plan(stats, digests),
            "version": self._version,
        }

    def _display_path(self, row: Dict[str, Any]) -> str:
        try:
            path = self._path_for(row)
        except Exception:
            path = None
        return str(path) if path is not None else row["rel_path"]

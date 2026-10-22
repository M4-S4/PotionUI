import hashlib
import os
import threading
import time

import pytest

from src.features.models import digests as digests_module
from src.features.models import locations_repository as locations_module
from src.features.models.digest_queue_repository import (
    MAX_ATTEMPTS,
    PRIORITY_BACKGROUND,
    PRIORITY_CHOSEN,
    PRIORITY_DEMAND,
    PRIORITY_TWIN,
    DigestQueueRepository,
)
from src.features.models.digests import DigestCandidate, DigestManager, ThroughputMeter
from src.features.models.file_digest import DigestCancelled, quick_signature
from src.features.models.hash_cache_repository import model_hash_cache_repo
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.locations_repository import ModelLocationsRepository
from src.features.models.read_throttle import ReadThrottle
from src.features.models.repository import model_repo
from src.platform.runtime.thread_priority import ThreadPriority
from src.platform.settings.repository import SettingRepository
from tests.features.models.digest_support import DigestLibrary
from tests.features.models.fakes import FakeClock, drain_manager, wait_for_idle, wait_until

STAMP = "2026-01-01T00:00:00"
MAX_INLINE = digests_module.MAX_INLINE_CANDIDATES


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _job(priority=PRIORITY_BACKGROUND, demanded=False):
    job = digests_module._Job(
        key="k", kind="location", path=None, size=0, mtime_ns=None, priority=priority, filename="f.safetensors"
    )
    job.demanded = demanded
    return job


class Env:
    def __init__(self, tmp_path, real_sleep=False, idle_poll=0.01, write_lock=None):
        self.library = DigestLibrary(tmp_path)
        self.queue = DigestQueueRepository()
        self.locations = ModelLocationsRepository()
        self.clock = FakeClock()
        self.generation_running = [False]
        self.applied = []
        sleeper = time.sleep if real_sleep else self.clock.sleep
        self.governor = IndexingGovernor(
            throttle=ReadThrottle(
                clock=self.clock.monotonic, sleep=sleeper, busy=lambda: self.generation_running[0], poll_seconds=0.01
            )
        )
        self.hidden = set()
        self.manager = DigestManager(
            self.queue,
            self.governor,
            self._path_for,
            locations_write_lock=write_lock,
            on_applied=lambda ids: self.applied.append(set(ids)),
            setting_repository=SettingRepository(),
            clock=self.clock.time,
            monotonic=self.clock.monotonic,
            sleep=sleeper,
            idle_poll_seconds=idle_poll,
            pause_poll_seconds=0.01,
        )

    def _path_for(self, row):
        if row["rel_path"] in self.hidden:
            return None
        return self.library.path_for(row)

    def add(self, name, content, priority=PRIORITY_BACKGROUND, **kwargs):
        item = self.library.add(name, content, **kwargs)
        if kwargs.get("sha256") is None:
            self.queue.enqueue([item.location_id], priority, "test", STAMP)
        return item

    def model(self, item):
        return model_repo.get_by_id(item.model.id, include_providers=False, include_tags=False)


@pytest.fixture
def env(tmp_path, threadsafe_db):
    return Env(tmp_path)


@pytest.fixture
def live_env(tmp_path, threadsafe_db):
    created = Env(tmp_path, real_sleep=True)
    yield created
    created.manager.stop()


@pytest.fixture
def sleepy_env(tmp_path, threadsafe_db):
    created = Env(tmp_path, real_sleep=True, idle_poll=30.0)
    yield created
    created.manager.stop()


@pytest.fixture
def hashed(monkeypatch):
    seen = []
    real = digests_module.digest_file

    def spy(path, *args, **kwargs):
        seen.append(os.path.basename(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(digests_module, "digest_file", spy)
    return seen


@pytest.fixture
def priorities(monkeypatch):
    applied = []
    monkeypatch.setattr(digests_module, "set_current_thread_priority", lambda priority: applied.append(priority))
    return applied


class TestHashingAJob:
    def test_a_pending_location_gets_its_digest_everywhere_it_is_recorded(self, env):
        content = b"weights" * 100
        item = env.add("a.safetensors", content)

        assert drain_manager(env.manager) == 1

        location = env.locations.get_by_id(item.location_id)
        assert (location["sha256"], location["quick_sig"]) == (_sha(content), quick_signature(str(item.path)))
        model = env.model(item)
        assert (model.sha256, model.file_size) == (_sha(content), len(content))
        assert env.queue.get(item.location_id) is None
        cached = model_hash_cache_repo.get(str(item.path))
        stat = item.path.stat()
        assert (cached.sha256, cached.size, cached.mtime_ns) == (_sha(content), stat.st_size, stat.st_mtime_ns)

    def test_applied_models_are_reported_in_one_batch(self, env):
        first = env.add("a.safetensors", b"a" * 50)
        second = env.add("b.safetensors", b"b" * 60)

        drain_manager(env.manager)

        assert env.applied == [{first.model.id, second.model.id}]

    def test_applied_models_go_out_as_one_batch_once_the_oldest_has_waited_long_enough(self, env, monkeypatch):
        first = env.add("a.safetensors", b"a" * 50)
        second = env.add("b.safetensors", b"b" * 60)
        third = env.add("c.safetensors", b"c" * 70)
        real = digests_module.digest_file

        def slow(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            if os.path.basename(path) == "b.safetensors":
                env.clock.wall += 6
            return result

        monkeypatch.setattr(digests_module, "digest_file", slow)

        drain_manager(env.manager)

        assert env.applied == [{first.model.id, second.model.id}, {third.model.id}]

    def test_a_status_request_flushes_models_that_have_waited_past_the_deadline(self, env):
        first = env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 60)
        env.manager._run_job(env.manager._claim_next(None), None)
        assert env.applied == []

        env.manager.status()
        assert env.applied == []
        env.clock.wall += 6
        env.manager.status()

        wait_until(lambda: env.applied)
        assert env.applied == [{first.model.id}]

    def test_a_read_in_progress_flushes_models_that_have_waited_past_the_deadline(self, env, monkeypatch):
        first = env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 60)
        real = digests_module.digest_file
        seen = []

        def reading(path, size, cancel_check=None, pacing=None, on_progress=None, **kwargs):
            if os.path.basename(path) == "b.safetensors":
                env.clock.wall += 6
                on_progress(10)
                seen.append([set(batch) for batch in env.applied])
            return real(path, size, cancel_check, pacing, on_progress=on_progress, **kwargs)

        monkeypatch.setattr(digests_module, "digest_file", reading)

        drain_manager(env.manager)

        assert seen == [[{first.model.id}]]

    def test_an_idle_paused_worker_flushes_models_that_have_waited_past_the_deadline(self, live_env):
        first = live_env.add("a.safetensors", b"a" * 50)
        live_env.add("b.safetensors", b"b" * 60)
        live_env.manager.pause()
        live_env.manager.start()

        live_env.manager._mark_applied(first.model.id)
        assert live_env.applied == []
        live_env.clock.wall += 6

        wait_until(lambda: live_env.applied)
        assert live_env.applied == [{first.model.id}]

    def test_the_version_moves_when_a_digest_lands(self, env):
        env.add("a.safetensors", b"a" * 50)
        before = env.manager.version

        drain_manager(env.manager)

        assert env.manager.version > before

    def test_nothing_runs_when_the_queue_is_empty(self, env):
        assert drain_manager(env.manager) == 0
        assert env.applied == []

    def test_jobs_run_by_priority_then_by_smaller_size(self, env, hashed):
        env.add("bg_big.safetensors", b"1" * 300)
        env.add("bg_small.safetensors", b"2" * 10)
        env.add("chosen.safetensors", b"3" * 200, priority=PRIORITY_CHOSEN)
        env.add("twin.safetensors", b"4" * 100, priority=PRIORITY_TWIN)
        env.add("demand.safetensors", b"5" * 400, priority=PRIORITY_DEMAND)

        drain_manager(env.manager)

        assert hashed == [
            "demand.safetensors", "twin.safetensors", "chosen.safetensors", "bg_small.safetensors", "bg_big.safetensors",
        ]

    def test_background_jobs_are_paced_and_demand_jobs_are_not(self, env):
        env.add("bg.safetensors", b"1" * 50)
        env.add("bg2.safetensors", b"2" * 60)
        drain_manager(env.manager)
        paced = len(env.clock.sleeps)
        env.add("demand.safetensors", b"3" * 70, priority=PRIORITY_DEMAND)
        env.add("demand2.safetensors", b"4" * 80, priority=PRIORITY_DEMAND)

        drain_manager(env.manager)

        assert paced > 0
        assert len(env.clock.sleeps) == paced


class TestJobsThatCannotRun:
    def test_a_location_on_an_offline_root_is_postponed_without_a_failure(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        env.hidden.add("a.safetensors")

        drain_manager(env.manager)

        row = env.queue.get(item.location_id)
        assert (row["attempts"], row["not_before"]) == (0, env.clock.wall + 60)
        assert env.model(item).sha256 is None

    def test_a_missing_file_backs_off_then_gives_up_and_can_be_retried(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        item.path.unlink()

        drain_manager(env.manager)
        first = env.queue.get(item.location_id)
        assert (first["attempts"], first["not_before"]) == (1, env.clock.wall + 60)
        assert drain_manager(env.manager) == 0

        env.clock.wall += 60
        drain_manager(env.manager)
        second = env.queue.get(item.location_id)
        assert (second["attempts"], second["not_before"]) == (2, env.clock.wall + 600)

        env.clock.wall += 600
        drain_manager(env.manager)
        third = env.queue.get(item.location_id)
        assert (third["attempts"], third["not_before"]) == (MAX_ATTEMPTS, None)
        assert drain_manager(env.manager) == 0
        failed = env.manager.status()["failed"]
        assert len(failed) == 1 and failed[0]["error"]

        assert env.manager.retry_failed() == 1
        assert env.queue.get(item.location_id)["attempts"] == 0

    def test_a_file_that_changed_since_it_was_scanned_is_dropped_for_the_next_scan_to_requeue(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        item.path.write_bytes(b"a" * 60)

        drain_manager(env.manager)

        assert env.queue.get(item.location_id) is None
        assert env.model(item).sha256 is None
        assert env.locations.get_by_id(item.location_id)["sha256"] is None

    def test_a_file_that_changes_while_it_is_read_is_not_recorded(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        real = digests_module.digest_file

        def touching(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            return result

        monkeypatch.setattr(digests_module, "digest_file", touching)

        drain_manager(env.manager)

        assert env.locations.get_by_id(item.location_id)["sha256"] is None
        row = env.queue.get(item.location_id)
        assert row["attempts"] == 1 and "changed" in row["last_error"]


class TestApplyingADigest:
    def test_a_model_without_a_digest_takes_it(self, env):
        item = env.add("a.safetensors", b"a" * 50)

        drain_manager(env.manager)

        assert env.model(item).sha256 == _sha(b"a" * 50)

    def test_a_second_copy_with_the_same_content_changes_nothing_on_the_model(self, env):
        content = b"a" * 50
        first = env.add("a.safetensors", content, sha256=_sha(content))
        copy = env.add("a.safetensors", content, rel_path="copies/a.safetensors", model=first.model)

        drain_manager(env.manager)

        assert env.locations.get_by_id(copy.location_id)["sha256"] == _sha(content)
        assert env.locations.get_by_id(copy.location_id)["status"] == "present"
        assert env.model(first).sha256 == _sha(content)

    def test_a_digest_another_model_already_owns_is_logged_and_the_model_is_still_reported_as_applied(self, env, caplog):
        content = b"same" * 20
        env.add("owner.safetensors", content, sha256=_sha(content))
        twin = env.add("twin.safetensors", content)

        with caplog.at_level("WARNING", logger="src.features.models.digests"):
            drain_manager(env.manager)

        assert "Model 'twin.safetensors' has the same content as 'owner.safetensors'" in caplog.text
        assert twin.model.id in env.applied[-1]

    def test_a_location_that_already_has_a_digest_is_left_alone(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        real = digests_module.digest_file

        def racing(path, *args, **kwargs):
            result = real(path, *args, **kwargs)
            env.library.set_digest(item, "r" * 64, "q" * 64)
            return result

        monkeypatch.setattr(digests_module, "digest_file", racing)

        drain_manager(env.manager)

        assert env.locations.get_by_id(item.location_id)["sha256"] == "r" * 64
        assert env.model(item).sha256 is None
        assert env.queue.get(item.location_id) is None


class TestCandidates:
    def _candidate(self, item, key="k1"):
        stat = item.path.stat()
        return DigestCandidate(key=key, path=str(item.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def test_a_candidate_is_hashed_and_handed_back_with_its_digest(self, env):
        item = env.library.add("c.safetensors", b"c" * 70)
        done = []

        env.manager.submit_candidates([self._candidate(item)], lambda c, digest, error: done.append((c.key, digest, error)))
        drain_manager(env.manager)

        key, digest, error = done[0]
        assert (key, error) == ("k1", None)
        assert digest.sha256 == _sha(b"c" * 70)
        assert env.manager.status()["duplicate_checks"] == 0

    def test_candidates_run_before_background_jobs(self, env, hashed):
        env.add("bg.safetensors", b"1" * 10)
        item = env.library.add("c.safetensors", b"c" * 500)

        env.manager.submit_candidates([self._candidate(item)], lambda *args: None)
        drain_manager(env.manager)

        assert hashed == ["c.safetensors", "bg.safetensors"]

    def test_a_candidate_goes_before_a_queued_job_of_the_same_priority(self, env, hashed):
        env.add("queued.safetensors", b"1" * 10, priority=PRIORITY_TWIN)
        item = env.library.add("c.safetensors", b"c" * 500)

        env.manager.submit_candidates([self._candidate(item)], lambda *args: None)
        drain_manager(env.manager)

        assert hashed == ["c.safetensors", "queued.safetensors"]

    def test_a_candidate_that_changed_is_handed_back_with_an_error_at_once(self, env):
        item = env.library.add("c.safetensors", b"c" * 70)
        candidate = self._candidate(item)
        item.path.write_bytes(b"c" * 80)
        done = []

        env.manager.submit_candidates([candidate], lambda c, digest, error: done.append((digest, error)))
        drain_manager(env.manager)

        assert done == [(None, "The file changed after it was found")]

    def test_an_unreadable_candidate_is_retried_then_reported(self, env):
        item = env.library.add("c.safetensors", b"c" * 70)
        candidate = self._candidate(item)
        item.path.unlink()
        done = []

        env.manager.submit_candidates([candidate], lambda c, digest, error: done.append((digest, error)))
        jobs = drain_manager(env.manager, max_jobs=10)

        assert jobs == MAX_ATTEMPTS
        assert len(done) == 1 and done[0][0] is None and done[0][1]


class TestCandidateDigestsAreRemembered:
    def _candidate(self, item):
        stat = item.path.stat()
        return DigestCandidate(key="k1", path=str(item.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)

    def test_the_digest_of_a_candidate_goes_into_the_hash_cache_for_its_size_and_mtime(self, env):
        item = env.library.add("c.safetensors", b"c" * 70)
        stat = item.path.stat()

        env.manager.submit_candidates([self._candidate(item)], lambda *args: None)
        drain_manager(env.manager)

        cached = model_hash_cache_repo.get(str(item.path))
        assert (cached.size, cached.mtime_ns, cached.sha256) == (stat.st_size, stat.st_mtime_ns, _sha(b"c" * 70))

    def test_a_candidate_whose_file_changes_while_it_is_read_is_not_cached_and_is_not_decided_on_stale_bytes(
        self, env, monkeypatch
    ):
        item = env.library.add("c.safetensors", b"c" * 70)
        real = digests_module.digest_file

        def touching(*args, **kwargs):
            result = real(*args, **kwargs)
            stat = os.stat(item.path)
            os.utime(item.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return result

        monkeypatch.setattr(digests_module, "digest_file", touching)
        done = []

        env.manager.submit_candidates([self._candidate(item)], lambda c, digest, error: done.append((digest, error)))
        drain_manager(env.manager, max_jobs=5)

        assert done == [(None, "The file changed after it was found")]
        assert model_hash_cache_repo.get(str(item.path)) is None

    def test_a_cache_that_cannot_be_written_does_not_stop_the_candidate(self, env, monkeypatch):
        item = env.library.add("c.safetensors", b"c" * 70)
        monkeypatch.setattr(model_hash_cache_repo, "put", lambda *args: (_ for _ in ()).throw(RuntimeError("full")))
        done = []

        env.manager.submit_candidates([self._candidate(item)], lambda c, digest, error: done.append((digest, error)))
        drain_manager(env.manager)

        assert len(done) == 1 and done[0][1] is None and done[0][0].sha256 == _sha(b"c" * 70)


class TestCandidatesThatWaitForTwins:
    def _candidate(self, item, requires, key="k1"):
        stat = item.path.stat()
        return DigestCandidate(
            key=key, path=str(item.path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, requires=tuple(requires)
        )

    def _submit(self, env, candidate):
        done = []
        env.manager.submit_candidates([candidate], lambda c, digest, error: done.append((c.key, digest, error)))
        return done

    def test_a_candidate_is_handed_back_only_once_its_twins_have_digests(self, env):
        content = b"t" * 70
        twin = env.add("twin.safetensors", content)
        contested = env.library.add("new.safetensors", content)
        done = self._submit(env, self._candidate(contested, [twin.location_id]))

        drain_manager(env.manager, max_jobs=1)

        assert done == []
        assert env.manager.status()["duplicate_checks"] == 1
        assert env.model(twin).sha256 is None

        drain_manager(env.manager)

        assert [(key, digest.sha256, error) for key, digest, error in done] == [("k1", _sha(content), None)]
        assert env.model(twin).sha256 == _sha(content)
        assert env.manager.status()["duplicate_checks"] == 0

    def test_every_waiting_candidate_is_checked_with_one_queue_query(self, env, monkeypatch):
        for index in range(3):
            twin = env.add(f"twin{index}.safetensors", bytes([index]) * 70)
            contested = env.library.add(f"new{index}.safetensors", bytes([index]) * 70)
            self._submit(env, self._candidate(contested, [twin.location_id], key=f"k{index}"))
        drain_manager(env.manager, max_jobs=3)
        assert env.manager.status()["duplicate_checks"] == 3
        calls = []
        real = env.queue.waiting_among
        monkeypatch.setattr(env.queue, "waiting_among", lambda ids: calls.append(list(ids)) or real(ids))

        env.manager._release_awaiting()

        assert len(calls) == 1 and len(calls[0]) == 3

    def test_nothing_is_asked_of_the_queue_when_no_candidate_waits(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(env.queue, "waiting_among", lambda ids: calls.append(ids) or [])

        env.manager._release_awaiting()

        assert calls == []

    def test_a_candidate_without_requirements_is_handed_back_at_once(self, env):
        contested = env.library.add("new.safetensors", b"n" * 70)
        done = self._submit(env, self._candidate(contested, []))

        drain_manager(env.manager)

        assert len(done) == 1

    def test_a_twin_that_keeps_failing_stops_holding_the_candidate_back(self, env):
        twin = env.add("twin.safetensors", b"t" * 70)
        twin.path.unlink()
        contested = env.library.add("new.safetensors", b"t" * 70)
        done = self._submit(env, self._candidate(contested, [twin.location_id]))
        drain_manager(env.manager, max_jobs=1)
        assert done == []

        for _ in range(MAX_ATTEMPTS):
            drain_manager(env.manager, max_jobs=3)
            env.clock.wall += 1000

        assert len(done) == 1
        assert done[0][1] is not None

    def test_a_twin_that_disappears_stops_holding_the_candidate_back(self, env):
        twin = env.add("twin.safetensors", b"t" * 70)
        contested = env.library.add("new.safetensors", b"t" * 70)
        done = self._submit(env, self._candidate(contested, [twin.location_id]))
        drain_manager(env.manager, max_jobs=1)
        assert done == []

        env.library.set_status(twin, "missing")
        env.manager.reconcile()

        assert len(done) == 1

    def test_submitting_a_candidate_again_while_it_waits_does_not_read_it_twice(self, env, hashed):
        twin = env.add("twin.safetensors", b"t" * 70)
        contested = env.library.add("new.safetensors", b"t" * 70)
        candidate = self._candidate(contested, [twin.location_id])
        done = self._submit(env, candidate)
        drain_manager(env.manager, max_jobs=1)

        done_again = self._submit(env, candidate)
        drain_manager(env.manager)

        assert hashed.count("new.safetensors") == 1
        assert len(done) == 1 and done_again == []


class TestFindByDigest:
    def test_a_model_with_the_digest_is_returned_without_looking_at_anything_else(self, env, hashed):
        content = b"a" * 50
        item = env.add("a.safetensors", content, sha256=_sha(content))

        match = env.manager.find_by_digest(_sha(content).upper() + " ", None, "inline")

        assert (match.model.id, match.candidates) == (item.model.id, 0)
        assert hashed == []

    def test_without_a_size_an_unknown_digest_is_a_plain_miss(self, env):
        env.add("a.safetensors", b"a" * 50)

        assert env.manager.find_by_digest(_sha(b"x"), None, "inline") == digests_module.DigestMatch(None, 0)
        assert env.manager.find_by_digest("", 50, "inline") == digests_module.DigestMatch(None, 0)

    def test_never_counts_the_files_that_could_be_it_and_reads_nothing(self, env, hashed):
        env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 50)
        env.add("other.safetensors", b"c" * 60)

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "never")

        assert (match.model, match.candidates) == (None, 2)
        assert hashed == []
        assert {row["priority"] for row in env.queue.next_jobs(10, 0)} == {PRIORITY_BACKGROUND}

    def test_background_queues_the_candidates_for_a_caller_that_is_waiting_and_reports_a_miss(self, env):
        content = b"a" * 50
        env.add("a.safetensors", content)
        env.add("other.safetensors", b"c" * 60)

        miss = env.manager.find_by_digest(_sha(content), 50, "background")

        assert (miss.model, miss.candidates) == (None, 1)
        assert [(row["filename"], row["priority"]) for row in env.queue.next_jobs(10, 0)][0] == (
            "a.safetensors", PRIORITY_CHOSEN,
        )
        drain_manager(env.manager)
        assert env.manager.find_by_digest(_sha(content), 50, "background").model is not None

    def test_background_candidates_are_read_at_a_paced_priority_never_as_demand(self, env):
        env.add("a.safetensors", b"a" * 50)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "background")

        job = env.manager._claim_next(None)
        assert job.priority < PRIORITY_DEMAND
        assert env.manager._pacing_for(job) is not None
        assert env.queue.get(job.key)["priority"] == PRIORITY_CHOSEN

    def test_only_a_few_candidates_are_queued_per_lookup_and_the_rest_wait_for_the_next_one(self, env, caplog):
        items = [env.library.add(f"m{index:02d}.safetensors", bytes([index]) * 50) for index in range(12)]

        with caplog.at_level("INFO", logger="src.features.models.digests"):
            first = env.manager.find_by_digest(_sha(b"zzz"), 50, "background")

        assert first.candidates == 12
        assert len(_queued_ids(env)) == digests_module.MAX_BACKGROUND_CANDIDATES
        assert "Queued 8 of 12" in caplog.text and "the other 4" in caplog.text

        drain_manager(env.manager)
        env.manager.find_by_digest(_sha(b"zzz"), 50, "background")

        remaining = {item.location_id for item in items if env.model(item).sha256 is None}
        assert len(remaining) == 4 and _queued_ids(env) == remaining

    def test_a_candidate_that_already_failed_out_is_neither_requeued_nor_promoted(self, env):
        items = [env.library.add(f"m{index:02d}.safetensors", bytes([index]) * 50) for index in range(10)]
        env.queue.enqueue([items[0].location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        env.queue.enqueue([items[1].location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(items[1].location_id, "boom", 0.0, ())

        env.manager.find_by_digest(_sha(b"zzz"), 50, "background")

        assert env.queue.get(items[1].location_id)["attempts"] == MAX_ATTEMPTS
        assert env.queue.get(items[1].location_id)["priority"] == PRIORITY_BACKGROUND
        assert env.queue.get(items[0].location_id)["priority"] == PRIORITY_CHOSEN
        promoted = [item for item in items if (env.queue.get(item.location_id) or {}).get("priority") == PRIORITY_CHOSEN]
        assert len(promoted) == digests_module.MAX_BACKGROUND_CANDIDATES

    def test_candidates_that_wait_in_the_queue_are_promoted_ahead_of_new_ones(self, env):
        items = [env.library.add(f"m{index:02d}.safetensors", bytes([index]) * 50) for index in range(10)]
        env.queue.enqueue([items[9].location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "background")

        assert env.queue.get(items[9].location_id)["priority"] == PRIORITY_CHOSEN
        assert env.queue.get(items[8].location_id) is None

    def _pace_spy(self, monkeypatch):
        seen = []
        real = digests_module.digest_file

        def spy(path, size, cancel_check, pacing, *args, **kwargs):
            seen.append((os.path.basename(path), pacing is not None))
            return real(path, size, cancel_check, pacing, *args, **kwargs)

        monkeypatch.setattr(digests_module, "digest_file", spy)
        return seen

    def test_paced_verifies_at_most_two_candidates_through_the_paced_pool_and_returns_the_match(self, env, monkeypatch):
        wanted = b"w" * 50
        env.library.add("a.safetensors", wanted)
        env.library.add("b.safetensors", b"b" * 50)
        env.library.add("c.safetensors", b"c" * 50)
        seen = self._pace_spy(monkeypatch)
        env.clock.on_sleep = lambda: drain_manager(env.manager)

        match = env.manager.find_by_digest(_sha(wanted), 50, "paced")

        assert match.model is not None and match.model.filename == "a.safetensors"
        assert match.candidates == 3
        assert seen == [("a.safetensors", True), ("b.safetensors", True)]

    def test_paced_candidates_are_queued_at_a_paced_priority_never_as_demand(self, env):
        env.library.add("a.safetensors", b"a" * 50)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced", timeout=0)

        assert {row["priority"] for row in env.queue.next_jobs(10, 0)} == {PRIORITY_CHOSEN}
        job = env.manager._claim_next(None)
        assert env.manager._pacing_for(job) is not None

    def test_paced_gives_up_at_the_deadline_without_reading_anything_when_the_pool_does_not_run(self, env, hashed):
        env.library.add("a.safetensors", b"a" * 50)

        match = env.manager.find_by_digest(_sha(b"a" * 50), 50, "paced", timeout=5)

        assert (match.model, match.candidates) == (None, 1)
        assert hashed == []
        assert env.clock.monotonic() >= 5

    def test_paced_does_not_wait_while_the_user_has_paused_indexing(self, env, hashed):
        env.library.add("a.safetensors", b"a" * 50)
        env.manager.pause()

        match = env.manager.find_by_digest(_sha(b"a" * 50), 50, "paced")

        assert (match.model, match.candidates) == (None, 1)
        assert hashed == [] and env.clock.sleeps == []

    def test_paced_skips_a_candidate_that_already_failed_out_and_takes_the_next_two(self, env, monkeypatch):
        bad = env.library.add("a.safetensors", b"a" * 50)
        env.queue.enqueue([bad.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(bad.location_id, "boom", 0.0, ())
        env.library.add("b.safetensors", b"b" * 50)
        env.library.add("c.safetensors", b"c" * 50)
        env.library.add("d.safetensors", b"d" * 50)
        seen = self._pace_spy(monkeypatch)
        env.clock.on_sleep = lambda: drain_manager(env.manager)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced")

        assert [name for name, _ in seen] == ["b.safetensors", "c.safetensors"]

    def test_paced_does_not_wait_for_a_candidate_that_is_backing_off_after_a_failure(self, env, hashed):
        item = env.library.add("a.safetensors", b"a" * 50)
        env.queue.enqueue([item.location_id], PRIORITY_CHOSEN, "match", STAMP)
        env.queue.record_failure(item.location_id, "boom", env.clock.time(), (60.0,))

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "paced")

        assert (match.model, match.candidates) == (None, 1)
        assert env.clock.sleeps == [] and hashed == []

    def test_paced_stops_waiting_once_a_candidate_that_was_claimed_fails(self, env):
        item = env.library.add("a.safetensors", b"a" * 50)
        item.path.unlink()
        env.clock.on_sleep = lambda: drain_manager(env.manager)

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "paced")

        assert (match.model, match.candidates) == (None, 1)
        assert len(env.clock.sleeps) == 1
        assert env.queue.get(item.location_id)["attempts"] == 1

    def test_paced_does_not_wait_for_a_candidate_on_a_root_that_is_offline(self, env, hashed):
        item = env.library.add("a.safetensors", b"a" * 50)
        env.hidden.add(item.path.name)

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "paced")

        assert (match.model, match.candidates) == (None, 1)
        assert env.clock.sleeps == [] and hashed == []

    def test_paced_reports_how_many_candidates_were_really_checked(self, env):
        env.library.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        env.library.add("c.safetensors", b"c" * 50)
        seen = []
        env.clock.on_sleep = lambda: drain_manager(env.manager)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced", on_progress=lambda done, total: seen.append((done, total)))

        assert seen[0] == (0, 2) and seen[-1] == (2, 2)
        assert all(total == 2 for _, total in seen) and [done for done, _ in seen] == sorted(done for done, _ in seen)

    def test_paced_reports_a_file_that_finished_between_the_last_report_and_the_end_of_the_wait(self, env, monkeypatch):
        env.library.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        seen = []
        real = env.manager._claimable
        calls = []

        def finishing(location_ids):
            if not calls:
                drain_manager(env.manager)
            calls.append(1)
            return real(location_ids)

        monkeypatch.setattr(env.manager, "_claimable", finishing)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced", on_progress=lambda done, total: seen.append((done, total)))

        assert seen == [(0, 2), (2, 2)]

    def test_paced_reports_only_what_was_checked_when_the_wait_runs_out(self, env):
        env.library.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        seen = []
        calls = []

        def one_job_then_nothing():
            if not calls:
                drain_manager(env.manager, max_jobs=1)
            calls.append(1)

        env.clock.on_sleep = one_job_then_nothing

        env.manager.find_by_digest(
            _sha(b"zzz"), 50, "paced", timeout=3, on_progress=lambda done, total: seen.append((done, total))
        )

        assert seen == [(0, 2), (1, 2)]

    def test_paced_reports_nothing_checked_when_indexing_is_paused(self, env):
        env.library.add("a.safetensors", b"a" * 50)
        env.manager.pause()
        seen = []

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced", on_progress=lambda done, total: seen.append((done, total)))

        assert seen == [(0, 1)]

    def test_the_wait_for_paced_candidates_is_fifteen_minutes(self):
        assert digests_module.PACED_WAIT_SECONDS == 900

    def test_a_lookup_that_does_not_read_says_how_many_candidates_are_queued_or_running(self, env):
        waiting = env.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        backing = env.add("c.safetensors", b"c" * 50)
        env.queue.record_failure(backing.location_id, "boom", env.clock.time(), (60.0,))
        failed = env.add("d.safetensors", b"d" * 50)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(failed.location_id, "boom", env.clock.time(), (60.0,))

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "never")

        assert (match.candidates, match.waiting) == (4, 1)
        assert waiting.location_id in _queued_ids(env)

    def test_paced_takes_the_candidates_that_are_already_queued_first(self, env, monkeypatch):
        env.library.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        waiting = env.library.add("z.safetensors", b"z" * 50)
        env.queue.enqueue([waiting.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        seen = self._pace_spy(monkeypatch)
        env.clock.on_sleep = lambda: drain_manager(env.manager)

        env.manager.find_by_digest(_sha(b"zzz"), 50, "paced")

        assert [name for name, _ in seen][0] == "z.safetensors" and len(seen) == 2

    def test_paced_with_a_known_digest_or_no_candidates_reads_nothing(self, env, hashed):
        content = b"k" * 50
        env.add("known.safetensors", content, sha256=_sha(content))

        assert env.manager.find_by_digest(_sha(content), 50, "paced").model is not None
        assert env.manager.find_by_digest(_sha(b"zzz"), 77, "paced") == digests_module.DigestMatch(None, 0)
        assert hashed == []

    def test_inline_hashes_the_candidates_and_returns_the_one_that_matches(self, env, hashed):
        wanted = b"w" * 50
        env.add("a.safetensors", b"a" * 50)
        target = env.add("b.safetensors", wanted)
        env.add("other.safetensors", b"c" * 60)

        match = env.manager.find_by_digest(_sha(wanted), 50, "inline")

        assert (match.model.id, match.candidates) == (target.model.id, 2)
        assert "other.safetensors" not in hashed
        assert env.model(target).sha256 == _sha(wanted)

    def test_inline_stops_after_the_cap_and_still_reports_how_many_there_were(self, env, hashed):
        for index in range(MAX_INLINE + 2):
            env.add(f"m{index:02d}.safetensors", bytes([index]) * 50)

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "inline")

        assert (match.model, match.candidates) == (None, MAX_INLINE + 2)
        assert len(hashed) == MAX_INLINE

    def test_a_file_that_is_already_ready_or_gone_is_not_a_candidate(self, env, hashed):
        ready = env.add("ready.safetensors", b"r" * 50, sha256=_sha(b"r" * 50))
        gone = env.add("gone.safetensors", b"g" * 50)
        env.library.set_status(gone, "missing")

        match = env.manager.find_by_digest(_sha(b"zzz"), 50, "inline")

        assert (match.model, match.candidates) == (None, 0)
        assert hashed == []

    def test_a_zero_or_missing_size_never_turns_empty_pending_files_into_candidates(self, env):
        env.add("empty.safetensors", b"")

        assert env.manager.find_by_digest(_sha(b"x"), 0, "never") == digests_module.DigestMatch(None, 0)
        assert env.manager.find_by_digest(_sha(b"x"), None, "never") == digests_module.DigestMatch(None, 0)

    def test_an_unknown_mode_is_refused(self, env):
        with pytest.raises(ValueError, match="hash_candidates"):
            env.manager.find_by_digest(_sha(b"x"), 50, "sometimes")

    def test_the_pending_count_covers_only_files_that_are_going_to_be_hashed(self, env):
        queued = env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 50)
        env.add("c.safetensors", b"c" * 50, sha256=_sha(b"c" * 50))
        never = env.library.add("never.safetensors", b"n" * 50)
        failed = env.add("failed.safetensors", b"f" * 50)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(failed.location_id, "boom", 0.0, ())

        assert env.manager.pending_count() == 2
        assert env.queue.get(never.location_id) is None
        assert queued.location_id in _queued_ids(env)

    def test_files_that_will_never_be_queued_are_not_counted_while_the_choice_is_on_demand(self, env):
        env.library.add("a.safetensors", b"a" * 50)
        env.library.add("b.safetensors", b"b" * 50)
        env.manager.set_policy("on_demand")

        assert env.manager.pending_count() == 0


GIB = 1024 ** 3


def _resize(item, size):
    from src.platform.database import get_cursor

    with get_cursor() as cursor:
        cursor.execute("UPDATE model_locations SET size = ? WHERE id = ?", (size, item.location_id))


def _queued_ids(env):
    return {row["location_id"] for row in env.queue.next_jobs(1000, 0)}


class TestPolicy:
    def _pending(self, env, count, prefix="m"):
        return [env.library.add(f"{prefix}{index:03d}.safetensors", bytes([index % 250]) * 10) for index in range(count)]

    def test_a_fresh_install_has_no_policy_yet(self, env):
        assert env.manager.policy == "unset"

    @pytest.mark.parametrize("policy", ["unset", "background", "on_demand"])
    def test_a_policy_is_remembered(self, env, policy):
        env.manager.set_policy(policy)

        assert env.manager.policy == policy

    def test_an_unknown_policy_is_refused_and_changes_nothing(self, env):
        env.manager.set_policy("background")

        with pytest.raises(ValueError, match="policy"):
            env.manager.set_policy("sometimes")

        assert env.manager.policy == "background"

    def test_a_missing_policy_setting_row_refuses_the_change(self, env):
        SettingRepository().delete_setting_by_key("model_digest_policy")

        with pytest.raises(RuntimeError, match="model_digest_policy"):
            env.manager.set_policy("background")

    def test_an_unreadable_or_garbled_stored_policy_reads_as_unset(self, env):
        SettingRepository().update_setting_value_by_key("model_digest_policy", "warp")

        assert env.manager.policy == "unset"

    def test_choosing_background_queues_everything_that_is_pending(self, env):
        items = self._pending(env, 3)

        queued = env.manager.set_policy("background")

        assert queued == 3
        assert _queued_ids(env) == {item.location_id for item in items}

    def test_choosing_on_demand_drops_the_background_queue_but_keeps_picked_and_demanded_files(self, env):
        background = env.library.add("bg.safetensors", b"1" * 10)
        picked = env.library.add("picked.safetensors", b"2" * 10)
        demanded = env.library.add("demand.safetensors", b"3" * 10)
        env.queue.enqueue([background.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        env.queue.enqueue([picked.location_id], PRIORITY_CHOSEN, "pick", STAMP)
        env.queue.enqueue([demanded.location_id], PRIORITY_DEMAND, "demand", STAMP)

        env.manager.set_policy("on_demand")

        assert _queued_ids(env) == {picked.location_id, demanded.location_id}

    def test_going_back_to_unset_leaves_the_queue_alone(self, env):
        item = env.add("a.safetensors", b"a" * 10)
        env.manager.set_policy("background")

        env.manager.set_policy("unset")

        assert _queued_ids(env) == {item.location_id}

    def test_a_policy_change_moves_the_version(self, env):
        before = env.manager.version

        env.manager.set_policy("on_demand")

        assert env.manager.version > before


class TestPlan:
    def _pending(self, env, count, prefix="m"):
        return [env.library.add(f"{prefix}{index:03d}.safetensors", bytes([index % 250]) * 10) for index in range(count)]

    def test_nothing_pending_means_nothing_to_choose(self, env):
        assert env.manager.plan() == {
            "awaiting_choice": False, "files": 0, "bytes": 0, "min_files": 20, "min_bytes": 64 * GIB,
        }

    def test_twenty_files_do_not_ask_but_twenty_one_do(self, env):
        self._pending(env, 20)
        assert env.manager.plan()["awaiting_choice"] is False

        self._pending(env, 1, prefix="extra")
        plan = env.manager.plan()

        assert (plan["awaiting_choice"], plan["files"]) == (True, 21)

    def test_sixty_four_gib_do_not_ask_but_one_more_byte_does(self, env):
        item = env.library.add("big.safetensors", b"x" * 10)
        _resize(item, 64 * GIB)
        assert env.manager.plan()["awaiting_choice"] is False

        _resize(item, 64 * GIB + 1)
        plan = env.manager.plan()

        assert (plan["awaiting_choice"], plan["files"], plan["bytes"]) == (True, 1, 64 * GIB + 1)

    def test_only_an_unset_policy_ever_asks(self, env):
        self._pending(env, 21)
        env.manager.set_policy("on_demand")

        assert env.manager.plan()["awaiting_choice"] is False
        env.manager.set_policy("unset")
        assert env.manager.plan()["awaiting_choice"] is True

    def test_queued_and_failed_files_are_not_counted_as_waiting_for_a_choice(self, env):
        items = self._pending(env, 24)
        env.queue.enqueue([item.location_id for item in items[:2]], PRIORITY_CHOSEN, "pick", STAMP)
        env.queue.enqueue([items[2].location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(items[2].location_id, "unreadable", 0, ())

        plan = env.manager.plan()

        assert (plan["files"], plan["bytes"]) == (21, 210)

    def test_the_status_block_carries_the_same_plan_and_the_policy(self, env):
        self._pending(env, 21)

        status = env.manager.status()

        assert status["plan"] == env.manager.plan()
        assert (status["policy"], status["deferred"]) == ("unset", 21)


class TestApplyingThePolicy:
    def _pending(self, env, count):
        return [env.library.add(f"m{index:03d}.safetensors", bytes([index % 250]) * 10) for index in range(count)]

    def test_a_small_library_is_indexed_silently_and_remembers_that(self, env):
        items = self._pending(env, 3)

        queued = env.manager.apply_policy("scan")

        assert queued == 3
        assert env.manager.policy == "background"
        assert {row["reason"] for row in env.queue.next_jobs(10, 0)} == {"scan"}
        assert _queued_ids(env) == {item.location_id for item in items}

    def test_an_empty_library_leaves_the_choice_unmade(self, env):
        assert env.manager.apply_policy("scan") == 0
        assert env.manager.apply_policy("startup") == 0
        assert env.manager.policy == "unset"
        assert env.manager.plan()["awaiting_choice"] is False

    def test_a_library_with_nothing_left_to_hash_leaves_the_choice_unmade(self, env):
        env.library.add("ready.safetensors", b"r" * 10, sha256="d" * 64)

        assert env.manager.apply_policy("scan") == 0
        assert env.manager.policy == "unset"

    def test_files_that_are_all_queued_or_failed_do_not_make_the_choice(self, env):
        items = self._pending(env, 2)
        env.queue.enqueue([items[0].location_id], PRIORITY_CHOSEN, "pick", STAMP)
        env.queue.enqueue([items[1].location_id], PRIORITY_CHOSEN, "pick", STAMP)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(items[1].location_id, "boom", 0.0, ())

        assert env.manager.apply_policy("scan") == 0
        assert env.manager.policy == "unset"

    def test_the_first_small_library_makes_the_choice_and_a_later_big_one_is_not_decided_for_the_user(self, env):
        assert env.manager.apply_policy("scan") == 0
        assert env.manager.policy == "unset"
        self._pending(env, 1)

        assert env.manager.apply_policy("scan") == 1
        assert env.manager.policy == "background"

    def test_a_big_first_library_waits_for_the_choice_after_an_empty_start(self, env):
        env.manager.apply_policy("startup")
        self._pending(env, 21)

        assert env.manager.apply_policy("scan") == 0
        assert env.manager.policy == "unset"
        assert env.manager.plan()["awaiting_choice"] is True

    def test_a_big_library_waits_for_the_choice_and_queues_nothing(self, env):
        self._pending(env, 21)

        queued = env.manager.apply_policy("scan")

        assert queued == 0
        assert env.manager.policy == "unset"
        assert _queued_ids(env) == set()

    def test_one_huge_file_is_enough_to_ask(self, env):
        item = env.library.add("huge.safetensors", b"x" * 10)
        _resize(item, 100 * GIB)

        assert env.manager.apply_policy("scan") == 0
        assert env.manager.policy == "unset"

    def test_the_background_policy_queues_every_pass(self, env):
        self._pending(env, 21)
        env.manager.set_policy("background")
        env.queue.clear_background()
        extra = env.library.add("late.safetensors", b"y" * 10)

        queued = env.manager.apply_policy("scan")

        assert queued == 22
        assert extra.location_id in _queued_ids(env)

    def test_the_on_demand_policy_queues_nothing(self, env):
        self._pending(env, 3)
        env.manager.set_policy("on_demand")

        assert env.manager.apply_policy("scan") == 0
        assert _queued_ids(env) == set()

    def test_applying_twice_does_not_queue_a_file_twice(self, env):
        self._pending(env, 3)

        env.manager.apply_policy("scan")
        env.manager.apply_policy("scan")

        assert len(_queued_ids(env)) == 3

    def test_files_that_were_picked_while_the_choice_was_open_are_not_counted_again(self, env):
        items = self._pending(env, 22)
        env.queue.enqueue([item.location_id for item in items[:2]], PRIORITY_CHOSEN, "pick", STAMP)

        assert env.manager.apply_policy("scan") == 20
        assert env.manager.policy == "background"
        assert env.manager.plan()["files"] == 0


class TestExplicitQueueActionsRearmFailedFiles:
    def _failed(self, env, name="a.safetensors"):
        item = env.add(name, b"a" * 50)
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(item.location_id, "mount was gone", 0.0, ())
        return item

    def test_index_all_gives_failed_files_fresh_attempts_and_they_get_hashed(self, env):
        item = self._failed(env)
        fresh = env.library.add("b.safetensors", b"b" * 60)
        queued = env.manager.queue_all()

        assert queued == 2
        row = env.queue.get(item.location_id)
        assert (row["attempts"], row["last_error"], row["not_before"]) == (0, None, None)
        drain_manager(env.manager)
        assert env.model(item).sha256 == _sha(b"a" * 50)
        assert env.model(fresh).sha256 == _sha(b"b" * 60)

    def test_picking_models_re_arms_only_the_picked_failed_files(self, env):
        picked = self._failed(env, "a.safetensors")
        other = self._failed(env, "b.safetensors")

        queued = env.manager.queue_models([picked.model.id])

        assert queued == 1
        assert env.queue.get(picked.location_id)["attempts"] == 0
        assert env.queue.get(other.location_id)["attempts"] == MAX_ATTEMPTS
        drain_manager(env.manager)
        assert env.model(picked).sha256 == _sha(b"a" * 50)
        assert env.model(other).sha256 is None

    def test_picking_by_filter_re_arms_the_failed_files_in_the_match(self, env):
        matched = self._failed(env, "lora_a.safetensors")
        other = self._failed(env, "vae_b.safetensors")

        queued = env.manager.queue_matching("lora", None, [])

        assert queued == 1
        assert env.queue.get(matched.location_id)["attempts"] == 0
        assert env.queue.get(other.location_id)["attempts"] == MAX_ATTEMPTS

    def test_the_automatic_policy_never_re_arms_a_failed_file(self, env):
        item = self._failed(env)
        env.manager.set_policy("background")
        env.manager.apply_policy("scan")

        assert env.queue.get(item.location_id)["attempts"] == MAX_ATTEMPTS
        assert env.manager.status()["failed"] != []

    def test_a_file_that_failed_out_is_not_counted_as_queued_until_it_is_re_armed(self, env):
        item = self._failed(env)

        assert env.manager.status()["queued"] == 0
        assert env.manager.pending_count() == 0

        env.manager.queue_models([item.model.id])

        assert env.manager.status()["queued"] == 1
        assert env.manager.pending_count() == 1


class TestChoosingFiles:
    def test_queue_all_queues_every_pending_file_in_the_background(self, env):
        items = [env.library.add(f"m{i}.safetensors", bytes([i]) * 10) for i in range(3)]

        assert env.manager.queue_all() == 3

        assert {row["priority"] for row in env.queue.next_jobs(10, 0)} == {PRIORITY_BACKGROUND}
        assert _queued_ids(env) == {item.location_id for item in items}

    def test_queue_all_keeps_the_higher_priority_of_a_file_that_was_already_picked(self, env):
        item = env.library.add("a.safetensors", b"a" * 10)
        env.queue.enqueue([item.location_id], PRIORITY_CHOSEN, "pick", STAMP)

        env.manager.queue_all()

        assert env.queue.get(item.location_id)["priority"] == PRIORITY_CHOSEN

    def test_picked_models_are_queued_ahead_of_the_background_work(self, env):
        picked = env.library.add("picked.safetensors", b"p" * 10)
        other = env.library.add("other.safetensors", b"o" * 10)
        env.queue.enqueue([other.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        queued = env.manager.queue_models([picked.model.id, "unknown"])

        assert queued == 1
        assert [row["location_id"] for row in env.queue.next_jobs(10, 0)] == [picked.location_id, other.location_id]
        assert env.queue.get(picked.location_id)["priority"] == PRIORITY_CHOSEN

    def test_matching_queues_the_filtered_set_minus_the_unticked_models(self, env):
        keep = env.library.add("lora_a.safetensors", b"a" * 10)
        skip = env.library.add("lora_b.safetensors", b"b" * 10)
        env.library.add("vae_c.safetensors", b"c" * 10)

        queued = env.manager.queue_matching("lora", None, [skip.model.id])

        assert queued == 1
        assert _queued_ids(env) == {keep.location_id}

    def test_listing_reports_the_page_the_totals_of_the_whole_match_and_the_type_counts(self, env):
        for index in range(5):
            env.library.add(f"m{index}.safetensors", bytes([index]) * 10)

        page = env.manager.list_pending("m", None, None, 0, 2)

        assert [item["filename"] for item in page["items"]] == ["m0.safetensors", "m1.safetensors"]
        assert (page["total"], page["total_bytes"], page["by_type"]) == (5, 50, {"lora": 5})


class TestPause:
    def test_pausing_stops_dispatch_and_is_remembered(self, env):
        env.add("a.safetensors", b"a" * 50)

        env.manager.pause()

        assert drain_manager(env.manager) == 0
        assert env.manager.paused is True
        assert SettingRepository().get_setting_by_key("model_indexing_paused").get_typed_value() is True

    def test_resuming_lets_the_queue_continue(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        env.manager.pause()
        env.manager.resume()

        assert drain_manager(env.manager) == 1

        assert env.model(item).sha256 == _sha(b"a" * 50)
        assert SettingRepository().get_setting_by_key("model_indexing_paused").get_typed_value() is False

    def test_a_missing_pause_setting_row_refuses_the_change_and_leaves_the_manager_running(self, env):
        SettingRepository().delete_setting_by_key("model_indexing_paused")

        with pytest.raises(RuntimeError, match="model_indexing_paused"):
            env.manager.pause()

        assert env.manager.paused is False

    def test_a_new_manager_starts_paused_when_the_flag_was_left_on(self, env):
        env.manager.pause()

        manager = DigestManager(env.queue, env.governor, env._path_for, setting_repository=SettingRepository())

        assert manager.paused is True

    def test_a_pause_holds_a_read_that_is_already_running(self, live_env):
        live_env.manager.pause()
        pacing = digests_module._ManagerPacing(live_env.manager, live_env.governor.throttle, _job())
        finished = threading.Event()

        def run():
            pacing.wait(10)
            finished.set()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        assert finished.wait(0.2) is False
        live_env.manager.resume()
        assert finished.wait(5) is True
        thread.join(5)

    def test_the_hold_covers_both_the_user_pause_and_a_running_generation(self, env):
        pacing = digests_module._ManagerPacing(env.manager, env.governor.throttle, _job())

        assert pacing.holding() is False
        env.generation_running[0] = True
        assert pacing.holding() is True
        env.generation_running[0] = False
        env.manager.pause()
        assert pacing.holding() is True

    def test_a_pause_that_is_cancelled_releases_the_read(self, live_env):
        live_env.manager.pause()
        pacing = digests_module._ManagerPacing(live_env.manager, live_env.governor.throttle, _job())

        assert pacing.wait(10, should_stop=lambda: True) is False

    def test_a_demanded_job_ignores_the_pause_and_a_running_generation(self, live_env):
        live_env.manager.pause()
        live_env.generation_running[0] = True
        pacing = digests_module._ManagerPacing(live_env.manager, live_env.governor.throttle, _job(demanded=True))

        assert pacing.holding() is False
        assert pacing.wait(10 * 1024 * 1024 * 1024) is True
        assert pacing.wait(10, should_stop=lambda: True) is False

    def test_demanding_a_job_that_is_parked_in_the_pause_releases_it(self, live_env):
        live_env.manager.pause()
        job = _job()
        pacing = digests_module._ManagerPacing(live_env.manager, live_env.governor.throttle, job)
        outcome = []
        thread = threading.Thread(target=lambda: outcome.append(pacing.wait(10)), daemon=True)
        thread.start()
        time.sleep(0.1)
        assert outcome == []

        job.demanded = True
        thread.join(5)

        assert outcome == [True]

    def test_demanding_a_job_that_is_parked_for_a_generation_releases_it(self, live_env):
        live_env.generation_running[0] = True
        job = _job()
        pacing = digests_module._ManagerPacing(live_env.manager, live_env.governor.throttle, job)
        outcome = []
        thread = threading.Thread(target=lambda: outcome.append(pacing.wait(10)), daemon=True)
        thread.start()
        time.sleep(0.1)
        assert outcome == []

        job.demanded = True
        thread.join(5)

        assert outcome == [True]


class TestEnsureDigest:
    def test_an_existing_digest_is_returned_without_reading(self, env, hashed):
        content = b"a" * 50
        item = env.add("a.safetensors", content, sha256=_sha(content))

        assert env.manager.ensure_digest(item.model.id) == _sha(content)
        assert hashed == []

    def test_a_pending_model_is_hashed_in_the_callers_thread_without_pacing(self, env):
        content = b"a" * 50
        item = env.add("a.safetensors", content)

        assert env.manager.ensure_digest(item.model.id) == _sha(content)

        assert env.clock.sleeps == []
        assert env.model(item).sha256 == _sha(content)
        assert env.queue.get(item.location_id) is None

    def test_it_ignores_a_user_pause_because_a_caller_is_waiting(self, env):
        content = b"a" * 50
        item = env.add("a.safetensors", content)
        env.manager.pause()

        assert env.manager.ensure_digest(item.model.id) == _sha(content)

    def test_a_model_that_cannot_be_read_has_no_digest(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        env.hidden.add("a.safetensors")

        assert env.manager.ensure_digest(item.model.id) is None
        assert env.queue.get(item.location_id)["priority"] == PRIORITY_DEMAND

    def test_an_unknown_model_has_no_digest(self, env):
        assert env.manager.ensure_digest("missing") is None

    def test_a_caller_waits_for_the_worker_that_already_holds_the_location(self, env):
        content = b"a" * 50
        item = env.add("a.safetensors", content)
        holder = env.manager._claim_next(None)
        assert holder.key == item.location_id
        result = {}

        waiter = threading.Thread(target=lambda: result.update(sha=env.manager.ensure_digest(item.model.id, timeout=5)), daemon=True)
        waiter.start()
        waiter.join(0.2)
        assert waiter.is_alive()
        env.manager._execute(holder, None, None)
        env.manager._release(holder, None)
        waiter.join(5)

        assert result["sha"] == _sha(content)

    def test_repeated_requests_for_the_same_model_share_one_demand_worker(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        started = threading.Event()
        release = threading.Event()
        calls = []
        real = live_env.manager.ensure_digest

        def held(model_id, timeout=None):
            calls.append(model_id)
            started.set()
            release.wait(5)
            return real(model_id, timeout)

        monkeypatch.setattr(live_env.manager, "ensure_digest", held)

        first = live_env.manager.request_digest(item.model.id)
        assert started.wait(5)
        second = live_env.manager.request_digest(item.model.id)
        third = live_env.manager.request_digest(item.model.id)
        release.set()
        wait_until(lambda: live_env.model(item).sha256 is not None)

        assert [first["state"], second["state"], third["state"]] == ["queued", "queued", "queued"]
        assert calls == [item.model.id]

    def test_a_finished_demand_worker_is_replaced_by_the_next_request(self, live_env):
        first_item = live_env.add("a.safetensors", b"a" * 50)
        second_item = live_env.add("b.safetensors", b"b" * 50)

        live_env.manager.request_digest(first_item.model.id)
        wait_until(lambda: live_env.model(first_item).sha256 is not None)
        wait_until(lambda: not live_env.manager._demand_threads)
        live_env.manager.request_digest(second_item.model.id)
        wait_until(lambda: live_env.model(second_item).sha256 is not None)

        assert live_env.model(first_item).sha256 and live_env.model(second_item).sha256

    def test_request_digest_reports_the_state_and_an_estimate(self, live_env):
        content = b"a" * 50
        item = live_env.add("a.safetensors", content)

        queued = live_env.manager.request_digest(item.model.id)

        assert queued["state"] == "queued" and isinstance(queued["eta_seconds"], int)
        deadline = time.monotonic() + 5
        while live_env.model(item).sha256 is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert live_env.model(item).sha256 == _sha(content)
        assert live_env.manager.request_digest(item.model.id) == {"state": "ready", "eta_seconds": 0}
        assert live_env.manager.request_digest("missing")["state"] == "missing"


class TestReconcile:
    def test_a_running_read_is_cancelled_when_its_location_disappears(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        keep = env.add("b.safetensors", b"b" * 50)
        real = digests_module.digest_file

        def vanishing(path, size, cancel_check=None, *args, **kwargs):
            if path.endswith("a.safetensors"):
                env.library.set_status(item, "missing")
                env.manager.reconcile()
                assert cancel_check() is True
            return real(path, size, cancel_check, *args, **kwargs)

        monkeypatch.setattr(digests_module, "digest_file", vanishing)

        drain_manager(env.manager)

        assert env.locations.get_by_id(item.location_id)["sha256"] is None
        assert env.queue.get(item.location_id) is None
        assert env.model(keep).sha256 == _sha(b"b" * 50)

    def test_reconcile_reports_how_many_rows_it_dropped(self, env):
        done = env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 50)
        env.library.set_digest(done, "z" * 64)

        assert env.manager.reconcile() == 1


class TestStatus:
    def test_counts_bytes_and_levels(self, env):
        env.add("queued.safetensors", b"q" * 100)
        env.library.add("deferred.safetensors", b"d" * 40)
        env.add("ready.safetensors", b"r" * 25, sha256="r" * 64)

        status = env.manager.status()

        assert (status["pending_total"], status["queued"], status["deferred"], status["ready"]) == (2, 1, 1, 1)
        assert (status["pending_bytes"], status["queued_bytes"], status["ready_bytes"]) == (140, 100, 25)
        assert status["level"] == "low" and status["workers"] == 1
        assert [(l["id"], l["workers"], l["rate_bps"], l["pauses_for_generation"]) for l in status["levels"]] == [
            ("low", 1, 96 * 1024 * 1024, True),
            ("normal", 2, 384 * 1024 * 1024, True),
            ("max", 4, None, False),
        ]
        assert status["current"] == [] and status["failed"] == []

    def test_a_running_generation_holds_background_work_and_says_so(self, env):
        env.add("a.safetensors", b"a" * 50)
        env.generation_running[0] = True

        status = env.manager.status()

        assert (status["busy"], status["paused"], status["pause_reason"]) == (True, False, "generation")

    def test_the_user_pause_wins_over_the_generation_hold(self, env):
        env.add("a.safetensors", b"a" * 50)
        env.generation_running[0] = True
        env.manager.pause()

        status = env.manager.status()

        assert (status["busy"], status["paused"], status["pause_reason"]) == (True, True, None)

    def test_max_does_not_pause_for_a_generation_but_still_reports_it_as_busy(self, env):
        env.add("a.safetensors", b"a" * 50)
        env.generation_running[0] = True
        env.governor.set_level("max")

        status = env.manager.status()

        assert (status["busy"], status["pause_reason"]) == (True, None)

    def test_an_idle_queue_has_no_hold(self, env):
        env.generation_running[0] = True

        assert env.manager.status()["pause_reason"] is None

    def test_the_estimate_appears_after_the_warmup_and_follows_the_measured_rate(self, env):
        env.add("first.safetensors", b"f" * 1000)
        drain_manager(env.manager)
        env.add("second.safetensors", b"s" * 5000)

        assert env.manager.status()["eta_seconds"] is None
        env.clock.now += 20

        status = env.manager.status()
        assert status["throughput_bps"] == 50
        assert status["eta_seconds"] == 100

    def test_the_estimate_counts_only_what_is_left_of_the_file_being_read(self, live_env, priorities, monkeypatch):
        live_env.add("a.safetensors", b"a" * 100)
        entered = threading.Event()
        release = threading.Event()
        real = digests_module.digest_file

        def partway(path, size, cancel_check=None, pacing=None, chunk_size=4 * 1024 * 1024, on_progress=None):
            on_progress(25)
            entered.set()
            release.wait(5)
            return real(path, size, cancel_check, pacing)

        monkeypatch.setattr(digests_module, "digest_file", partway)
        live_env.manager.start()
        assert entered.wait(5)
        live_env.clock.now = 10

        status = live_env.manager.status()
        release.set()

        assert [item["done"] for item in status["current"]] == [25]
        assert status["eta_seconds"] == 30

    def test_no_estimate_while_paused(self, env):
        env.add("first.safetensors", b"f" * 1000)
        drain_manager(env.manager)
        env.add("second.safetensors", b"s" * 5000)
        env.clock.now += 20
        env.manager.pause()

        assert env.manager.status()["eta_seconds"] is None

    def test_failed_jobs_are_listed_with_their_path_and_error(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        item.path.unlink()
        for _ in range(MAX_ATTEMPTS):
            drain_manager(env.manager)
            env.clock.wall += 1000

        status = env.manager.status()

        failed = status["failed"]
        assert failed == [{"path": str(item.path), "error": failed[0]["error"]}]
        assert failed[0]["error"]
        assert (status["queued"], status["deferred"], status["pending_total"]) == (0, 0, 1)


class TestThroughputMeter:
    def test_no_rate_before_the_warmup_then_bytes_over_elapsed_time(self):
        clock = FakeClock()
        meter = ThroughputMeter(clock.monotonic)
        meter.add(1000)
        clock.now = 5

        assert meter.rate() is None
        clock.now = 20

        assert meter.rate() == 50

    def test_old_samples_leave_the_window_and_the_meter_restarts(self):
        clock = FakeClock()
        meter = ThroughputMeter(clock.monotonic)
        meter.add(1000)
        clock.now = 100

        assert meter.rate() is None
        meter.add(500)
        clock.now = 120
        assert meter.rate() == 25


class TestThePool:
    def test_workers_drain_the_queue_and_stop_cleanly(self, live_env, priorities):
        items = [live_env.add(f"{i}.safetensors", bytes([i + 1]) * 100) for i in range(4)]

        live_env.manager.start()

        assert wait_for_idle(live_env.manager, 10) is True
        assert all(live_env.model(item).sha256 for item in items)
        live_env.manager.stop()
        assert set().union(*live_env.applied) == {item.model.id for item in items}
        assert priorities and set(priorities) == {ThreadPriority.BACKGROUND}

    def test_stopping_reports_models_that_were_still_waiting_for_the_next_batch(self, env):
        env.add("queued.safetensors", b"q" * 50)
        env.manager._mark_applied("m1")
        assert env.applied == []

        env.manager.stop()

        assert env.applied == [{"m1"}]

    def test_the_pool_has_one_worker_at_low_and_follows_level_changes(self, live_env, priorities):
        live_env.manager.start()
        assert _live_workers() == 1

        live_env.manager.set_level("normal")
        wait_until(lambda: _live_workers() == 2)
        live_env.manager.set_level("max")
        wait_until(lambda: _live_workers() == 4)

        assert live_env.manager.status()["level"] == "max"
        assert ThreadPriority.NORMAL in priorities
        live_env.manager.stop()
        wait_until(lambda: _live_workers() == 0)

    def test_a_paused_pool_leaves_the_queue_alone(self, live_env, priorities, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        polls = []
        real = live_env.manager._claim_next

        def counting(worker):
            result = real(worker)
            polls.append(result)
            return result

        monkeypatch.setattr(live_env.manager, "_claim_next", counting)
        live_env.manager.pause()
        live_env.manager.start()

        wait_until(lambda: len(polls) >= 5)

        assert polls[:5] == [None] * 5
        assert live_env.model(item).sha256 is None
        assert live_env.queue.get(item.location_id) is not None
        live_env.manager.resume()
        assert wait_for_idle(live_env.manager, 10) is True
        assert live_env.model(item).sha256 == _sha(b"a" * 50)

    def test_the_last_models_are_reported_as_soon_as_the_queue_runs_dry(self, live_env, priorities):
        item = live_env.add("a.safetensors", b"a" * 50)

        live_env.manager.start()

        wait_until(lambda: live_env.applied)
        assert live_env.applied == [{item.model.id}]

    def test_enqueueing_wakes_an_idle_pool(self, sleepy_env, priorities):
        sleepy_env.manager.start()
        time.sleep(0.1)
        item = sleepy_env.library.add("a.safetensors", b"a" * 50)

        sleepy_env.manager.enqueue_locations([item.location_id], "test")

        assert wait_for_idle(sleepy_env.manager, 3) is True
        assert sleepy_env.model(item).sha256 == _sha(b"a" * 50)

    def test_changing_the_level_cancels_the_read_in_progress_and_a_new_worker_finishes_it(
        self, live_env, priorities, monkeypatch
    ):
        item = live_env.add("a.safetensors", b"a" * 50)
        calls = []
        first_started = threading.Event()
        release = threading.Event()
        real = digests_module.digest_file

        def blocking_once(path, size, cancel_check=None, *args, **kwargs):
            calls.append(path)
            if len(calls) == 1:
                first_started.set()
                while not release.is_set():
                    if cancel_check():
                        raise DigestCancelled(path)
                    time.sleep(0.005)
            return real(path, size, cancel_check, *args, **kwargs)

        monkeypatch.setattr(digests_module, "digest_file", blocking_once)
        live_env.manager.start()
        assert first_started.wait(5)

        try:
            live_env.manager.set_level("normal")
            wait_until(lambda: live_env.model(item).sha256 is not None)
        finally:
            release.set()

        assert len(calls) == 2
        assert live_env.model(item).sha256 == _sha(b"a" * 50)

    def test_stopping_cancels_a_read_made_on_a_callers_behalf(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        started = threading.Event()
        result = {}

        def blocking(path, size, cancel_check=None, *args, **kwargs):
            started.set()
            while not cancel_check():
                time.sleep(0.005)
            raise DigestCancelled(path)

        monkeypatch.setattr(digests_module, "digest_file", blocking)
        caller = threading.Thread(target=lambda: result.update(sha=live_env.manager.ensure_digest(item.model.id)), daemon=True)
        caller.start()
        assert started.wait(5)

        live_env.manager.stop()
        caller.join(5)

        assert not caller.is_alive()
        assert result["sha"] is None

    def test_enqueue_pending_queues_every_unhashed_location_once(self, live_env):
        first = live_env.library.add("a.safetensors", b"a" * 50)
        live_env.library.add("b.safetensors", b"b" * 60)

        assert live_env.manager.enqueue_pending("scan") == 2
        assert live_env.manager.enqueue_pending("scan") == 0
        assert first.location_id in {row["location_id"] for row in live_env.queue.next_jobs(10, 0)}

    def test_stopping_a_pool_that_is_hashing_cancels_the_read(self, live_env, priorities, monkeypatch):
        live_env.add("a.safetensors", b"a" * 50)
        entered = threading.Event()
        release = threading.Event()
        real = digests_module.digest_file

        def slow(path, size, cancel_check=None, *args, **kwargs):
            entered.set()
            while not release.is_set():
                if cancel_check():
                    raise DigestCancelled(path)
                time.sleep(0.005)
            return real(path, size, cancel_check, *args, **kwargs)

        monkeypatch.setattr(digests_module, "digest_file", slow)
        live_env.manager.start()
        assert entered.wait(5)

        live_env.manager.stop()

        assert _live_workers() == 0
        assert len(live_env.queue.next_jobs(1, 0)) == 1

class TestUnexpectedFailures:
    def _raising(self, monkeypatch, error):
        reads = []

        def failing(path, *args, **kwargs):
            reads.append(os.path.basename(path))
            raise error

        monkeypatch.setattr(digests_module, "digest_file", failing)
        return reads

    def test_an_unexpected_error_counts_an_attempt_backs_off_and_is_listed_after_the_third(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        reads = self._raising(monkeypatch, RuntimeError("boom"))

        drain_manager(env.manager, max_jobs=5)
        first = env.queue.get(item.location_id)
        assert (first["attempts"], first["not_before"]) == (1, env.clock.wall + 60)
        assert "RuntimeError: boom" in first["last_error"]
        assert drain_manager(env.manager, max_jobs=5) == 0

        env.clock.wall += 60
        drain_manager(env.manager, max_jobs=5)
        second = env.queue.get(item.location_id)
        assert (second["attempts"], second["not_before"]) == (2, env.clock.wall + 600)

        env.clock.wall += 600
        drain_manager(env.manager, max_jobs=5)
        third = env.queue.get(item.location_id)
        assert (third["attempts"], third["not_before"]) == (MAX_ATTEMPTS, None)
        env.clock.wall += 100_000
        assert drain_manager(env.manager, max_jobs=5) == 0

        assert reads == ["a.safetensors"] * MAX_ATTEMPTS
        failed = env.manager.status()["failed"]
        assert len(failed) == 1 and "boom" in failed[0]["error"]

    def test_the_same_row_is_not_read_again_in_a_loop_by_the_pool(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        reads = self._raising(monkeypatch, RuntimeError("boom"))

        live_env.manager.start()
        wait_until(lambda: live_env.queue.get(item.location_id)["attempts"] >= 1)
        time.sleep(0.2)

        assert len(reads) == 1
        assert live_env.queue.get(item.location_id)["not_before"] is not None

    def test_a_candidate_that_keeps_failing_unexpectedly_is_reported_after_the_third_attempt(self, env, monkeypatch):
        reads = self._raising(monkeypatch, ValueError("odd"))
        item = env.add("a.safetensors", b"a" * 50, sha256=_sha(b"a" * 50))
        candidate = DigestCandidate(key="c1", path=str(item.path), size=50, mtime_ns=item.path.stat().st_mtime_ns)
        done = []
        env.manager.submit_candidates([candidate], lambda c, digest, error: done.append((digest, error)))

        ran = drain_manager(env.manager, max_jobs=10)

        assert ran == MAX_ATTEMPTS
        assert len(reads) == MAX_ATTEMPTS
        assert done == [(None, "ValueError: odd")]
        assert env.manager.status()["duplicate_checks"] == 0

    def test_a_failure_while_recording_a_failure_does_not_stop_the_worker(self, env, monkeypatch):
        env.add("a.safetensors", b"a" * 50)
        env.add("b.safetensors", b"b" * 60)
        self._raising(monkeypatch, RuntimeError("boom"))

        def broken(*args, **kwargs):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(env.queue, "record_failure", broken)

        assert drain_manager(env.manager, max_jobs=4) == 4

    def test_a_pool_worker_survives_a_queue_that_cannot_be_read_for_a_moment(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        real = live_env.manager._claim_next
        failures = []

        def flaky(worker):
            if len(failures) < 2:
                failures.append(1)
                raise RuntimeError("database is locked")
            return real(worker)

        monkeypatch.setattr(live_env.manager, "_claim_next", flaky)

        live_env.manager.start()

        wait_until(lambda: live_env.model(item).sha256 is not None)
        assert len(failures) == 2

    def test_a_failure_in_the_model_step_leaves_the_row_pending_and_a_retry_applies(self, env, monkeypatch):
        content = b"a" * 50
        item = env.add("a.safetensors", content)
        real = locations_module.model_repo.update_digest
        calls = []

        def failing_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("disk I/O error")
            return real(*args, **kwargs)

        monkeypatch.setattr(locations_module.model_repo, "update_digest", failing_once)

        drain_manager(env.manager, max_jobs=5)

        row = env.queue.get(item.location_id)
        assert (row["attempts"], row["not_before"]) == (1, env.clock.wall + 60)
        assert env.locations.get_by_id(item.location_id)["sha256"] is None
        assert env.model(item).sha256 is None
        assert [r["location_id"] for r in env.queue.pending_locations_for_model(item.model.id)] == [item.location_id]
        assert env.applied == []

        env.clock.wall += 60
        drain_manager(env.manager, max_jobs=5)

        assert env.locations.get_by_id(item.location_id)["sha256"] == _sha(content)
        assert env.model(item).sha256 == _sha(content)
        assert env.queue.get(item.location_id) is None
        assert env.applied == [{item.model.id}]


class TestDemandOnAParkedJob:
    def _park(self, live_env, item, user_pause=False):
        live_env.generation_running[0] = True
        live_env.manager.start()
        wait_until(lambda: live_env.manager.status()["current"])
        if user_pause:
            live_env.manager.pause()
        assert live_env.model(item).sha256 is None

    def _demand(self, live_env, item, timeout=5):
        result = {}
        started = time.monotonic()
        thread = threading.Thread(
            target=lambda: result.update(sha=live_env.manager.ensure_digest(item.model.id, timeout=timeout)), daemon=True
        )
        thread.start()
        thread.join(timeout + 2)
        return result, time.monotonic() - started

    def test_a_demand_releases_a_pool_job_parked_for_a_generation(self, live_env):
        content = b"a" * 50
        item = live_env.add("a.safetensors", content)
        self._park(live_env, item)

        result, elapsed = self._demand(live_env, item)

        assert result["sha"] == _sha(content)
        assert elapsed < 2

    def test_a_demand_releases_a_pool_job_parked_by_the_user_pause_as_well(self, live_env):
        content = b"a" * 50
        item = live_env.add("a.safetensors", content)
        self._park(live_env, item, user_pause=True)

        result, elapsed = self._demand(live_env, item)

        assert result["sha"] == _sha(content)
        assert elapsed < 2
        assert live_env.manager.paused is True

    def test_a_demand_for_a_read_that_never_finishes_gives_up_at_its_timeout(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        started = threading.Event()

        def stuck(path, size, cancel_check=None, *args, **kwargs):
            started.set()
            while not cancel_check():
                time.sleep(0.01)
            raise DigestCancelled(path)

        monkeypatch.setattr(digests_module, "digest_file", stuck)
        live_env.manager.start()
        assert started.wait(5)

        result, elapsed = self._demand(live_env, item, timeout=0.3)

        assert result == {"sha": None}
        assert 0.25 <= elapsed < 3

    def test_a_requested_digest_stops_waiting_on_a_stuck_read_after_the_demand_timeout(self, live_env, monkeypatch):
        item = live_env.add("a.safetensors", b"a" * 50)
        started = threading.Event()

        def stuck(path, size, cancel_check=None, *args, **kwargs):
            started.set()
            while not cancel_check():
                time.sleep(0.01)
            raise DigestCancelled(path)

        monkeypatch.setattr(digests_module, "digest_file", stuck)
        monkeypatch.setattr(digests_module, "DEMAND_WAIT_SECONDS", 0.3)
        live_env.manager.start()
        assert started.wait(5)

        assert live_env.manager.request_digest(item.model.id)["state"] == "queued"

        wait_until(lambda: not live_env.manager._demand_threads, timeout=3)
        assert live_env.manager.status()["current"]


class TestApplyingUnderTheWriteLock:
    def test_the_apply_waits_for_the_shared_locations_write_lock(self, tmp_path, threadsafe_db):
        lock = threading.Lock()
        env = Env(tmp_path, write_lock=lock)
        content = b"a" * 50
        item = env.add("a.safetensors", content)
        finished = threading.Event()
        thread = threading.Thread(target=lambda: (drain_manager(env.manager), finished.set()), daemon=True)

        with lock:
            thread.start()
            assert finished.wait(0.3) is False
            assert env.locations.get_by_id(item.location_id)["sha256"] is None
            assert env.model(item).sha256 is None
        assert finished.wait(5) is True

        assert env.locations.get_by_id(item.location_id)["sha256"] == _sha(content)
        assert env.model(item).sha256 == _sha(content)


class TestBestEffortSteps:
    def test_a_hash_cache_that_cannot_be_written_does_not_stop_the_digest_being_recorded(self, env, monkeypatch):
        content = b"a" * 50
        item = env.add("a.safetensors", content)

        def refusing(*args, **kwargs):
            raise RuntimeError("cache is read-only")

        monkeypatch.setattr(env.manager._hash_cache, "put", refusing)

        drain_manager(env.manager)

        assert env.model(item).sha256 == _sha(content)
        assert env.locations.get_by_id(item.location_id)["sha256"] == _sha(content)
        assert env.applied == [{item.model.id}]

    def test_a_failing_applied_callback_does_not_lose_the_digests_or_later_batches(self, env):
        first = env.add("a.safetensors", b"a" * 50)
        calls = []

        def flaky(model_ids):
            calls.append(set(model_ids))
            if len(calls) == 1:
                raise RuntimeError("projector is down")

        env.manager._on_applied = flaky

        drain_manager(env.manager)
        second = env.add("b.safetensors", b"b" * 60)
        drain_manager(env.manager)

        assert calls == [{first.model.id}, {second.model.id}]
        assert env.model(first).sha256 and env.model(second).sha256

    def test_a_failing_applied_callback_during_a_read_does_not_fail_that_read(self, env, monkeypatch):
        env.add("a.safetensors", b"a" * 50)
        second = env.add("b.safetensors", b"b" * 60)
        real = digests_module.digest_file

        def reading(path, size, cancel_check=None, pacing=None, on_progress=None, **kwargs):
            if os.path.basename(path) == "b.safetensors":
                env.clock.wall += 6
                on_progress(10)
            return real(path, size, cancel_check, pacing, on_progress=on_progress, **kwargs)

        def broken(model_ids):
            raise RuntimeError("projector is down")

        monkeypatch.setattr(digests_module, "digest_file", reading)
        env.manager._on_applied = broken

        drain_manager(env.manager, max_jobs=5)

        assert env.model(second).sha256 == _sha(b"b" * 60)
        assert env.queue.get(second.location_id) is None
        assert env.manager.status()["failed"] == []


class TestFailedFiles:
    def _fail(self, env, item):
        for _ in range(MAX_ATTEMPTS):
            env.queue.record_failure(item.location_id, "boom", 0.0, ())

    def test_a_file_that_failed_out_answers_failed_and_is_not_read_again(self, env, hashed, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        self._fail(env, item)
        started = []
        monkeypatch.setattr(env.manager, "_start_demand_thread", lambda model_id: started.append(model_id))

        first = env.manager.request_digest(item.model.id)
        second = env.manager.request_digest(item.model.id)

        assert first == second == {"state": "failed", "eta_seconds": None}
        assert started == [] and hashed == []
        assert env.queue.get(item.location_id)["priority"] == PRIORITY_BACKGROUND

    def test_a_model_with_one_good_and_one_failed_copy_still_hashes_the_good_one(self, env, monkeypatch):
        bad = env.add("a.safetensors", b"a" * 50)
        good = env.library.add("b.safetensors", b"a" * 50, model=bad.model, rel_path="sub/b.safetensors")
        env.queue.enqueue([good.location_id], PRIORITY_BACKGROUND, "test", STAMP)
        self._fail(env, bad)
        monkeypatch.setattr(env.manager, "_start_demand_thread", lambda model_id: None)

        result = env.manager.request_digest(bad.model.id)

        assert result["state"] == "queued"
        assert env.queue.get(good.location_id)["priority"] == PRIORITY_DEMAND
        assert env.queue.get(bad.location_id)["priority"] == PRIORITY_BACKGROUND

    def test_a_retry_gives_a_failed_file_fresh_attempts_and_hashes_it(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        self._fail(env, item)
        started = []
        monkeypatch.setattr(env.manager, "_start_demand_thread", lambda model_id: started.append(model_id))

        result = env.manager.request_digest(item.model.id, retry=True)

        assert result["state"] == "queued"
        assert started == [item.model.id]
        row = env.queue.get(item.location_id)
        assert (row["attempts"], row["last_error"]) == (0, None)

    def test_a_retry_only_touches_the_files_of_that_model(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        other = env.add("b.safetensors", b"b" * 60)
        self._fail(env, item)
        self._fail(env, other)
        monkeypatch.setattr(env.manager, "_start_demand_thread", lambda model_id: None)

        env.manager.request_digest(item.model.id, retry=True)

        assert env.queue.get(item.location_id)["attempts"] == 0
        assert env.queue.get(other.location_id)["attempts"] == MAX_ATTEMPTS

    def test_a_retry_of_a_healthy_file_changes_nothing_else(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        monkeypatch.setattr(env.manager, "_start_demand_thread", lambda model_id: None)

        assert env.manager.request_digest(item.model.id, retry=True)["state"] == "queued"

    def test_repeated_demand_for_an_unreadable_file_stops_after_the_attempt_limit(self, env, monkeypatch):
        item = env.add("a.safetensors", b"a" * 50)
        item.path.unlink()
        runs = []
        real = env.manager._run_demand
        monkeypatch.setattr(env.manager, "_run_demand", lambda job, deadline: runs.append(job.key) or real(job, deadline))

        for _ in range(MAX_ATTEMPTS + 3):
            assert env.manager.ensure_digest(item.model.id) is None

        assert len(runs) == MAX_ATTEMPTS
        assert env.manager.request_digest(item.model.id)["state"] == "failed"

    def test_ensure_digest_does_not_read_a_failed_file_either(self, env, hashed):
        item = env.add("a.safetensors", b"a" * 50)
        self._fail(env, item)

        assert env.manager.ensure_digest(item.model.id) is None
        assert hashed == []

    def test_the_status_lists_the_failed_file_the_request_refuses(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        self._fail(env, item)

        failed = env.manager.status()["failed"]

        assert [entry["error"] for entry in failed] == ["boom"]


class TestRequestStates:
    def test_a_model_without_a_readable_location_is_unavailable(self, env):
        item = env.add("a.safetensors", b"a" * 50)
        env.library.set_status(item, "missing")

        assert env.manager.request_digest(item.model.id) == {"state": "unavailable", "eta_seconds": None}
        assert env.queue.get(item.location_id)["priority"] == PRIORITY_BACKGROUND
        assert env.manager._demand_threads == {}

    def test_an_unknown_model_is_missing_and_a_digested_one_is_ready(self, env):
        content = b"a" * 50
        item = env.add("a.safetensors", content, sha256=_sha(content))

        assert env.manager.request_digest("nope") == {"state": "missing", "eta_seconds": None}
        assert env.manager.request_digest(item.model.id) == {"state": "ready", "eta_seconds": 0}


def _live_workers():
    return len([t for t in threading.enumerate() if t.name.startswith("model-digest-") and t.is_alive()])

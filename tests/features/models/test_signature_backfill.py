import os

import pytest

from src.features.models import digests as digests_module
from src.features.models.file_digest import quick_signature
from tests.features.models.test_digest_manager import Env, _sha


@pytest.fixture
def env(tmp_path, threadsafe_db, monkeypatch):
    monkeypatch.setattr(digests_module, "INLINE_HASH_LIMIT", 10)
    return Env(tmp_path)


def _legacy(env, name, content):
    return env.add(name, content, sha256=_sha(content))


def _sig(env, item):
    return env.locations.get_by_id(item.location_id)["quick_sig"]


def _drain(env):
    count = 0
    while env.manager._backfill_next_signature():
        count += 1
    return count


class TestSignatureBackfill:
    def test_a_legacy_location_gets_its_signature_without_a_new_digest(self, env):
        item = _legacy(env, "a.safetensors", b"a" * 200)

        assert _drain(env) == 1

        assert _sig(env, item) == quick_signature(str(item.path), 200)
        assert env.locations.get_by_id(item.location_id)["sha256"] == _sha(b"a" * 200)

    def test_every_legacy_location_is_signed_and_the_next_call_finds_nothing(self, env):
        items = [_legacy(env, f"{n}.safetensors", bytes([n]) * (100 + n)) for n in range(1, 4)]

        assert _drain(env) == 3

        assert all(_sig(env, item) for item in items)
        assert env.manager._backfill_next_signature() is False

    def test_small_files_and_already_signed_locations_are_left_alone(self, env):
        small = _legacy(env, "small.safetensors", b"s" * 5)
        signed = _legacy(env, "signed.safetensors", b"x" * 200)
        env.library.set_digest(signed, _sha(b"x" * 200), "kept")

        assert _drain(env) == 0

        assert _sig(env, small) is None
        assert _sig(env, signed) == "kept"

    def test_nothing_is_read_while_a_generation_runs_and_work_resumes_after(self, env, monkeypatch):
        item = _legacy(env, "a.safetensors", b"a" * 200)
        reads = []
        real = digests_module.quick_signature
        monkeypatch.setattr(digests_module, "quick_signature", lambda *a: reads.append(1) or real(*a))
        env.generation_running[0] = True

        assert env.manager._backfill_next_signature() is False
        assert (reads, _sig(env, item)) == ([], None)

        env.generation_running[0] = False
        assert env.manager._backfill_next_signature() is True
        assert _sig(env, item) is not None

    def test_nothing_is_read_while_indexing_is_paused(self, env):
        item = _legacy(env, "a.safetensors", b"a" * 200)
        env.manager.pause()

        assert env.manager._backfill_next_signature() is False
        assert _sig(env, item) is None

    def test_a_file_that_changed_since_it_was_indexed_is_not_signed(self, env):
        item = _legacy(env, "a.safetensors", b"a" * 200)
        stat = item.path.stat()
        os.utime(item.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

        assert _drain(env) == 1

        assert _sig(env, item) is None
        assert env.manager._backfill_next_signature() is False

    def test_a_location_on_an_offline_root_is_skipped_without_stopping_the_rest(self, env):
        offline = _legacy(env, "a.safetensors", b"a" * 200)
        online = _legacy(env, "b.safetensors", b"b" * 300)
        env.hidden.add("a.safetensors")

        assert _drain(env) == 2

        assert (_sig(env, offline), _sig(env, online) is not None) == (None, True)

    def test_reads_are_paced_by_the_throttle(self, env):
        env.governor.throttle.configure(6_000_000, True, False)
        _legacy(env, "a.safetensors", b"a" * 200)
        _legacy(env, "b.safetensors", b"b" * 300)

        _drain(env)

        assert env.clock.sleeps

    def test_a_job_in_flight_holds_the_backfill_back(self, env):
        item = _legacy(env, "a.safetensors", b"a" * 200)
        env.manager._inflight["busy"] = digests_module._Job(
            key="busy", kind="location", path=None, size=0, mtime_ns=None, priority=0, filename="x"
        )

        assert env.manager._backfill_next_signature() is False
        assert _sig(env, item) is None

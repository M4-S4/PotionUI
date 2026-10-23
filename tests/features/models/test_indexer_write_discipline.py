import pytest

from src.features.models import indexer as indexer_module
from tests.features.models.rows_first_support import SMALL_LIMIT, Harness, _blob

GUARDED = (
    "_commit_indexed_file",
    "_commit_pending_file",
    "_recompute_availability_flags",
    "_recompute_types",
    "_retype",
)


@pytest.fixture
def h(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    return Harness(tmp_path)


@pytest.fixture
def held(h, monkeypatch):
    seen = {name: [] for name in GUARDED}
    for name in GUARDED:
        real = getattr(h.scanner, name)

        def recording(*args, _name=name, _real=real, **kwargs):
            seen[_name].append(h.scanner.write_lock.locked())
            return _real(*args, **kwargs)

        monkeypatch.setattr(h.scanner, name, recording)
    return seen


def _assert_every_call_held_the_lock(seen, *names):
    for name in names:
        assert seen[name], f"{name} was never called"
        assert all(seen[name]), f"{name} ran without the write lock"


class TestTheScannerWritesUnderTheSharedLock:
    def test_an_inline_hashed_file_is_committed_under_the_lock(self, h, held):
        h.put("home", "a.safetensors", b"a" * 5)

        h.index()

        _assert_every_call_held_the_lock(held, "_commit_indexed_file", "_recompute_availability_flags")

    def test_a_file_without_a_digest_is_committed_under_the_lock(self, h, held):
        h.put("home", "a.safetensors", _blob(1))

        h.index()

        _assert_every_call_held_the_lock(held, "_commit_pending_file", "_recompute_availability_flags")

    def test_a_contested_file_is_committed_under_the_lock_once_its_digest_arrives(self, h, held):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.put("lib", "b.safetensors", content)
        h.index()
        held["_commit_indexed_file"].clear()

        h.settle()

        _assert_every_call_held_the_lock(held, "_commit_indexed_file")

    def test_a_retype_after_the_copies_were_reconciled_holds_the_lock_too(self, h, held, monkeypatch):
        monkeypatch.setattr(h.scanner, "_reconcile_disagreeing_copies", lambda *args, **kwargs: True)
        h.put("home", "a.safetensors", b"a" * 5)

        h.index()

        assert len(held["_retype"]) == 2
        _assert_every_call_held_the_lock(held, "_retype")

    def test_settling_digested_models_retypes_and_recomputes_under_the_lock(self, h, held):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        for name in GUARDED:
            held[name].clear()

        h.scanner.settle_digested({h.model("a.safetensors").id})

        _assert_every_call_held_the_lock(held, "_recompute_types", "_recompute_availability_flags")

    def test_the_end_of_a_pass_retypes_and_recomputes_under_the_lock(self, h, held):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        for name in GUARDED:
            held[name].clear()
        path.unlink()

        h.index()

        _assert_every_call_held_the_lock(held, "_recompute_types", "_recompute_availability_flags")

    def test_applying_a_digest_holds_the_same_lock_the_scanner_exposes(self, h, monkeypatch):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        seen = []
        real = h.manager._locations.apply_digest_and_dequeue

        def recording(*args, **kwargs):
            seen.append(h.scanner.write_lock.locked())
            return real(*args, **kwargs)

        monkeypatch.setattr(h.manager._locations, "apply_digest_and_dequeue", recording)

        h.settle()

        assert seen == [True]


class TestTheAvailabilityRecomputeOnlyLoadsWhatItTouched:
    def _spy(self, h, monkeypatch):
        calls = []
        real = h.scanner.locations.winners_by_model

        def spy(root_ids, model_ids=None):
            calls.append(None if model_ids is None else set(model_ids))
            return real(root_ids, model_ids)

        monkeypatch.setattr(h.scanner.locations, "winners_by_model", spy)
        return calls

    def test_committing_one_new_file_never_loads_every_location(self, h, monkeypatch):
        for index in range(5):
            h.put("home", f"old{index}.safetensors", _blob(index))
        h.index()
        h.settle()
        calls = self._spy(h, monkeypatch)
        h.put("lib", "new.safetensors", _blob(99, 300))

        h.index()

        new_id = h.model("new.safetensors").id
        assert calls and None not in calls
        assert any(ids == {new_id} for ids in calls)

    def test_each_per_file_commit_is_scoped_to_its_own_model(self, h, monkeypatch):
        for index in range(4):
            h.put("home", f"f{index}.safetensors", _blob(index, 200 + index))
        calls = self._spy(h, monkeypatch)

        h.index()

        per_file = [ids for ids in calls if len(ids) == 1]
        assert len(per_file) >= 4
        assert None not in calls

    def test_the_repository_returns_only_the_requested_models(self, h):
        for index in range(3):
            h.put("home", f"f{index}.safetensors", _blob(index, 200 + index))
        h.index()
        wanted = h.model("f1.safetensors")
        roots = list(h.scanner.resolver.online_root_ids())

        scoped = h.scanner.locations.winners_by_model(roots, {wanted.id})
        everything = h.scanner.locations.winners_by_model(roots)

        assert set(scoped) == {wanted.id}
        assert len(everything) == 3
        assert scoped[wanted.id]["id"] == everything[wanted.id]["id"]

    def test_an_empty_scope_loads_nothing_and_a_large_scope_is_chunked(self, h):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        roots = list(h.scanner.resolver.online_root_ids())
        locations = h.scanner.locations
        wanted = [h.model("a.safetensors").id] + [f"missing{n}" for n in range(1500)]

        assert locations.winners_by_model(roots, set()) == {}
        assert set(locations.winners_by_model(roots, wanted)) == {h.model("a.safetensors").id}

import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.features.models import digests as digests_module
from src.features.models import indexer as indexer_module
from src.features.models import twin_gate as twin_gate_module
from src.features.models.digest_queue_repository import (
    MAX_ATTEMPTS,
    PRIORITY_BACKGROUND,
    PRIORITY_CHOSEN,
    PRIORITY_TWIN,
    DigestQueueRepository,
)
from src.features.models.digests import DigestManager
from src.features.models.file_digest import quick_signature
from src.features.models.hash_cache_repository import model_hash_cache_repo
from src.features.models.indexer import FoundFile, ModelScanner
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.repository import model_repo
from src.features.models.read_throttle import ReadThrottle
from src.platform.runtime.model_headers import HeaderStatus
from src.platform.settings.repository import SettingRepository
from tests.features.models.fakes import FakeClock, drain_manager
from tests.features.models.rows_first_support import BLOB, SMALL_LIMIT, Harness, _blob, _unsampled_offset
from tests.features.models.test_indexer_roots import _models, _sha, _write
from tests.fixtures.model_index_fixtures import FLUX, LORA, Library


@pytest.fixture
def h(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    return Harness(tmp_path)


class TestRowsFirst:
    def test_a_large_file_becomes_a_usable_row_without_being_read(self, h, monkeypatch):
        h.put("home", "a.safetensors", _blob(1))
        reads = []
        monkeypatch.setattr(indexer_module, "read_chunks", lambda *args, **kwargs: reads.append(1) or iter(()))

        result = h.index()

        model = h.model("a.safetensors")
        assert (model.sha256, model.file_size, model.model_type, model.is_available) == (None, BLOB, "lora", True)
        location = h.locations(model)[0]
        assert (location["status"], location["sha256"], location["size"]) == ("present", None, BLOB)
        assert reads == []
        assert (result["indexed"], result["failed"], result["contested"]) == (1, 0, 0)

    def test_the_pass_queues_every_pending_location_in_the_background(self, h):
        h.put("home", "a.safetensors", _blob(1))
        h.put("lib", "b.safetensors", _blob(2, 300))

        h.index()

        assert h.queued() == [
            ("a.safetensors", "r_home", PRIORITY_BACKGROUND, "scan"),
            ("b.safetensors", "r_lib", PRIORITY_BACKGROUND, "scan"),
        ]

    def test_the_digest_arrives_later_with_the_signature_and_the_models_row_follows(self, h):
        content = _blob(1)
        path = h.put("home", "a.safetensors", content)
        h.index()

        assert h.settle() == 1

        model = h.model("a.safetensors")
        assert model.sha256 == _sha(content)
        location = h.scanner.locations.get_by_id(h.locations(model)[0]["id"])
        assert (location["sha256"], location["quick_sig"]) == (_sha(content), quick_signature(str(path)))
        assert h.queued() == []
        assert h.applied == [{model.id}]

    def test_small_files_are_still_hashed_inline(self, h):
        content = b"tiny"
        h.put("home", "tiny.safetensors", content)

        h.index()

        assert h.model("tiny.safetensors").sha256 == _sha(content)
        assert h.queued() == []

    def test_a_large_file_with_a_cached_digest_is_ready_at_once(self, h):
        content = _blob(1)
        path = h.put("home", "a.safetensors", content)
        stat = path.stat()
        model_hash_cache_repo.put(str(path), stat.st_size, stat.st_mtime_ns, _sha(content))

        h.index()

        assert h.model("a.safetensors").sha256 == _sha(content)
        assert h.queued() == []

    def test_a_directory_model_is_fingerprinted_inline(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        harness = Harness(tmp_path, with_llm=True)
        folder = harness.home / "llm" / "mymodel"
        _write(folder / "config.json", b"{}")
        _write(folder / "model-1.safetensors", _blob(1))

        harness.index()

        model = harness.model("mymodel")
        assert model.is_directory is True and model.sha256
        assert harness.queued() == []

    def test_an_unchanged_pending_file_is_neither_read_nor_queued_twice(self, h, monkeypatch):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        before = h.queued()
        reads = []
        monkeypatch.setattr(indexer_module, "read_chunks", lambda *args, **kwargs: reads.append(1) or iter(()))
        monkeypatch.setattr(twin_gate_module.SignatureCache, "of", lambda self, path, size=None: reads.append(1))

        second = h.index()

        assert (second["indexed"], second["skipped"]) == (0, 1)
        assert reads == []
        assert h.queued() == before

    def test_a_pending_file_that_changes_keeps_its_row_and_stays_pending(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        first = h.model("a.safetensors")
        path.write_bytes(_blob(2, 300))
        os.utime(path, ns=(1_000_000_000, 2_000_000_000))

        h.index()

        again = h.model("a.safetensors")
        assert again.id == first.id
        assert (again.sha256, again.file_size) == (None, 300)
        location = h.locations(again)[0]
        assert (location["size"], location["sha256"]) == (300, None)
        assert h.queued() == [("a.safetensors", "r_home", PRIORITY_BACKGROUND, "scan")]

    def test_a_ready_file_that_changes_goes_back_to_pending_and_is_hashed_again(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        assert h.model("a.safetensors").sha256 == _sha(_blob(1))
        replaced = _blob(2, 300)
        path.write_bytes(replaced)
        os.utime(path, ns=(1_000_000_000, 2_000_000_000))

        h.index()

        model = h.model("a.safetensors")
        assert (model.sha256, model.file_size) == (None, 300)
        assert h.locations(model)[0]["sha256"] is None
        h.settle()
        assert h.model("a.safetensors").sha256 == _sha(replaced)

    def test_a_changed_copy_keeps_the_models_digest_while_another_copy_still_holds_it(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        copy = h.put("lib", "a.safetensors", content)
        h.index()
        h.settle(max_jobs=60)
        assert h.model("a.safetensors").sha256 == _sha(content)
        copy.write_bytes(_blob(2, 300))
        os.utime(copy, ns=(1_000_000_000, 2_000_000_000))

        h.index()

        model = h.model("a.safetensors")
        assert model.sha256 == _sha(content)
        by_root = {row["root_id"]: row for row in h.locations(model)}
        assert by_root["r_lib"]["sha256"] is None

    def test_a_file_with_no_same_size_twin_is_never_signed(self, h, monkeypatch):
        h.put("home", "a.safetensors", _blob(1))
        h.put("lib", "b.safetensors", _blob(2, 300))
        signed = []
        monkeypatch.setattr(twin_gate_module.SignatureCache, "of", lambda self, path, size=None: signed.append(path))

        h.index()

        assert signed == []

    def test_a_cancelled_pass_queues_nothing(self, h):
        h.put("home", "a.safetensors", _blob(1))

        h.scanner.index_models(max_workers=1, cancel_check=lambda: True)

        assert h.queued() == []


class TestSizeAndIdentity:
    def test_a_same_name_file_of_another_size_is_a_conflict_at_once(self, h):
        h.put("home", "x.safetensors", _blob(1))
        h.put("lib", "x.safetensors", _blob(2, 300))

        h.index()

        models = _models()
        assert len(models) == 1
        by_status = sorted((row["status"], row["sha256"]) for row in h.scanner.locations.list_for_model(models[0].id))
        assert by_status == [("conflict", None), ("present", None)]
        assert len(h.queued()) == 1

    def test_two_new_same_size_files_with_different_content_and_the_same_name_conflict_without_a_read(self, h, monkeypatch):
        h.put("home", "x.safetensors", _blob(1))
        h.put("lib", "x.safetensors", _blob(2))
        reads = []
        monkeypatch.setattr(indexer_module, "read_chunks", lambda *args, **kwargs: reads.append(1) or iter(()))

        result = h.index()

        assert result["contested"] == 0 and reads == []
        models = _models()
        assert len(models) == 1
        statuses = sorted(row["status"] for row in h.scanner.locations.list_for_model(models[0].id))
        assert statuses == ["conflict", "present"]

    def test_same_size_files_with_different_names_and_content_are_two_models(self, h):
        h.put("home", "a.safetensors", _blob(1))
        h.put("lib", "b.safetensors", _blob(2))

        result = h.index()

        assert sorted(m.filename for m in _models()) == ["a.safetensors", "b.safetensors"]
        assert result["contested"] == 0
        assert len(h.queued()) == 2


class TestTwinGate:
    def test_identical_files_under_different_names_are_compared_before_a_second_row_exists(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.put("lib", "b.safetensors", content)

        result = h.index()

        assert result["contested"] == 2 and _models() == []
        assert h.manager.status()["duplicate_checks"] == 2

        h.settle()

        assert [m.filename for m in _models()] == ["a.safetensors"]
        outcomes = [(found.abs_path.endswith("a.safetensors"), outcome.model is not None, outcome.duplicate_of is not None)
                    for found, outcome, error in h.contested]
        assert sorted(outcomes) == [(False, False, True), (True, True, False)]
        assert h.manager.status()["duplicate_checks"] == 0

    def test_identical_files_with_the_same_name_become_one_model_with_two_locations(self, h):
        content = _blob(1)
        h.put("home", "x.safetensors", content)
        h.put("lib", "x.safetensors", content)

        h.index()
        h.settle()

        models = _models()
        assert len(models) == 1 and models[0].sha256 == _sha(content)
        locations = h.scanner.locations.list_for_model(models[0].id)
        assert sorted((row["root_id"], row["status"]) for row in locations) == [("r_home", "present"), ("r_lib", "present")]
        signatures = {
            row["root_id"]: h.scanner.locations.get_by_id(row["id"])["quick_sig"] for row in locations
        }
        assert signatures == {
            "r_home": quick_signature(str(h.home / "loras" / "x.safetensors")),
            "r_lib": quick_signature(str(h.lib / "loras" / "x.safetensors")),
        }

    def test_a_twin_whose_signature_cannot_be_read_but_whose_digest_is_known_still_contests_the_new_file(self, h, monkeypatch):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        from src.platform.database import get_cursor

        with get_cursor() as cursor:
            cursor.execute("UPDATE model_locations SET quick_sig = NULL")
        real_location_path = h.scanner.location_path
        monkeypatch.setattr(h.scanner, "location_path", lambda row: None)
        h.put("lib", "b.safetensors", content)

        result = h.index()

        assert result["contested"] == 1
        monkeypatch.setattr(h.scanner, "location_path", real_location_path)
        h.settle()
        assert [m.filename for m in _models()] == ["a.safetensors"]

    def test_files_that_look_equal_to_the_signature_but_differ_are_told_apart_by_the_full_digest(self, h, monkeypatch):
        size = 4 * 1024 * 1024
        first = bytearray(_blob(1, size))
        second = bytearray(first)
        second[_unsampled_offset(size)] ^= 0xFF
        h.put("home", "x.safetensors", bytes(first))
        h.put("lib", "x.safetensors", bytes(second))

        result = h.index()

        assert result["contested"] == 2
        h.settle()

        models = _models()
        assert len(models) == 1
        statuses = sorted(row["status"] for row in h.scanner.locations.list_for_model(models[0].id))
        assert statuses == ["conflict", "present"]

    def test_a_pending_file_that_changes_is_never_its_own_twin(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        path.write_bytes(_blob(1))
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        result = h.index()

        assert (result["contested"], result["failed"]) == (0, 0)
        assert [m.filename for m in _models()] == ["a.safetensors"]
        assert [row["status"] for row in h.locations(h.model("a.safetensors"))] == ["present"]

    def test_a_ready_file_that_is_only_touched_is_never_its_own_twin(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        result = h.index()

        assert (result["contested"], result["failed"]) == (0, 0)
        assert h.manager.status()["duplicate_checks"] == 0
        assert [m.filename for m in _models()] == ["a.safetensors"]

    def test_a_file_that_comes_back_to_its_own_path_is_never_its_own_missing_twin(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        content = path.read_bytes()
        path.unlink()
        h.index()
        from src.platform.database import get_cursor

        with get_cursor() as cursor:
            cursor.execute("DELETE FROM model_hash_cache")
        path.write_bytes(content)

        result = h.index()

        assert (result["contested"], result["failed"]) == (0, 0)
        assert [m.filename for m in _models()] == ["a.safetensors"]

    def test_the_other_copy_of_the_same_model_is_still_a_twin(self, h):
        content = _blob(1)
        h.put("home", "x.safetensors", content)
        h.index()
        h.settle()
        h.put("lib", "x.safetensors", content)

        result = h.index()

        assert result["contested"] == 1

    def test_a_file_identical_to_a_pending_one_waits_for_that_ones_digest_before_it_is_decided(self, h):
        content = _blob(1)
        h.put("home", "x.safetensors", content)
        h.index()
        assert h.model("x.safetensors").sha256 is None
        h.put("lib", "x.safetensors", content)

        result = h.index()

        assert result["contested"] == 1
        assert ("x.safetensors", "r_home", PRIORITY_TWIN, "twin") in h.queued()

        h.settle()

        models = _models()
        assert len(models) == 1 and models[0].sha256 == _sha(content)
        locations = h.scanner.locations.list_for_model(models[0].id)
        assert sorted((row["root_id"], row["status"]) for row in locations) == [("r_home", "present"), ("r_lib", "present")]

    def test_a_file_identical_to_a_ready_one_needs_only_its_own_digest(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        h.put("lib", "b.safetensors", content)

        result = h.index()

        assert result["contested"] == 1
        h.settle()

        assert [m.filename for m in _models()] == ["a.safetensors"]
        ((found, outcome, error),) = h.contested
        assert error is None and outcome.model is None and outcome.duplicate_of is not None

    def test_a_contested_file_that_cannot_be_read_is_reported_and_gets_no_row(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        gone = h.put("lib", "b.safetensors", content)
        h.index()
        gone.unlink()

        h.settle(max_jobs=10)

        assert [m.filename for m in _models()] == ["a.safetensors"]
        ((found, outcome, error),) = h.contested
        assert outcome is None and error

    def test_a_twin_in_an_offline_root_that_cannot_be_compared_does_not_hold_a_new_file_back(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        harness = Harness(tmp_path)
        content = _blob(1)
        harness.put("home", "a.safetensors", content)
        harness.index()
        monkeypatch.setattr(harness.scanner, "location_path", lambda row: None)
        harness.put("lib", "b.safetensors", content)

        result = harness.index()

        assert result["contested"] == 0
        assert sorted(m.filename for m in _models()) == ["a.safetensors", "b.safetensors"]


def _tag(model_id, name="style"):
    from src.platform.database import get_cursor
    from src.platform.util.ids import generate_ulid

    tag_id = generate_ulid()
    with get_cursor() as cursor:
        cursor.execute("INSERT INTO tags (id, name) VALUES (?, ?)", (tag_id, name))
        cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, ?)", (model_id, tag_id))
    return tag_id


def _tag_ids(model_id):
    from src.platform.database import get_cursor

    with get_cursor() as cursor:
        cursor.execute("SELECT tag_id FROM model_tags WHERE model_id = ?", (model_id,))
        return {row["tag_id"] for row in cursor.fetchall()}


def _statuses(h, model):
    return sorted((row["root_id"], row["model_type"], row["rel_path"], row["status"]) for row in h.locations(model))


class TestFailedFilesAcrossPasses:
    def _fail(self, h, path):
        model = h.model(path.name)
        location_id = h.locations(model)[0]["id"]
        for _ in range(MAX_ATTEMPTS):
            h.queue.record_failure(location_id, "boom", 0.0, ())
        return location_id

    def test_a_file_that_failed_out_stays_failed_while_it_is_unchanged(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        location_id = self._fail(h, path)

        h.index()
        h.index()

        assert h.queue.get(location_id)["attempts"] == MAX_ATTEMPTS

    def test_a_file_that_changes_gets_a_fresh_chance_but_another_failed_file_does_not(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        other = h.put("lib", "b.safetensors", _blob(2, 300))
        h.index()
        other_id = self._fail(h, other)
        location_id = self._fail(h, path)
        path.write_bytes(_blob(2))
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        h.index()

        row = h.queue.get(location_id)
        assert (row["attempts"], row["last_error"]) == (0, None)
        assert h.queue.get(other_id)["attempts"] == MAX_ATTEMPTS


class TestPassEndReconcile:
    def test_a_pending_file_deleted_before_its_digest_loses_its_queue_row_and_frees_the_file_waiting_for_it(self, h):
        content = _blob(1)
        gone = h.put("home", "a.safetensors", content)
        h.index()
        h.put("lib", "b.safetensors", content)
        h.index()
        assert ("a.safetensors", "r_home", PRIORITY_TWIN, "twin") in h.queued()

        h.settle(max_jobs=1)

        assert h.manager.status()["duplicate_checks"] == 1
        assert _models()[0].filename == "a.safetensors" and len(_models()) == 1
        gone.unlink()

        h.index()

        assert [entry for entry in h.queued() if entry[0] == "a.safetensors"] == []
        assert h.manager.status()["duplicate_checks"] == 0
        by_name = {m.filename: m for m in _models()}
        assert sorted(by_name) == ["a.safetensors", "b.safetensors"]
        assert by_name["b.safetensors"].sha256 == _sha(content)
        assert h.model("a.safetensors").is_available is False


class TestMovedFiles:
    @pytest.fixture
    def settled(self, h):
        content = _blob(1)
        path = h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        model = h.model("a.safetensors")
        tag = _tag(model.id)
        return h, path, model, tag, content

    def test_a_renamed_large_file_stays_the_same_model_with_its_tags(self, settled, caplog):
        h, path, model, tag, content = settled
        path.rename(path.with_name("b.safetensors"))

        result = h.index()

        assert result["contested"] == 1
        with caplog.at_level("WARNING", logger="src.features.models.digests"):
            h.settle()

        assert [m.filename for m in _models()] == ["b.safetensors"]
        survivor = h.model("b.safetensors")
        assert (survivor.id, survivor.sha256, survivor.is_available) == (model.id, _sha(content), True)
        assert _tag_ids(survivor.id) == {tag}
        assert _statuses(h, survivor) == [
            ("r_home", "lora", "a.safetensors", "missing"),
            ("r_home", "lora", "b.safetensors", "present"),
        ]
        assert "has the same content as" not in caplog.text

    def test_a_large_file_moved_to_another_root_keeps_its_model_and_tags(self, settled):
        h, path, model, tag, content = settled
        target = h.lib / "loras" / "a.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)

        result = h.index()

        assert result["contested"] == 1
        h.settle()

        assert [m.filename for m in _models()] == ["a.safetensors"]
        survivor = h.model("a.safetensors")
        assert (survivor.id, survivor.sha256, survivor.is_available) == (model.id, _sha(content), True)
        assert _tag_ids(survivor.id) == {tag}
        present = [row for row in h.locations(survivor) if row["status"] == "present"]
        assert [row["root_id"] for row in present] == ["r_lib"]

    def test_a_large_file_moved_into_a_subfolder_keeps_its_model_and_tags(self, settled):
        h, path, model, tag, content = settled
        target = path.parent / "sub" / "a.safetensors"
        target.parent.mkdir()
        path.rename(target)

        result = h.index()

        assert result["contested"] == 1
        h.settle()

        assert [m.filename for m in _models()] == ["a.safetensors"]
        survivor = h.model("a.safetensors")
        assert (survivor.id, survivor.sha256, survivor.is_available) == (model.id, _sha(content), True)
        assert _tag_ids(survivor.id) == {tag}
        assert sorted((row["rel_path"], row["status"]) for row in h.locations(survivor)) == [
            ("a.safetensors", "missing"), ("sub/a.safetensors", "present"),
        ]

    def test_a_large_file_moved_into_another_type_folder_keeps_its_model_and_tags(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        harness = Harness(tmp_path, extra_types=("vae",))
        content = _blob(1)
        old = harness.put("home", "a.safetensors", content)
        harness.index()
        harness.settle()
        model = harness.model("a.safetensors")
        tag = _tag(model.id)
        target = harness.home / "vae" / "a.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        old.rename(target)

        result = harness.index()

        assert result["contested"] == 1
        harness.settle()

        assert len(_models()) == 1
        survivor = harness.model("a.safetensors")
        assert (survivor.id, survivor.sha256, survivor.is_available) == (model.id, _sha(content), True)
        assert _tag_ids(survivor.id) == {tag}
        present = [row for row in harness.locations(survivor) if row["status"] == "present"]
        assert [row["model_type"] for row in present] == ["vae"]

    def test_a_renamed_file_is_hashed_once_and_not_again_on_the_next_pass(self, settled, monkeypatch):
        h, path, model, tag, content = settled
        path.rename(path.with_name("b.safetensors"))
        h.index()
        h.settle()
        reads = []
        real = indexer_module.read_chunks
        monkeypatch.setattr(indexer_module, "read_chunks", lambda *a, **k: reads.append(1) or real(*a, **k))

        result = h.index()

        assert (result["contested"], result["new_files"], reads) == (0, 0, [])

    def test_a_renamed_file_with_a_different_signature_than_the_missing_copy_is_a_new_model(self, settled):
        h, path, model, tag, content = settled
        path.unlink()
        h.put("home", "b.safetensors", _blob(2))

        result = h.index()

        assert result["contested"] == 0
        assert sorted(m.filename for m in _models()) == ["a.safetensors", "b.safetensors"]
        assert h.model("a.safetensors").is_available is False
        assert _tag_ids(h.model("b.safetensors").id) == set()

    def test_a_missing_copy_that_was_never_hashed_does_not_hold_a_new_file_back(self, h):
        old = h.put("home", "a.safetensors", _blob(1))
        h.index()
        old.rename(old.with_name("b.safetensors"))

        result = h.index()

        assert result["contested"] == 0
        assert sorted(m.filename for m in _models()) == ["a.safetensors", "b.safetensors"]


class TestAskingAboutAModelThatWasFoldedAway:
    @pytest.fixture
    def folded(self, h, monkeypatch):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        owner = h.model("a.safetensors")
        monkeypatch.setattr(h.scanner, "_size_twins", lambda *args, **kwargs: [])
        path.rename(path.with_name("b.safetensors"))
        h.index()
        return h, owner, h.model("b.safetensors")

    def test_asking_for_the_digest_while_the_file_is_hashed_returns_the_owners_digest(self, folded):
        h, owner, fresh = folded

        sha = h.manager.ensure_digest(fresh.id)

        assert sha == owner.sha256
        assert model_repo.get_by_id(fresh.id, include_providers=False) is None
        assert h.model("b.safetensors").id == owner.id

    def test_asking_again_after_the_join_answers_with_the_owner(self, folded):
        h, owner, fresh = folded
        h.settle()

        assert h.manager.ensure_digest(fresh.id) == owner.sha256
        assert h.manager.request_digest(fresh.id) == {"state": "ready", "eta_seconds": 0, "merged_into": owner.id}

    def test_a_model_that_never_existed_is_still_missing(self, folded):
        h, owner, fresh = folded
        h.settle()

        assert h.manager.ensure_digest("nope") is None
        assert h.manager.request_digest("nope") == {"state": "missing", "eta_seconds": None}

    def test_only_the_most_recent_joins_are_remembered(self, folded, monkeypatch):
        h, owner, fresh = folded
        monkeypatch.setattr(digests_module, "MAX_REMEMBERED_MERGES", 2)
        for index in range(3):
            h.manager._note_merge(SimpleNamespace(merged_from=f"gone{index}", model_id=owner.id, model_filename="b"))

        assert h.manager.request_digest("gone0")["state"] == "missing"
        assert h.manager.request_digest("gone2")["state"] == "ready"

    def test_a_join_whose_owner_has_since_been_deleted_has_no_digest_to_give(self, folded):
        h, owner, fresh = folded
        h.settle()
        model_repo.delete(owner.id)

        assert h.manager.ensure_digest(fresh.id) is None


class TestADigestLandingOnAnOrphanedModel:
    @pytest.fixture
    def orphaned(self, h, monkeypatch):
        content = _blob(1)
        path = h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        owner = h.model("a.safetensors")
        tag = _tag(owner.id)
        monkeypatch.setattr(h.scanner, "_size_twins", lambda *args, **kwargs: [])
        path.rename(path.with_name("b.safetensors"))
        result = h.index()
        assert result["contested"] == 0
        assert sorted(m.filename for m in _models()) == ["a.safetensors", "b.safetensors"]
        return h, owner, tag, content

    def test_the_digest_joins_the_new_file_to_the_original_model_instead_of_logging_a_duplicate(self, orphaned, caplog):
        h, owner, tag, content = orphaned

        with caplog.at_level("INFO", logger="src.features.models.digests"):
            h.settle()

        assert [m.filename for m in _models()] == ["b.safetensors"]
        survivor = h.model("b.safetensors")
        assert (survivor.id, survivor.sha256, survivor.is_available) == (owner.id, _sha(content), True)
        assert _tag_ids(survivor.id) == {tag}
        assert "has the same content as" not in caplog.text and "was joined to that model" in caplog.text
        assert h.applied[-1] == {owner.id}
        assert [row["status"] for row in h.locations(survivor) if row["rel_path"] == "b.safetensors"] == ["present"]

    def test_a_filename_already_taken_in_the_models_type_blocks_the_join(self, tmp_path, mock_db, monkeypatch, caplog):
        from src.features.models.records import Model

        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        h = Harness(tmp_path, extra_types=("vae",))
        old = h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        taken = model_repo.create(Model(filename="b.safetensors", file_size=1, sha256="f" * 64, model_type="lora"))
        monkeypatch.setattr(h.scanner, "_size_twins", lambda *args, **kwargs: [])
        target = h.home / "vae" / "b.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        old.rename(target)
        h.index()
        before = sorted((m.filename, m.model_type) for m in _models())
        assert before == [("a.safetensors", "lora"), ("b.safetensors", "lora"), ("b.safetensors", "vae")]

        with caplog.at_level("WARNING", logger="src.features.models.digests"):
            h.settle()

        assert sorted((m.filename, m.model_type) for m in _models()) == before
        assert model_repo.get_by_id(taken.id, include_providers=False).sha256 == "f" * 64
        assert caplog.text.count("has the same content as") == 1


class TestConflictsAcrossPasses:
    def test_a_same_name_file_of_another_size_stays_a_conflict_and_the_model_keeps_its_data(self, h):
        first = _blob(1)
        h.put("home", "x.safetensors", first)
        h.index()
        h.settle()
        h.put("lib", "x.safetensors", _blob(2, 300))
        h.index()
        model = h.model("x.safetensors")
        assert (model.sha256, model.file_size) == (_sha(first), BLOB)

        second_pass = h.index()
        h.settle()

        model = h.model("x.safetensors")
        assert (model.sha256, model.file_size) == (_sha(first), BLOB)
        by_root = {row["root_id"]: (row["status"], row["sha256"], row["size"]) for row in h.locations(model)}
        assert by_root == {"r_home": ("present", _sha(first), BLOB), "r_lib": ("conflict", None, 300)}
        assert h.queued() == []
        assert second_pass["failed"] == 0

    def test_an_inline_hashed_conflict_is_just_as_durable(self, h):
        h.put("home", "x.safetensors", b"a" * 5)
        h.put("lib", "x.safetensors", b"b" * 6)
        h.index()
        model = h.model("x.safetensors")
        before = (model.sha256, model.file_size)

        h.index()

        model = h.model("x.safetensors")
        assert (model.sha256, model.file_size) == before
        assert sorted(row["status"] for row in h.locations(model)) == ["conflict", "present"]


class TestContestedDigestsAreCached:
    def test_a_file_decided_as_a_duplicate_is_not_read_again_by_the_next_pass(self, h, monkeypatch):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        h.put("lib", "b.safetensors", content)
        h.index()
        h.settle()
        assert [m.filename for m in _models()] == ["a.safetensors"]
        reads = []
        real = indexer_module.read_chunks
        monkeypatch.setattr(indexer_module, "read_chunks", lambda *a, **k: reads.append(1) or real(*a, **k))
        from src.features.models import file_digest as file_digest_module

        real_digest = file_digest_module.digest_file
        digest_reads = []
        monkeypatch.setattr(
            "src.features.models.digests.digest_file", lambda *a, **k: digest_reads.append(a[0]) or real_digest(*a, **k)
        )

        second = h.index()
        h.settle()

        assert second["contested"] == 0 and reads == [] and digest_reads == []
        assert [m.filename for m in _models()] == ["a.safetensors"]

    def test_the_digest_of_a_contested_file_is_stored_for_its_size_and_mtime(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        path = h.put("lib", "b.safetensors", content)
        h.index()
        h.settle()

        stat = path.stat()

        cached = model_hash_cache_repo.get(str(path))
        assert (cached.size, cached.mtime_ns, cached.sha256) == (stat.st_size, stat.st_mtime_ns, _sha(content))

    def test_a_contested_file_that_changes_while_it_is_read_is_not_cached_and_retried(self, h, monkeypatch):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        path = h.put("lib", "b.safetensors", content)
        h.index()
        from src.features.models import digests as digests_module

        real_digest = digests_module.digest_file

        def touching(*args, **kwargs):
            result = real_digest(*args, **kwargs)
            stat = os.stat(path)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return result

        monkeypatch.setattr(digests_module, "digest_file", touching)

        h.settle(max_jobs=5)

        assert model_hash_cache_repo.get(str(path)) is None
        ((found, outcome, error),) = h.contested
        assert outcome is None and error


class TestTimestampsAreUtc:
    def test_a_new_pending_model_is_stamped_with_an_aware_utc_time(self, h, monkeypatch):
        created = []
        real = model_repo.create
        monkeypatch.setattr(model_repo, "create", lambda model: created.append(model) or real(model))
        h.put("home", "a.safetensors", _blob(1))

        h.index()

        assert created[0].indexed_at.utcoffset() == timedelta(0)

    def test_a_refreshed_model_is_stamped_with_an_aware_utc_time(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        h.index()
        model = h.model("a.safetensors")
        stat = path.stat()
        found = h.scanner.found_file_for(
            h.scanner.resolver.to_logical(path), path, stat.st_size, stat.st_mtime_ns, False
        )
        model.indexed_at = None

        ModelScanner._refresh_model(model, found)

        assert model.indexed_at.utcoffset() == timedelta(0)


class TestPassEndAvailability:
    def test_reviving_through_a_stale_copy_never_rewrites_the_digest(self, h):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        model = h.model("a.safetensors")
        model_repo.mark_unavailable(model.id)
        stale = model_repo.get_by_id(model.id, include_providers=False, include_tags=False)
        stale.sha256 = None

        h.scanner._revive(stale)

        revived = h.model("a.safetensors")
        assert (revived.sha256, revived.is_available, revived.unavailable_at) == (_sha(content), True, None)
        assert (stale.is_available, stale.unavailable_at) == (True, None)

    def test_the_pass_end_recompute_runs_under_the_write_lock(self, h, monkeypatch):
        content = _blob(1)
        path = h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        path.unlink()
        held = []
        real = h.scanner._recompute_availability_flags

        def recording(model_ids):
            held.append(h.scanner.write_lock.locked())
            return real(model_ids)

        monkeypatch.setattr(h.scanner, "_recompute_availability_flags", recording)

        h.index()

        assert held and all(held)

    def test_reviving_a_model_never_rewrites_a_digest_the_pool_applied_in_between(self, h, monkeypatch):
        content = _blob(1)
        h.put("home", "a.safetensors", content)
        h.index()
        h.settle()
        model = h.model("a.safetensors")
        model_repo.mark_unavailable(model.id)
        stale = model_repo.get_by_id(model.id, include_providers=False, include_tags=False)
        stale.sha256 = None
        real_get = model_repo.get_by_id
        monkeypatch.setattr(
            model_repo, "get_by_id",
            lambda model_id, **kwargs: stale if model_id == model.id else real_get(model_id, **kwargs),
        )

        h.scanner._recompute_availability_flags({model.id})

        monkeypatch.setattr(model_repo, "get_by_id", real_get)
        revived = h.model("a.safetensors")
        assert (revived.sha256, revived.is_available) == (_sha(content), True)


class TestHeaderTypedFiles:
    @pytest.fixture
    def lib(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", 1)
        library = Library(tmp_path)
        governor = IndexingGovernor(throttle=ReadThrottle(clock=FakeClock().monotonic, sleep=FakeClock().sleep))
        library.scanner = ModelScanner(library.scanner.resolver, governor=governor)
        library.applied = []
        library.manager = DigestManager(
            DigestQueueRepository(),
            governor,
            library.scanner.location_path,
            locations_write_lock=library.scanner.write_lock,
            on_applied=lambda ids: (library.applied.append(set(ids)), library.scanner.settle_digested(set(ids))),
            setting_repository=SettingRepository(),
        )
        library.scanner.attach_digests(library.manager)
        return library

    def test_the_type_comes_from_the_header_read_during_the_scan_and_the_verdict_is_stored_with_the_digest(self, lib):
        path = lib.put(lib.checkpoints, "flux-dev.safetensors", FLUX)

        lib.index()

        model = lib.model("flux-dev.safetensors")
        assert (model.model_type, model.type_source, model.sha256) == ("diffusion_model", "header", None)
        assert lib.scanner.types.get_verdicts([_sha(path.read_bytes())]) == {}

        drain_manager(lib.manager, max_jobs=5)

        digest = _sha(path.read_bytes())
        verdict = lib.scanner.types.get_verdicts([digest])[digest]
        assert (verdict["status"], verdict["model_type"]) == ("decided", "diffusion_model")
        settled = lib.model("flux-dev.safetensors")
        assert (settled.model_type, settled.type_source, settled.sha256) == ("diffusion_model", "header", digest)
        assert lib.applied == [{settled.id}]

    def test_an_unrecognised_header_is_undefined_straight_away(self, lib):
        lib.put(lib.checkpoints, "mystery.safetensors", LORA)

        lib.index()

        model = lib.model("mystery.safetensors")
        assert (model.model_type, model.type_source, model.sha256) == ("undefined", "header", None)

    def test_an_unreadable_header_keeps_the_folder_type_and_is_reported(self, lib, monkeypatch):
        lib.put(lib.checkpoints, "broken.safetensors", FLUX)

        class Unreadable:
            status = HeaderStatus.IO_ERROR
            error = "disk fell off"
            view = None

        monkeypatch.setattr(indexer_module, "read_header", lambda *args, **kwargs: Unreadable())

        result = lib.index()

        model = lib.model("broken.safetensors")
        assert (model.model_type, model.type_source) == ("checkpoint", "folder")
        assert result["failed_files"] == [{"path": str(lib.checkpoints / "broken.safetensors"), "error": "Could not read the file header: disk fell off"}]

    def test_a_pickle_file_is_never_opened_for_its_header(self, lib, monkeypatch):
        (lib.checkpoints / "old.ckpt").write_bytes(b"pickle bytes that are long enough")
        opened = []
        monkeypatch.setattr(indexer_module, "read_header", lambda *args, **kwargs: opened.append(1))

        lib.index()

        assert opened == []
        assert lib.model("old.ckpt").type_source == "folder"


class TestSettleDigestedVerdicts:
    @pytest.fixture
    def both_unscanned(self, tmp_path, mock_db):
        library = Library(tmp_path, scan_checkpoints=False, scan_diffusion=False)
        return library

    def _drop_verdicts(self):
        from src.platform.database import get_cursor

        with get_cursor() as cursor:
            cursor.execute("DELETE FROM model_header_verdicts")

    def test_copies_filed_under_two_unscanned_types_get_their_verdict_once_the_digest_is_known(self, both_unscanned):
        library = both_unscanned
        library.put(library.checkpoints, "flux.safetensors", FLUX, tag="same")
        library.put(library.diffusion, "flux.safetensors", FLUX, tag="same")
        library.index()
        self._drop_verdicts()
        model = [m for m in _models() if m.filename == "flux.safetensors"][0]
        digest = model_repo.get_by_id(model.id, include_providers=False, include_tags=False).sha256
        assert digest and library.scanner.types.get_verdicts([digest]) == {}

        library.scanner.settle_digested({model.id})

        assert library.scanner.types.get_verdicts([digest])[digest]["model_type"] == "diffusion_model"

    def test_a_single_unscanned_copy_needs_no_verdict(self, both_unscanned):
        library = both_unscanned
        library.put(library.checkpoints, "flux.safetensors", FLUX)
        library.index()
        self._drop_verdicts()
        model = [m for m in _models() if m.filename == "flux.safetensors"][0]

        library.scanner.settle_digested({model.id})

        from src.platform.database import get_cursor

        with get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM model_header_verdicts")
            assert cursor.fetchone()["n"] == 0


class TestEnsureVerdictForLocation:
    @pytest.fixture
    def library(self, tmp_path, mock_db):
        library = Library(tmp_path, scan_checkpoints=False)
        library.path = library.put(library.checkpoints, "flux.safetensors", FLUX)
        library.index()
        model = library.model("flux.safetensors")
        library.location = library.scanner.locations.list_for_model(model.id)[0]
        library.sha = _sha(library.path.read_bytes())
        return library

    def _verdicts(self, library):
        return library.scanner.types.get_verdicts([library.sha])

    def test_a_verdict_is_made_from_the_file_even_when_its_folder_is_not_header_scanned(self, library):
        assert self._verdicts(library) == {}

        library.scanner.ensure_verdict_for_location(library.location)

        assert self._verdicts(library)[library.sha]["model_type"] == "diffusion_model"

    def test_a_location_without_a_digest_or_a_header_extension_is_left_alone(self, library):
        library.scanner.ensure_verdict_for_location({**library.location, "sha256": None})
        library.scanner.ensure_verdict_for_location({**library.location, "rel_path": "old.ckpt"})

        assert self._verdicts(library) == {}

    def test_a_location_on_an_offline_root_is_left_alone(self, library, monkeypatch):
        monkeypatch.setattr(library.scanner, "location_path", lambda row: None)

        library.scanner.ensure_verdict_for_location(library.location)

        assert self._verdicts(library) == {}

    def test_an_unreadable_header_is_a_warning_not_an_error(self, library, monkeypatch):
        class Unreadable:
            status = HeaderStatus.IO_ERROR
            error = "disk fell off"
            view = None

        monkeypatch.setattr(indexer_module, "read_header", lambda *args, **kwargs: Unreadable())

        library.scanner.ensure_verdict_for_location(library.location)

        assert self._verdicts(library) == {}


class TestSettleDigested:
    def test_models_without_a_digest_or_with_nothing_to_classify_are_left_alone(self, h):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        model = h.model("a.safetensors")

        assert h.scanner.settle_digested({model.id}) == []
        assert h.scanner.settle_digested(set()) == []
        assert h.model("a.safetensors").model_type == "lora"


class TestSkippedDuplicateEntry:
    def test_the_entry_names_the_file_its_root_and_what_it_duplicates(self, h):
        h.put("home", "a.safetensors", _blob(1))
        h.index()
        h.settle()
        model = h.model("a.safetensors")
        found = FoundFile("r_lib", "lora", "b.safetensors", str(h.lib / "loras" / "b.safetensors"), BLOB, 1)
        duplicate = indexer_module.DuplicateOf(model.id, "lora", "r_home", "/x/a.safetensors")

        entry = h.scanner.skipped_duplicate_entry(found, duplicate)

        assert entry == {
            "path": Path(found.abs_path).as_posix(),
            "root": "r_lib",
            "same_as": {"model_id": model.id, "model_type": "lora", "root": "r_home", "path": "/x/a.safetensors"},
        }


class TestSingleFileIndexing:
    def test_a_file_indexed_on_demand_is_queued_at_the_chosen_priority(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        stat = path.stat()
        found = h.scanner.found_file_for(
            h.scanner.resolver.to_logical(path), path, stat.st_size, stat.st_mtime_ns, False
        )

        outcome = h.scanner.index_file(found, queue_priority=PRIORITY_CHOSEN)

        assert outcome.model is not None and outcome.location_id is not None
        assert h.queued() == [("a.safetensors", "r_home", PRIORITY_CHOSEN, "pick")]

    def test_without_a_priority_nothing_is_queued_by_the_file_itself(self, h):
        path = h.put("home", "a.safetensors", _blob(1))
        stat = path.stat()
        found = h.scanner.found_file_for(
            h.scanner.resolver.to_logical(path), path, stat.st_size, stat.st_mtime_ns, False
        )

        h.scanner.index_file(found)

        assert h.queued() == []


class TestWithoutADigestPool:
    def test_a_scanner_with_no_pool_hashes_every_file_inline(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        harness = Harness(tmp_path)
        harness.scanner.attach_digests(None)
        content = _blob(1)
        harness.put("home", "a.safetensors", content)

        result = harness.index()

        assert harness.model("a.safetensors").sha256 == _sha(content)
        assert result["contested"] == 0

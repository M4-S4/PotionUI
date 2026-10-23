import os

import pytest

from src.features.models import digests as digests_module
from src.features.models import indexer as indexer_module
from src.features.models.file_digest import INLINE_HASH_LIMIT
from tests.features.models.fakes import drain_manager
from tests.features.models.rows_first_support import Harness
from tests.features.models.test_indexer_roots import _models

MIB = 1024 * 1024
BIG_SIZES = [INLINE_HASH_LIMIT + 1 + index for index in range(25)]
TINY = [b"embedding-%d" % index * 40 for index in range(5)]


def _sparse(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.truncate(size)


@pytest.fixture
def reads(monkeypatch):
    seen = {"inline": [], "pool": []}
    real_chunks = indexer_module.read_chunks
    real_digest = digests_module.digest_file

    def chunks(path, *args, **kwargs):
        seen["inline"].append(path)
        return real_chunks(path, *args, **kwargs)

    def digest(path, *args, **kwargs):
        seen["pool"].append(path)
        return real_digest(path, *args, **kwargs)

    monkeypatch.setattr(indexer_module, "read_chunks", chunks)
    monkeypatch.setattr(digests_module, "digest_file", digest)
    return seen


@pytest.fixture
def library(tmp_path, mock_db):
    harness = Harness(tmp_path)
    for index, size in enumerate(BIG_SIZES):
        _sparse(harness.home / "loras" / f"lora{index:03d}.safetensors", size)
    for index, content in enumerate(TINY):
        harness.put("lib", f"embedding{index}.safetensors", content)
    return harness


def test_the_inline_limit_is_one_mebibyte():
    assert INLINE_HASH_LIMIT == MIB


class TestAScanNeverReadsABigLibrary:
    def test_files_just_over_the_inline_limit_are_rows_at_once_with_zero_full_reads(self, library, reads):
        result = library.index()

        assert (result["indexed"], result["failed"], result["contested"]) == (len(BIG_SIZES) + len(TINY), 0, 0)
        assert reads["pool"] == []
        assert sorted(os.path.basename(path) for path in reads["inline"]) == [
            f"embedding{index}.safetensors" for index in range(len(TINY))
        ]

    def test_the_rows_exist_and_only_the_tiny_files_have_a_digest(self, library, reads):
        library.index()

        models = {m.filename: m for m in _models()}
        assert all(model.is_available for model in models.values())
        assert all(models[f"lora{i:03d}.safetensors"].sha256 is None for i in range(len(BIG_SIZES)))
        assert all(models[f"embedding{i}.safetensors"].sha256 for i in range(len(TINY)))
        assert models["lora000.safetensors"].file_size == BIG_SIZES[0]

    def test_a_second_scan_of_the_same_library_reads_nothing_at_all(self, library, reads):
        library.index()
        reads["inline"].clear()

        library.index()

        assert reads == {"inline": [], "pool": []}


class TestNothingIsReadWhileTheChoiceIsOpen:
    def test_a_big_library_waits_for_the_choice_and_the_pool_has_nothing_to_hash(self, library, reads):
        library.index()

        assert library.manager.policy == "unset"
        assert library.manager.plan()["awaiting_choice"] is True
        assert library.queued() == []
        assert drain_manager(library.manager) == 0
        assert reads["pool"] == []

    def test_a_scan_after_a_restart_still_queues_and_hashes_nothing_while_the_choice_is_open(self, library, reads):
        library.index()
        library.manager.apply_policy("startup")
        library.index()

        assert library.queued() == []
        assert drain_manager(library.manager) == 0
        assert reads["pool"] == []

    def test_choosing_background_is_what_starts_the_reading(self, library, reads):
        library.index()

        library.manager.set_policy("background")
        drain_manager(library.manager, max_jobs=3)

        assert len(reads["pool"]) == 3
        assert all(os.path.getsize(path) > INLINE_HASH_LIMIT for path in reads["pool"])

    def test_changing_ones_mind_to_on_demand_takes_the_library_back_out_of_the_queue(self, library, reads):
        library.index()
        library.manager.set_policy("background")
        assert len(library.queued()) == len(BIG_SIZES)

        library.manager.set_policy("on_demand")

        assert drain_manager(library.manager) == 0
        assert reads["pool"] == []

    def test_a_small_library_goes_background_by_itself_and_is_read_only_by_the_paced_pool(self, tmp_path, mock_db, reads):
        harness = Harness(tmp_path)
        for index in range(3):
            _sparse(harness.home / "loras" / f"small{index}.safetensors", (5 + index) * MIB)

        harness.index()

        assert harness.manager.policy == "background"
        assert len(harness.queued()) == 3
        assert reads["pool"] == [] and reads["inline"] == []

    def test_two_big_files_that_look_identical_are_still_compared_even_while_the_choice_is_open(self, library, reads):
        for name in ("copy_a.safetensors", "copy_b.safetensors"):
            _sparse(library.home / "loras" / name, 7 * MIB + 12345)

        result = library.index()
        drain_manager(library.manager)

        assert result["contested"] == 2
        assert library.manager.policy == "unset"
        assert sorted(os.path.basename(path) for path in reads["pool"]) == ["copy_a.safetensors", "copy_b.safetensors"]
        assert [m.filename for m in _models() if m.filename.startswith("copy_")] == ["copy_a.safetensors"]

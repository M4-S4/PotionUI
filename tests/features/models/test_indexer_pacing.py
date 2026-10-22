import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from unittest.mock import patch

from src.features.models import indexer as indexer_module
from src.features.models.indexer import ModelScanner, ScanCancelled
from src.features.models.indexing_governor import INDEXING_LEVEL_SETTING_KEY, IndexingGovernor
from src.features.models.read_throttle import ReadThrottle
from src.platform.runtime.thread_priority import ThreadPriority
from src.platform.settings.repository import SettingRepository
from tests.features.models.fakes import FakeClock
from tests.features.models.test_indexer_roots import _binding, _models, _resolver, _root, _write

MIB = 1024 * 1024


def _library(tmp_path, names=("a.safetensors", "b.safetensors")):
    home = tmp_path / "home"
    for name in names:
        _write(home / "loras" / name, name.encode() * 64)
    return _resolver([_root("r_home", home)], [_binding("r_home", "lora", home / "loras", 0)])


def _governor(level, clock=None, busy=lambda: False):
    clock = clock or FakeClock()
    governor = IndexingGovernor(throttle=ReadThrottle(clock=clock.monotonic, sleep=clock.sleep, busy=busy))
    governor.set_level(level)
    return governor, clock


@pytest.fixture
def pool(monkeypatch):
    created = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, max_workers=None, initializer=None, **kwargs):
            created.append(max_workers)
            super().__init__(max_workers=max_workers, initializer=initializer, **kwargs)

    monkeypatch.setattr(indexer_module, "ThreadPoolExecutor", RecordingPool)
    return created


@pytest.fixture
def priorities(monkeypatch):
    applied = []
    monkeypatch.setattr(
        indexer_module, "set_current_thread_priority",
        lambda priority: applied.append((priority, threading.current_thread() is threading.main_thread())),
    )
    return applied


class TestWorkersFollowTheLevel:
    @pytest.mark.parametrize(
        "level,workers,priority",
        [("low", 1, ThreadPriority.BACKGROUND), ("normal", 2, ThreadPriority.BELOW_NORMAL), ("max", 4, ThreadPriority.NORMAL)],
    )
    def test_pool_size_and_thread_priority(self, level, workers, priority, tmp_path, mock_db, pool, priorities):
        governor, _ = _governor(level)
        scanner = ModelScanner(_library(tmp_path), governor=governor)

        result = scanner.index_models()

        assert result["indexed"] == 2
        assert pool == [workers]
        assert priorities
        assert {entry[0] for entry in priorities} == {priority}
        assert not any(on_main for _, on_main in priorities)

    def test_an_explicit_worker_count_overrides_the_level(self, tmp_path, mock_db, pool, priorities):
        governor, _ = _governor("low")
        scanner = ModelScanner(_library(tmp_path), governor=governor)

        scanner.index_models(max_workers=3)

        assert pool == [3]

    def test_a_scanner_without_a_governor_keeps_the_unrestricted_defaults(self, tmp_path, mock_db, pool, priorities):
        scanner = ModelScanner(_library(tmp_path))

        scanner.index_models()

        assert pool == [4]
        assert {entry[0] for entry in priorities} == {ThreadPriority.NORMAL}

    def test_digests_are_the_same_under_every_level(self, tmp_path, mock_db):
        expected = {
            name: hashlib.sha256(name.encode() * 64).hexdigest() for name in ("a.safetensors", "b.safetensors")
        }
        governor, _ = _governor("low")

        ModelScanner(_library(tmp_path), governor=governor).index_models()

        assert {m.filename: m.sha256 for m in _models()} == expected

    def test_the_level_is_read_again_for_every_pass(self, tmp_path, mock_db, pool, priorities):
        governor, _ = _governor("low")
        scanner = ModelScanner(_library(tmp_path), governor=governor)
        scanner.index_models()

        governor.set_level("normal")
        _write(tmp_path / "home" / "loras" / "c.safetensors", b"c" * 64)
        scanner.index_models()

        assert pool == [1, 2]

    @pytest.mark.parametrize("level,throttled", [("low", True), ("normal", True), ("max", False)])
    def test_the_index_pass_is_throttled_below_max_only(self, level, throttled, tmp_path, mock_db):
        governor, clock = _governor(level)

        ModelScanner(_library(tmp_path), governor=governor).index_models()

        assert bool(clock.sleeps) is throttled


    def test_a_level_changed_in_the_settings_table_applies_to_the_next_pass(self, tmp_path, mock_db):
        clock = FakeClock()
        governor = IndexingGovernor(SettingRepository(), throttle=ReadThrottle(clock=clock.monotonic, sleep=clock.sleep))
        governor.set_level("max")
        SettingRepository().update_setting_value_by_key(INDEXING_LEVEL_SETTING_KEY, "low")

        ModelScanner(_library(tmp_path), governor=governor).index_models()

        assert clock.sleeps


    def test_a_level_change_during_a_pass_reaches_the_files_still_to_come(self, tmp_path, mock_db):
        governor, clock = _governor("max")
        scanner = ModelScanner(_library(tmp_path, names=("a.safetensors", "b.safetensors", "c.safetensors")), governor=governor)
        level_changed = threading.Event()
        real_index_file = ModelScanner.index_file
        calls = []

        def gated(self, found, cancel_check=None, pacing=None, **kwargs):
            calls.append(found.rel_path)
            if len(calls) == 2:
                assert level_changed.wait(5)
            return real_index_file(self, found, cancel_check, pacing, **kwargs)

        def on_progress(current, total, message):
            if current >= 1 and not level_changed.is_set():
                governor.set_level("low")
                level_changed.set()

        scanner.set_progress_callback(on_progress)
        with patch.object(ModelScanner, "index_file", gated):
            scanner.index_models(max_workers=1)

        assert clock.sleeps
        assert len(_models()) == 3


class TestPacedHashing:
    def _file(self, tmp_path, size):
        path = tmp_path / "weights.safetensors"
        path.write_bytes(bytes(range(256)) * (size // 256) + bytes(size % 256))
        return path

    def test_without_pacing_nothing_sleeps(self, tmp_path):
        path = self._file(tmp_path, 4096)

        digest = ModelScanner(resolver=None).calculate_sha256(str(path), chunk_size=1024)

        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_a_generation_pause_holds_the_read_and_resumes_with_the_same_digest(self, tmp_path):
        path = self._file(tmp_path, 4096)
        clock = FakeClock()
        running = [False]
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: running[0], clock=clock.monotonic, sleep=clock.sleep)
        clock.on_sleep = lambda: running.__setitem__(0, len(clock.sleeps) < 2)

        class StartsAGeneration:
            def __init__(self):
                self.started = False

            def feed(self, chunk):
                if not self.started:
                    self.started = True
                    running[0] = True

        digest = ModelScanner(resolver=None).calculate_sha256(
            str(path), chunk_size=1024, pacing=throttle, tap=StartsAGeneration()
        )

        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        assert clock.sleeps == [0.5, 0.5]

    def test_cancelling_during_a_pause_stops_the_hash(self, tmp_path):
        path = self._file(tmp_path, 4096)
        clock = FakeClock()
        throttle = ReadThrottle(pauses_for_generation=True, busy=lambda: True, clock=clock.monotonic, sleep=clock.sleep)

        with pytest.raises(ScanCancelled):
            ModelScanner(resolver=None).calculate_sha256(
                str(path), chunk_size=1024, pacing=throttle, cancel_check=lambda: len(clock.sleeps) >= 2
            )

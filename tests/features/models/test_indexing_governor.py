from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.features.models.indexing_governor import (
    DEFAULT_LEVEL,
    INDEXING_LEVEL_SETTING_KEY,
    LEVEL_LOW,
    LEVEL_MAX,
    LEVEL_NORMAL,
    PROFILES,
    IndexingGovernor,
    InvalidIndexingLevel,
    local_generation_probe,
    local_native_backend_ids,
    normalize_level,
)
from src.features.models.read_throttle import ReadThrottle
from src.platform.runtime.thread_priority import ThreadPriority
from src.platform.settings.repository import SettingRepository
from tests.features.models.fakes import FakeClock

MIB = 1024 * 1024


class TestProfiles:
    def test_low_is_the_default_and_the_gentlest(self):
        low = PROFILES[LEVEL_LOW]

        assert DEFAULT_LEVEL == LEVEL_LOW
        assert (low.workers, low.priority) == (1, ThreadPriority.BACKGROUND)
        assert low.read_bytes_per_second == 96 * MIB
        assert low.pauses_for_generation is True
        assert low.evict_page_cache is True

    def test_normal_uses_two_workers_and_a_higher_cap(self):
        normal = PROFILES[LEVEL_NORMAL]

        assert (normal.workers, normal.priority) == (2, ThreadPriority.BELOW_NORMAL)
        assert normal.read_bytes_per_second == 384 * MIB
        assert normal.pauses_for_generation is True

    def test_max_is_unrestricted_like_the_previous_behaviour(self):
        maximum = PROFILES[LEVEL_MAX]

        assert (maximum.workers, maximum.priority) == (4, ThreadPriority.NORMAL)
        assert maximum.read_bytes_per_second is None
        assert maximum.pauses_for_generation is False

    @pytest.mark.parametrize("value,expected", [("low", "low"), (" Normal ", "normal"), ("MAX", "max"), ("fast", None), (None, None), (3, None)])
    def test_levels_are_normalised(self, value, expected):
        assert normalize_level(value) == expected


class TestGovernorWithoutStorage:
    def test_an_unmanaged_governor_never_throttles(self):
        governor = IndexingGovernor.unmanaged()

        profile = governor.refresh()

        assert profile.level == LEVEL_MAX
        assert governor.throttle.wait(10**12) is True
        assert governor.throttle.holding() is False

    def test_set_level_rejects_unknown_values(self):
        governor = IndexingGovernor()

        with pytest.raises(InvalidIndexingLevel):
            governor.set_level("turbo")

        assert governor.level() == LEVEL_LOW


class TestGovernorWithStoredSetting:
    def test_the_level_round_trips_through_the_settings_table(self, mock_db):
        first = IndexingGovernor(SettingRepository())

        first.set_level("normal")
        second = IndexingGovernor(SettingRepository())

        assert second.level() == LEVEL_NORMAL
        assert SettingRepository().get_setting_by_key(INDEXING_LEVEL_SETTING_KEY).value == "normal"

    def test_changing_the_level_again_updates_the_same_row(self, mock_db):
        governor = IndexingGovernor(SettingRepository())
        governor.set_level("normal")
        governor.set_level("max")

        rows = [s for s in SettingRepository().get_all_settings() if s.key == INDEXING_LEVEL_SETTING_KEY]

        assert [row.value for row in rows] == ["max"]
        assert governor.level() == LEVEL_MAX

    def test_a_stored_value_that_is_not_a_level_falls_back_to_the_default(self, mock_db):
        governor = IndexingGovernor(SettingRepository())
        governor.set_level("normal")
        SettingRepository().update_setting_value_by_key(INDEXING_LEVEL_SETTING_KEY, "warp")

        assert governor.level() == LEVEL_LOW

    def test_a_missing_setting_row_refuses_the_change_and_keeps_the_default(self, mock_db):
        governor = IndexingGovernor(SettingRepository())
        SettingRepository().delete_setting_by_key(INDEXING_LEVEL_SETTING_KEY)

        with pytest.raises(RuntimeError, match=INDEXING_LEVEL_SETTING_KEY):
            governor.set_level("max")

        assert governor.level() == LEVEL_LOW

    def test_an_unreadable_settings_table_falls_back_to_the_default(self):
        repository = Mock()
        repository.get_setting_by_key.side_effect = RuntimeError("database is locked")

        assert IndexingGovernor(repository).level() == LEVEL_LOW


class TestGovernorDrivesTheThrottle:
    def _governor(self, repository=None, busy=lambda: False):
        clock = FakeClock()
        throttle = ReadThrottle(clock=clock.monotonic, sleep=clock.sleep, busy=busy)
        return IndexingGovernor(repository, throttle=throttle), clock

    def test_low_caps_reads_and_max_removes_the_cap(self):
        governor, clock = self._governor()
        pacing = governor.throttle

        pacing.wait(96 * MIB)
        pacing.wait(96 * MIB)
        capped = sum(clock.sleeps)
        governor.set_level("max")
        pacing.wait(96 * MIB)
        pacing.wait(96 * MIB)

        assert capped == 1.0
        assert sum(clock.sleeps) == capped

    def test_the_throttle_follows_the_pause_flag_of_the_level(self):
        governor, _ = self._governor(busy=lambda: True)

        governor.set_level("max")

        assert governor.throttle.wait(1) is True
        assert governor.throttle.holding() is False


class TestLocalBackendIds:
    def _backend(self, backend_id, driver):
        return SimpleNamespace(backend_id=backend_id, config=SimpleNamespace(driver=driver))

    def test_only_local_native_backends_count(self):
        registry = Mock()
        registry.get_backends_for_engine.return_value = [
            self._backend("local", "native.local"),
            self._backend("worker", "native.remote"),
        ]

        assert local_native_backend_ids(registry) == {"local"}
        registry.get_backends_for_engine.assert_called_once_with("native")

    def test_a_missing_or_failing_registry_means_none(self):
        failing = Mock()
        failing.get_backends_for_engine.side_effect = RuntimeError("not ready")

        assert local_native_backend_ids(None) == set()
        assert local_native_backend_ids(failing) == set()


class TestGenerationProbe:
    def _record(self, state, backend_id):
        return SimpleNamespace(state=SimpleNamespace(value=state), backend_id=backend_id)

    def _probe(self, records, local=("local",)):
        tracker = Mock()
        tracker.list_active.return_value = records
        return local_generation_probe(tracker, lambda: local)

    def test_a_running_generation_on_the_local_backend_is_busy(self):
        assert self._probe([self._record("running", "local")])() is True

    def test_a_queued_generation_is_not_busy_yet(self):
        assert self._probe([self._record("pending", "local")])() is False

    def test_a_generation_on_another_backend_does_not_pause_indexing(self):
        assert self._probe([self._record("running", "cloud")])() is False

    def test_no_local_backend_means_never_busy(self):
        assert self._probe([self._record("running", "local")], local=())() is False

    def test_nothing_running_is_idle(self):
        assert self._probe([])() is False

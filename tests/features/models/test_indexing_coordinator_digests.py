import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.features.models import indexer as indexer_module
from src.features.models.digest_queue_repository import PRIORITY_BACKGROUND, PRIORITY_CHOSEN
from src.features.models.digest_queue_repository import DigestQueueRepository
from src.features.models.digests import DigestManager
from src.features.models.hooks import MODEL_INDEX_HOOKS
from src.features.models.indexer import ModelScanner
from src.features.models.indexing_coordinator import CONTESTED_WARNING, ModelIndexingCoordinator
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.read_throttle import ReadThrottle
from src.features.models.type_manager import ModelTypeManager
from src.features.models.type_repository import ModelTypeRepository
from src.features.models.repository import model_repo
from src.platform.database import get_cursor
from tests.features.models.fakes import FakeClock, OpenPlugins, drain_manager
from tests.features.models.rows_first_support import SMALL_LIMIT, Harness, _blob
from src.platform.settings.repository import SettingRepository
from tests.features.models.test_indexer_roots import _models, _sha
from tests.fixtures.model_index_fixtures import FLUX, Library


class Stack:
    def __init__(self, tmp_path):
        self.harness = Harness(tmp_path)
        self.reconciler = MagicMock()
        self.reconciler.reconcile = AsyncMock()
        self.coordinator = ModelIndexingCoordinator(
            model_repository=model_repo,
            plugin_registry=OpenPlugins(),
            scanner=self.harness.scanner,
            native_availability_projector=self.reconciler,
            spawn=lambda target: target(),
        )
        self.harness.manager._on_applied = self.coordinator.on_digests_applied

    @property
    def reconciles(self):
        return self.reconciler.reconcile.await_count


@pytest.fixture
def stack(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    return Stack(tmp_path)


class TestScanAndDigests:
    def test_a_scan_makes_the_rows_and_leaves_the_digests_to_the_pool(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))

        status = stack.coordinator.start_indexing(trigger="manual")

        assert (status["state"], status["indexed"]) == ("done", 1)
        assert stack.harness.queued() == [("a.safetensors", "r_home", PRIORITY_BACKGROUND, "scan")]
        assert stack.harness.model("a.safetensors").sha256 is None

    def test_the_applied_digests_settle_the_models_and_refresh_availability(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")
        before = stack.reconciles

        stack.harness.settle()

        assert stack.harness.model("a.safetensors").sha256 == _sha(_blob(1))
        assert stack.reconciles == before + 1

    def test_type_conflicts_found_while_settling_are_added_to_the_status(self, stack, monkeypatch):
        conflict = {"model_id": "m1", "model_type": "checkpoint", "message": "blocked"}
        monkeypatch.setattr(stack.harness.scanner, "settle_digested", lambda ids: [conflict])

        stack.coordinator.on_digests_applied({"m1"})
        stack.coordinator.on_digests_applied({"m1"})

        conflicts = [c for c in stack.coordinator.status()["conflicts"] if c.get("model_id") == "m1"]
        assert conflicts == [conflict]


class TestContestedFiles:
    def test_the_status_reports_a_file_that_turned_out_to_duplicate_another_when_it_is_decided(self, stack):
        content = _blob(1)
        stack.harness.put("home", "a.safetensors", content)
        stack.harness.put("lib", "b.safetensors", content)
        stack.coordinator.start_indexing(trigger="manual")
        assert _models() == []
        before = stack.reconciles

        stack.harness.settle()

        status = stack.coordinator.status()
        assert [m.filename for m in _models()] == ["a.safetensors"]
        assert status["indexed"] == 1
        assert status["skipped_duplicates_total"] == 1
        assert status["skipped_duplicates"][0]["path"].endswith("b.safetensors")
        assert status["skipped_duplicates"][0]["same_as"]["model_id"] == _models()[0].id
        assert stack.reconciles > before

    def test_a_contested_file_that_cannot_be_read_is_listed_as_failed(self, stack):
        content = _blob(1)
        stack.harness.put("home", "a.safetensors", content)
        stack.coordinator.start_indexing(trigger="manual")
        stack.harness.settle()
        gone = stack.harness.put("lib", "b.safetensors", content)
        stack.coordinator.start_indexing(trigger="manual")
        gone.unlink()

        stack.harness.settle(max_jobs=10)

        failed = stack.coordinator.status()["failed_files"]
        assert [entry["path"] for entry in failed] == [str(gone)]
        assert failed[0]["error"]
        assert stack.coordinator.status()["failed_files_total"] == 1

    def test_a_failing_listener_refresh_does_not_break_the_decision(self, stack, monkeypatch):
        content = _blob(1)
        stack.harness.put("home", "a.safetensors", content)
        stack.harness.put("lib", "b.safetensors", content)
        stack.coordinator.start_indexing(trigger="manual")
        stack.reconciler.reconcile.side_effect = RuntimeError("projector is down")

        stack.harness.settle()

        assert [m.filename for m in _models()] == ["a.safetensors"]


class TestTheFirstRunChoice:
    def test_booting_with_nothing_indexed_does_not_make_the_choice(self, stack):
        assert stack.coordinator.resume_interrupted_indexing() is False

        assert stack.harness.manager.policy == "unset"

    def test_a_big_library_added_after_an_empty_first_boot_still_asks(self, stack):
        stack.coordinator.resume_interrupted_indexing()
        for index in range(21):
            stack.harness.put("home", f"m{index:03d}.safetensors", _blob(index))

        stack.coordinator.start_indexing(trigger="manual")

        digests = stack.coordinator.status()["digests"]
        assert (digests["policy"], digests["queued"], digests["plan"]["awaiting_choice"]) == ("unset", 0, True)

    def test_a_small_first_library_is_queued_and_remembered_by_the_scan_that_found_it(self, stack):
        stack.coordinator.resume_interrupted_indexing()
        stack.harness.put("home", "a.safetensors", _blob(1))

        stack.coordinator.resume_interrupted_indexing()

        assert stack.harness.manager.policy == "background"
        assert [entry[3] for entry in stack.harness.queued()] == ["scan"]

    def test_an_unset_policy_is_left_alone_at_startup_even_with_pending_files(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")
        stack.harness.manager._store_policy("unset")
        stack.harness.queue.remove([row["id"] for row in stack.harness.locations(stack.harness.model("a.safetensors"))])

        stack.coordinator.settle_digest_queue("startup")

        assert stack.harness.manager.policy == "unset"
        assert stack.harness.queued() == []

    def test_a_chosen_background_policy_is_applied_at_startup(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")
        stack.harness.queue.remove([row["id"] for row in stack.harness.locations(stack.harness.model("a.safetensors"))])

        stack.coordinator.settle_digest_queue("startup")

        assert stack.harness.queued() == [("a.safetensors", "r_home", PRIORITY_BACKGROUND, "startup")]


class TestDigestQueueAtStartup:
    def test_pending_locations_that_were_never_queued_are_queued_when_indexing_resumes(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")
        stack.harness.queue.remove([row["id"] for row in stack.harness.locations(stack.harness.model("a.safetensors"))])
        assert stack.harness.queued() == []

        started = stack.coordinator.resume_interrupted_indexing()

        assert started is False
        assert stack.harness.queued() == [("a.safetensors", "r_home", PRIORITY_BACKGROUND, "startup")]

    def test_locations_that_went_missing_are_dropped_from_the_queue_at_startup(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")
        with get_cursor() as cursor:
            cursor.execute("UPDATE model_locations SET status = 'missing'")
        assert len(stack.harness.queued()) == 1

        stack.coordinator.settle_digest_queue("startup")

        assert stack.harness.queued() == []

    def test_a_coordinator_whose_scanner_has_no_pool_has_nothing_to_settle(self, stack):
        stack.harness.scanner.attach_digests(None)

        stack.coordinator.settle_digest_queue("startup")


class TestLevelChanges:
    def test_a_new_level_reaches_the_digest_pool_so_its_workers_follow_it(self, stack, monkeypatch):
        seen = []
        real = stack.harness.manager.set_level
        monkeypatch.setattr(stack.harness.manager, "set_level", lambda level: seen.append(level) or real(level))

        result = stack.coordinator.update_settings(level="normal")

        assert seen == ["normal"]
        assert result == {"level": "normal", "policy": "unset"}
        assert stack.harness.manager.status()["workers"] == 2

    def test_an_unchanged_level_leaves_the_pool_alone(self, stack, monkeypatch):
        seen = []
        monkeypatch.setattr(stack.harness.manager, "set_level", lambda level: seen.append(level))

        result = stack.coordinator.update_settings()

        assert seen == [] and result == {"level": "low", "policy": "unset"}


class TestSingleFiles:
    def test_a_new_download_without_a_checksum_is_a_row_at_once_and_a_chosen_job(self, stack):
        path = stack.harness.put("home", "a.safetensors", _blob(1))

        result = stack.coordinator.index_downloaded(str(path))

        assert result["indexed"] is True and result["warning"] is None
        assert stack.harness.queued() == [("a.safetensors", "r_home", PRIORITY_CHOSEN, "pick")]
        assert stack.reconciles == 1

    def test_a_file_the_size_gate_cannot_clear_is_reported_as_being_compared(self, stack):
        content = _blob(1)
        stack.harness.put("home", "a.safetensors", content)
        stack.coordinator.start_indexing(trigger="manual")
        stack.harness.settle()
        reconciles = stack.reconciles
        path = stack.harness.put("lib", "b.safetensors", content)

        result = stack.coordinator.index_path(str(path))

        assert result["indexed"] is False and result["model_id"] is None
        assert result["warning"] == CONTESTED_WARNING
        assert stack.reconciles == reconciles


class DownloadStack:
    def __init__(self, tmp_path):
        self.library = Library(tmp_path, scan_checkpoints=False, scan_diffusion=False)
        clock = FakeClock()
        governor = IndexingGovernor(throttle=ReadThrottle(clock=clock.monotonic, sleep=clock.sleep))
        self.scanner = ModelScanner(self.library.scanner.resolver, governor=governor)
        self.manager = DigestManager(
            DigestQueueRepository(), governor, self.scanner.location_path,
            locations_write_lock=self.scanner.write_lock, setting_repository=SettingRepository(),
            clock=clock.time, monotonic=clock.monotonic, sleep=clock.sleep,
        )
        self.scanner.attach_digests(self.manager)
        self.types = ModelTypeManager(model_repo, self.scanner.types, self.scanner.recompute_types)
        reconciler = MagicMock()
        reconciler.reconcile = AsyncMock()
        self.coordinator = ModelIndexingCoordinator(
            model_repository=model_repo, plugin_registry=OpenPlugins(), scanner=self.scanner,
            native_availability_projector=reconciler, spawn=lambda target: target(), type_manager=self.types,
        )
        self.manager._on_applied = self.coordinator.on_digests_applied

    def assertion(self, path):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return ModelTypeRepository().get_assertions([sha]).get(sha)


@pytest.fixture
def downloads(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", 1)
    return DownloadStack(tmp_path)


class TestDownloadsWithoutAChecksum:
    def test_the_type_is_pinned_once_the_digest_lands(self, downloads):
        path = downloads.library.put(downloads.library.checkpoints, "flux.safetensors", FLUX)

        result = downloads.coordinator.index_downloaded(str(path))

        assert result["indexed"] is True and result["warning"] is None
        assert downloads.assertion(path) is None

        drain_manager(downloads.manager)

        assert downloads.assertion(path)["source"] == "download"
        model = model_repo.get_by_id(result["model_id"])
        assert (model.model_type, model.type_source) == ("diffusion_model", "download")

    def test_the_pin_is_made_only_once(self, downloads, monkeypatch):
        path = downloads.library.put(downloads.library.checkpoints, "flux.safetensors", FLUX)
        downloads.coordinator.index_downloaded(str(path))
        pins = []
        real = downloads.coordinator._pin_downloaded_type
        monkeypatch.setattr(downloads.coordinator, "_pin_downloaded_type", lambda model, loc: pins.append(model.id) or real(model, loc))

        drain_manager(downloads.manager)
        downloads.coordinator.on_digests_applied({model_repo.get_by_filename("flux.safetensors")[0].id})

        assert len(pins) == 1

    def test_a_file_that_waits_to_be_compared_is_pinned_when_it_is_decided(self, downloads):
        library = downloads.library
        first = library.put(library.diffusion, "flux.safetensors", FLUX, tag="same")
        downloads.coordinator.start_indexing(trigger="manual")
        drain_manager(downloads.manager)
        second = library.put(library.checkpoints, "flux.safetensors", FLUX, tag="same")

        result = downloads.coordinator.index_downloaded(str(second))

        assert result["indexed"] is False and result["warning"] == CONTESTED_WARNING
        assert downloads.assertion(second) is None

        drain_manager(downloads.manager)

        assert downloads.assertion(second)["source"] == "download"
        assert [m.filename for m in _models()] == ["flux.safetensors"]

    def test_a_download_that_is_not_header_classified_is_never_pinned(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        stack = Stack(tmp_path)
        path = stack.harness.put("home", "a.safetensors", _blob(1))

        result = stack.coordinator.index_downloaded(str(path))
        stack.harness.settle()

        assert result["indexed"] is True
        assert stack.coordinator._pins_awaiting_digest == {}


class RecordingPlugins:
    def __init__(self):
        self.fired = []

    def execute_hook(self, hook, initial_data):
        self.fired.append((hook, dict(initial_data)))
        return MagicMock(data={}), False

    def after_digests(self):
        return [data["model_ids"] for hook, data in self.fired if hook == MODEL_INDEX_HOOKS.after_digests]


class TimerClock:
    def __init__(self):
        self.now = 100.0
        self.timers = []

    def __call__(self):
        return self.now

    def start_timer(self, delay, target):
        timer = SimpleNamespace(delay=delay, target=target, cancelled=False)
        timer.cancel = lambda: setattr(timer, "cancelled", True)
        self.timers.append(timer)
        return timer


def _announcing_coordinator(scanner=None):
    plugins = RecordingPlugins()
    clock = TimerClock()
    coordinator = ModelIndexingCoordinator(
        model_repository=MagicMock(), plugin_registry=plugins, scanner=scanner or MagicMock(),
        native_availability_projector=MagicMock(reconcile=AsyncMock()), spawn=lambda target: target(),
        clock=clock, start_timer=clock.start_timer,
    )
    return coordinator, plugins, clock


class TestAfterDigestsHook:
    def test_the_first_batch_fires_the_hook_at_once_with_every_model_id_sorted(self):
        coordinator, plugins, clock = _announcing_coordinator()

        coordinator.on_digests_applied({"m2", "m1"})

        assert plugins.after_digests() == [["m1", "m2"]]
        assert clock.timers == []

    def test_a_second_batch_inside_the_window_waits_for_a_timer_and_the_ids_pile_up(self):
        coordinator, plugins, clock = _announcing_coordinator()
        coordinator.on_digests_applied({"m1"})
        clock.now += 2

        coordinator.on_digests_applied({"m2"})
        coordinator.on_digests_applied({"m3", "m2"})

        assert plugins.after_digests() == [["m1"]]
        assert [timer.delay for timer in clock.timers] == [3.0]

        clock.now += 3
        clock.timers[0].target()

        assert plugins.after_digests() == [["m1"], ["m2", "m3"]]

    def test_a_batch_after_the_window_fires_at_once_again(self):
        coordinator, plugins, clock = _announcing_coordinator()
        coordinator.on_digests_applied({"m1"})
        clock.now += 5

        coordinator.on_digests_applied({"m2"})

        assert plugins.after_digests() == [["m1"], ["m2"]]
        assert clock.timers == []

    def test_the_hook_is_never_fired_twice_within_five_seconds(self):
        coordinator, plugins, clock = _announcing_coordinator()
        coordinator.on_digests_applied({"m1"})
        for tick in range(4):
            clock.now += 1
            coordinator.on_digests_applied({f"x{tick}"})

        assert len(plugins.after_digests()) == 1
        assert len(clock.timers) == 1

    def test_shutting_down_cancels_the_pending_timer_and_drops_the_ids(self):
        coordinator, plugins, clock = _announcing_coordinator()
        coordinator.on_digests_applied({"m1"})
        clock.now += 1
        coordinator.on_digests_applied({"m2"})

        coordinator.shutdown()
        clock.timers[0].target()

        assert clock.timers[0].cancelled is True
        assert plugins.after_digests() == [["m1"]]

    def test_nothing_is_announced_and_no_timer_is_started_after_shutdown(self):
        coordinator, plugins, clock = _announcing_coordinator()
        coordinator.on_digests_applied({"m0"})
        coordinator.shutdown()
        clock.now += 1

        coordinator.on_digests_applied({"m1"})

        assert plugins.after_digests() == [["m0"]]
        assert clock.timers == []

    def test_a_hook_that_raises_does_not_break_the_apply(self, stack_with_failing_hook):
        stack_with_failing_hook.harness.put("home", "a.safetensors", _blob(1))
        stack_with_failing_hook.coordinator.start_indexing(trigger="manual")

        stack_with_failing_hook.harness.settle()

        assert stack_with_failing_hook.harness.model("a.safetensors").sha256 == _sha(_blob(1))

    def test_the_real_pool_announces_the_models_it_just_hashed(self, tmp_path, mock_db, monkeypatch):
        monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
        stack = Stack(tmp_path)
        plugins = RecordingPlugins()
        stack.coordinator.plugins = plugins
        stack.harness.put("home", "a.safetensors", _blob(1))
        stack.coordinator.start_indexing(trigger="manual")

        stack.harness.settle()

        model = stack.harness.model("a.safetensors")
        assert plugins.after_digests() == [[model.id]]


@pytest.fixture
def stack_with_failing_hook(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    stack = Stack(tmp_path)

    class Failing(OpenPlugins):
        def execute_hook(self, hook, initial_data):
            if hook == MODEL_INDEX_HOOKS.after_digests:
                raise RuntimeError("plugin exploded")
            return super().execute_hook(hook, initial_data)

    stack.coordinator.plugins = Failing()
    return stack


class TestUpdatingTheSettings:
    def test_the_policy_goes_to_the_digest_service_and_comes_back_in_the_answer(self, stack):
        result = stack.coordinator.update_settings(level="normal", policy="on_demand")

        assert result == {"level": "normal", "policy": "on_demand"}
        assert stack.harness.manager.policy == "on_demand"

    def test_leaving_everything_out_changes_nothing_and_still_reports_both(self, stack):
        assert stack.coordinator.update_settings() == {"level": "low", "policy": "unset"}

    def test_a_scanner_without_a_pool_reports_only_the_level(self, stack):
        stack.harness.scanner.attach_digests(None)

        assert stack.coordinator.update_settings(level="max", policy="background") == {"level": "max"}

    def test_the_status_block_is_left_out_when_there_is_no_pool(self, stack):
        stack.harness.scanner.attach_digests(None)

        assert "digests" not in stack.coordinator.status()

    def test_a_status_block_that_cannot_be_built_does_not_break_the_status(self, stack, monkeypatch):
        monkeypatch.setattr(stack.harness.manager, "status", lambda: (_ for _ in ()).throw(RuntimeError("locked")))

        assert stack.coordinator.status()["digests"] is None


class TestThePolicyGateAtTheEndOfAPass:
    def test_a_small_library_is_queued_silently_and_the_choice_is_remembered(self, stack):
        stack.harness.put("home", "a.safetensors", _blob(1))

        stack.coordinator.start_indexing(trigger="manual")

        assert stack.harness.manager.policy == "background"
        assert stack.coordinator.status()["digests"]["plan"]["awaiting_choice"] is False

    def test_a_library_over_the_threshold_queues_nothing_and_asks(self, stack):
        for index in range(21):
            stack.harness.put("home", f"m{index:03d}.safetensors", _blob(index))

        stack.coordinator.start_indexing(trigger="manual")

        digests = stack.coordinator.status()["digests"]
        assert (digests["policy"], digests["queued"], digests["plan"]["awaiting_choice"]) == ("unset", 0, True)

    def test_a_chosen_on_demand_policy_is_respected_by_the_next_pass_and_by_startup(self, stack):
        stack.harness.manager.set_policy("on_demand")
        stack.harness.put("home", "a.safetensors", _blob(1))

        stack.coordinator.start_indexing(trigger="manual")
        stack.coordinator.settle_digest_queue("startup")

        assert stack.harness.queued() == []

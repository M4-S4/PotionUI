import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.features.models import digests as digests_module
from src.features.models import indexer as indexer_module
from src.features.models.digest_queue_repository import PRIORITY_BACKGROUND, PRIORITY_CHOSEN
from src.features.models.repository import model_repo
from src.features.recipes.executors.artifacts_fetch import ArtifactsFetchExecutor
from src.features.recipes.executors.artifacts_plan import ArtifactsPlanExecutor
from src.features.recipes.executors.base import StepContext
from src.features.recipes.records import RecipeRun, RecipeRunStatus
from src.features.recipes.schema import Recipe, RecipeArtifact, RecipeChecksum, RecipeStep
from src.platform.runtime.gpu_profile import GpuProfile
from tests.features.models.fakes import drain_manager
from tests.features.models.rows_first_support import SMALL_LIMIT, Harness, _blob
from tests.features.models.test_indexer_roots import _sha

SIZE = 4096
WANTED = _blob(1, SIZE)


def _artifact():
    return RecipeArtifact(
        id="ckpt", kind="lora", model_type="lora", filename="wanted.safetensors", display_name="Wanted",
        size_bytes=SIZE, checksum=RecipeChecksum("sha256", _sha(WANTED)),
        provider_hint={"download_url": "https://example.test/wanted"},
    )


def _context(step_kind, progress=None):
    recipe = Recipe(id="x", schema_version=1, version=1, name="X", engine="native", artifacts=[_artifact()])
    run = RecipeRun(id="r1", recipe_id="x", recipe_version=1, scope="instance", status=RecipeRunStatus.RUNNING, created_by="o")
    step = RecipeStep(key=step_kind, kind=step_kind, title="Step", params={"artifact_ids": ["ckpt"]})
    return StepContext(
        run=run, recipe=recipe, step=step,
        report_progress=(lambda **kwargs: progress.append(kwargs)) if progress is not None else StepContext.report_progress,
    )


@pytest.fixture
def library(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", SMALL_LIMIT)
    harness = Harness(tmp_path)
    harness.manager.set_policy("on_demand")
    harness.clock.on_sleep = lambda: drain_manager(harness.manager)
    return harness


@pytest.fixture
def reads(monkeypatch):
    seen = []
    real = digests_module.digest_file

    def spy(path, size, cancel_check, pacing, *args, **kwargs):
        seen.append((os.path.basename(path), pacing is not None))
        return real(path, size, cancel_check, pacing, *args, **kwargs)

    monkeypatch.setattr(digests_module, "digest_file", spy)
    return seen


def _download_service():
    return SimpleNamespace(
        queue_model_download=AsyncMock(return_value=SimpleNamespace(id="dl-1")),
        get_download=lambda download_id: SimpleNamespace(status="completed", downloaded_bytes=None, total_bytes=None),
    )


def _put_candidates(library, queued_first=()):
    library.put("home", "wanted_copy.safetensors", WANTED)
    for index in range(3):
        library.put("home", f"other{index}.safetensors", _blob(2 + index, SIZE))
    library.index()
    ids = [
        row["id"]
        for name in queued_first
        for row in library.locations(library.model(name))
    ]
    library.queue.enqueue(ids, PRIORITY_BACKGROUND, "scan", "2026-01-01T00:00:00")


class TestThePlanStepReadsNothing:
    def test_same_size_files_are_queued_paced_and_reported_but_never_read_by_the_plan(self, library, reads):
        _put_candidates(library)
        executor = ArtifactsPlanExecutor(model_repo, gpu_profile_provider=GpuProfile, digests=library.manager)

        result = executor.execute(_context("artifacts.plan"))

        assert result.awaiting_consent is True
        assert reads == []
        assert result.consent_request["checking"] == [
            {"id": "ckpt", "display_name": "Wanted", "files": 4, "message": "Checking whether you already have Wanted"}
        ]
        priorities = {row["priority"] for row in library.queue.next_jobs(20, 0)}
        assert priorities == {PRIORITY_CHOSEN}
        assert [entry[3] for entry in library.queued()] == ["match"] * 4

    def test_a_file_that_is_there_under_its_own_name_needs_no_checking_at_all(self, library, reads):
        library.put("home", "wanted.safetensors", WANTED)
        library.index()
        executor = ArtifactsPlanExecutor(model_repo, gpu_profile_provider=GpuProfile, digests=library.manager)

        result = executor.execute(_context("artifacts.plan"))

        assert result.success is True and not result.awaiting_consent
        assert reads == [] and library.queued() == []


class TestTheFetchStepVerifiesAtMostTwoFilesPaced:
    def test_a_match_among_the_first_two_means_no_download(self, library, reads):
        _put_candidates(library, queued_first=("wanted_copy.safetensors", "other0.safetensors"))
        service = _download_service()
        progress = []
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch", progress))

        assert result.success is True and result.safe_output["fetched"] == []
        service.queue_model_download.assert_not_called()
        assert len(reads) == 2 and all(paced for _, paced in reads)
        assert progress == [
            {"progress_current": 0, "progress_total": 2, "progress_unit": "files"},
            {"progress_current": 2, "progress_total": 2, "progress_unit": "files"},
        ]

    def test_no_more_than_two_files_are_ever_read_and_a_match_beyond_them_is_downloaded_instead(self, library, reads):
        _put_candidates(library, queued_first=("other0.safetensors", "other1.safetensors"))
        service = _download_service()
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch"))

        assert len(reads) == 2 and all(paced for _, paced in reads)
        assert [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]
        service.queue_model_download.assert_awaited_once()

    def test_while_indexing_is_paused_nothing_is_read_and_the_step_does_not_wait(self, library, reads):
        _put_candidates(library)
        library.manager.pause()
        service = _download_service()
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch"))

        assert reads == []
        assert [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]
        service.queue_model_download.assert_awaited_once()

    def test_when_nothing_has_the_same_size_the_file_is_downloaded_without_any_checking(self, library, reads):
        library.put("home", "other.safetensors", _blob(9, SIZE + 1))
        library.index()
        service = _download_service()
        progress = []
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch", progress))

        assert reads == [] and [item["progress_unit"] for item in progress] == []
        assert [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]

    def test_when_the_pool_never_runs_nothing_is_read_the_file_is_downloaded_once_and_the_step_succeeds(
        self, library, reads, monkeypatch
    ):
        monkeypatch.setattr(digests_module, "PACED_WAIT_SECONDS", 3)
        library.clock.on_sleep = None
        _put_candidates(library)
        service = _download_service()
        progress = []
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch", progress))

        assert result.success is True and reads == []
        assert [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]
        service.queue_model_download.assert_awaited_once()
        assert progress == [{"progress_current": 0, "progress_total": 2, "progress_unit": "files"}]
        assert 3 <= library.clock.now < 10

    def test_a_wait_that_runs_out_after_one_file_reports_one_of_two(self, library, reads, monkeypatch):
        monkeypatch.setattr(digests_module, "PACED_WAIT_SECONDS", 3)
        calls = []

        def one_job_then_nothing():
            if not calls:
                drain_manager(library.manager, max_jobs=1)
            calls.append(1)

        library.clock.on_sleep = one_job_then_nothing
        _put_candidates(library)
        progress = []
        executor = ArtifactsFetchExecutor(_download_service(), model_repo, digests=library.manager)

        executor.execute(_context("artifacts.fetch", progress))

        assert len(reads) == 1
        assert [(item["progress_current"], item["progress_total"]) for item in progress] == [(0, 2), (1, 2)]

    def test_a_candidate_backing_off_after_failures_does_not_hold_the_step_back(self, library, reads):
        _put_candidates(library, queued_first=("other0.safetensors", "other1.safetensors"))
        for name in ("other0.safetensors", "other1.safetensors"):
            location = library.locations(library.model(name))[0]["id"]
            library.queue.enqueue([location], PRIORITY_CHOSEN, "match", "2026-01-01T00:00:00")
            library.queue.record_failure(location, "unreadable", library.clock.time(), (600.0,))
        service = _download_service()
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch"))

        assert result.success is True and reads == []
        assert library.clock.sleeps == []
        service.queue_model_download.assert_awaited_once()

    def test_a_candidate_on_an_offline_root_does_not_hold_the_step_back(self, library, reads, monkeypatch):
        _put_candidates(library, queued_first=("other0.safetensors", "other1.safetensors"))
        monkeypatch.setattr(library.manager, "_path_for", lambda row: None)
        service = _download_service()
        executor = ArtifactsFetchExecutor(service, model_repo, digests=library.manager)

        result = executor.execute(_context("artifacts.fetch"))

        assert result.success is True and reads == []
        assert library.clock.sleeps == []
        service.queue_model_download.assert_awaited_once()

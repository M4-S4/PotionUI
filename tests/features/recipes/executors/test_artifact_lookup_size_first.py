from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.features.models.digests import DigestMatch
from src.features.recipes.executors._artifact_lookup import find_artifact_model, find_slot_model
from src.features.recipes.executors._slot_description import describe_slot
from src.features.recipes.executors.artifacts_fetch import ArtifactsFetchExecutor
from src.features.recipes.executors.artifacts_plan import ArtifactsPlanExecutor
from src.features.recipes.executors.base import StepContext
from src.features.recipes.records import RecipeRun, RecipeRunStatus
from src.features.recipes.schema import (
    Recipe,
    RecipeArtifact,
    RecipeArtifactVariant,
    RecipeChecksum,
    RecipeStep,
)
from src.features.recipes.slot_variants import RecipeSlotVariants, VariantDownloadError
from src.platform.runtime.gpu_profile import GpuProfile

SHA = "a" * 64
SIZE = 4096


class Repo:
    def __init__(self, by_identity=None, by_sha=None):
        self.by_identity = dict(by_identity or {})
        self.by_sha = dict(by_sha or {})

    def get_by_identity(self, model_type, filename, include_providers=True):
        return self.by_identity.get((model_type, filename))

    def get_by_sha256(self, sha256, include_providers=True):
        return self.by_sha.get(sha256)


class Digests:
    def __init__(self, found=None, candidates=0):
        self.found = found
        self.candidates = candidates
        self.calls = []

    def find_by_digest(self, sha256, size, hash_candidates="never", timeout=None):
        self.calls.append((sha256, size, hash_candidates))
        if hash_candidates == "never":
            return DigestMatch(None, self.candidates)
        return DigestMatch(self.found, self.candidates)


def _model(model_type="checkpoint", available=True, filename="other-name.safetensors"):
    return SimpleNamespace(id="m1", model_type=model_type, is_available=available, filename=filename, file_path="/x/y")


def _artifact(size=SIZE, sha=SHA, aid="ckpt"):
    return RecipeArtifact(
        id=aid, kind="checkpoint", model_type="checkpoint", filename="model.safetensors", display_name="Model",
        size_bytes=size, checksum=RecipeChecksum("sha256", sha) if sha else None,
        provider_hint={"download_url": "https://example.test/m"},
    )


class TestFindArtifactModel:
    def test_a_name_match_needs_no_digest_at_all(self):
        repo = Repo(by_identity={("checkpoint", "model.safetensors"): _model()})
        digests = Digests(found=_model())

        assert find_artifact_model(repo, _artifact(), None, digests, "inline") is not None
        assert digests.calls == []

    def test_a_digest_that_is_already_recorded_needs_no_size_lookup(self):
        repo = Repo(by_sha={SHA: _model()})
        digests = Digests()

        assert find_artifact_model(repo, _artifact(), None, digests, "inline") is not None
        assert digests.calls == []

    def test_a_renamed_file_is_found_by_its_size_and_the_declared_checksum(self):
        found = _model()
        digests = Digests(found=found, candidates=1)

        result = find_artifact_model(Repo(), _artifact(), None, digests, "inline")

        assert result is found
        assert digests.calls == [(SHA, SIZE, "inline")]

    def test_the_mode_the_caller_chose_is_the_one_used(self):
        digests = Digests()

        find_artifact_model(Repo(), _artifact(), None, digests, "background")

        assert digests.calls == [(SHA, SIZE, "background")]

    def test_a_file_that_is_no_longer_available_does_not_count(self):
        digests = Digests(found=_model(available=False))

        assert find_artifact_model(Repo(), _artifact(), None, digests, "inline") is None

    def test_without_a_size_or_a_checksum_the_digest_service_is_not_asked(self):
        digests = Digests(found=_model())

        assert find_artifact_model(Repo(), _artifact(size=None), None, digests, "inline") is None
        assert find_artifact_model(Repo(), _artifact(sha=None), None, digests, "inline") is None
        assert digests.calls == []

    def test_without_a_digest_service_nothing_changes(self):
        assert find_artifact_model(Repo(), _artifact()) is None

    def test_a_match_of_the_wrong_type_is_re_typed_for_the_recipe(self):
        found = _model(model_type="lora")
        asserted = []
        type_manager = SimpleNamespace(assert_type=lambda sha, model_type, source: asserted.append((sha, model_type, source)))
        digests = Digests(found=found)

        find_artifact_model(Repo(by_sha={}), _artifact(), type_manager, digests, "inline")

        assert asserted == [(SHA, "checkpoint", "recipe")]


class TestSlots:
    def _variant(self, vid, sha):
        return RecipeArtifactVariant(
            id=vid, label=vid, precision="fp8", filename=f"{vid}.safetensors", size_bytes=SIZE,
            checksum=RecipeChecksum("sha256", sha), provider_hint={"download_url": "https://example.test/x"},
        )

    def _slot_artifact(self):
        return RecipeArtifact(
            id="dit", kind="checkpoint", model_type="checkpoint", filename="dit.safetensors", display_name="DiT",
            variants=(self._variant("v1", "1" * 64), self._variant("v2", "2" * 64)),
        )

    def test_every_variant_is_looked_for_by_size_and_checksum(self):
        digests = Digests()

        find_slot_model(Repo(), self._slot_artifact(), None, None, digests, "inline")

        assert digests.calls == [("1" * 64, SIZE, "inline"), ("2" * 64, SIZE, "inline")]

    def test_describing_a_slot_marks_the_variant_that_was_found_by_digest(self):
        found = _model(filename="renamed.safetensors")

        class OneVariantDigests(Digests):
            def find_by_digest(self, sha256, size, hash_candidates="never", timeout=None):
                return DigestMatch(found if sha256 == "2" * 64 else None, 1)

        slot, choice, installed = describe_slot(
            Repo(), self._slot_artifact(), GpuProfile(), None, OneVariantDigests(), "inline"
        )

        assert list(installed) == ["v2"]


def _recipe(artifact):
    return Recipe(id="x", schema_version=1, version=1, name="X", engine="native", artifacts=[artifact])


def _context(recipe, step_kind, progress):
    run = RecipeRun(id="r1", recipe_id="x", recipe_version=1, scope="instance", status=RecipeRunStatus.RUNNING, created_by="o")
    step = RecipeStep(key=step_kind, kind=step_kind, title="Step", params={"artifact_ids": ["ckpt"]})
    return StepContext(run=run, recipe=recipe, step=step, report_progress=lambda **kwargs: progress.append(kwargs))


class SettlingDigests(Digests):
    def __init__(self, found, candidates=2):
        super().__init__(found=None, candidates=candidates)
        self._settles_to = found
        self.settled = False

    def find_by_digest(self, sha256, size, hash_candidates="never", timeout=None, on_progress=None):
        self.calls.append((sha256, size, hash_candidates))
        if hash_candidates == "paced":
            checked = min(self.candidates, 2)
            if on_progress is not None:
                on_progress(0, checked)
                on_progress(checked, checked)
            self.settled = True
        model = self._settles_to if self.settled else None
        return DigestMatch(model, self.candidates)


class TestPlanExecutor:
    def test_a_renamed_file_that_is_already_known_counts_as_present_without_any_checking(self):
        progress = []
        digests = Digests(found=_model(), candidates=2)
        executor = ArtifactsPlanExecutor(Repo(), gpu_profile_provider=GpuProfile, digests=digests)

        result = executor.execute(_context(_recipe(_artifact()), "artifacts.plan", progress))

        assert result.success is True and "already_present" in result.safe_output
        assert progress == []
        assert [call[2] for call in digests.calls] == ["background"]

    def test_same_size_files_that_are_not_verified_yet_are_reported_as_being_checked_and_the_plan_goes_on(self):
        digests = Digests(found=None, candidates=3)
        executor = ArtifactsPlanExecutor(Repo(), gpu_profile_provider=GpuProfile, digests=digests)

        result = executor.execute(_context(_recipe(_artifact()), "artifacts.plan", []))

        assert result.awaiting_consent is True
        request = result.consent_request
        assert [item["id"] for item in request["artifacts"]] == ["ckpt"]
        assert request["artifacts"][0]["checking_files"] == 3
        assert request["checking"] == [
            {"id": "ckpt", "display_name": "Model", "files": 3, "message": "Checking whether you already have Model"}
        ]
        assert {call[2] for call in digests.calls} <= {"background", "never"}
        assert "inline" not in {call[2] for call in digests.calls} and "paced" not in {call[2] for call in digests.calls}

    def test_nothing_is_reported_when_no_file_could_be_it(self):
        executor = ArtifactsPlanExecutor(Repo(), gpu_profile_provider=GpuProfile, digests=Digests())

        result = executor.execute(_context(_recipe(_artifact()), "artifacts.plan", []))

        assert result.awaiting_consent is True
        assert "checking" not in result.consent_request
        assert "checking_files" not in result.consent_request["artifacts"][0]

    def test_a_digest_match_found_before_any_hashing_reports_nothing(self):
        progress = []

        class AlreadyKnown(Digests):
            def find_by_digest(self, sha256, size, hash_candidates="never", timeout=None):
                self.calls.append((sha256, size, hash_candidates))
                return DigestMatch(_model(), 0)

        executor = ArtifactsPlanExecutor(Repo(), gpu_profile_provider=GpuProfile, digests=AlreadyKnown())

        executor.execute(_context(_recipe(_artifact()), "artifacts.plan", progress))

        assert progress == []

    def test_without_a_digest_service_the_plan_is_unchanged(self):
        executor = ArtifactsPlanExecutor(Repo(), gpu_profile_provider=GpuProfile)

        result = executor.execute(_context(_recipe(_artifact()), "artifacts.plan", []))

        assert result.awaiting_consent is True


class TestFetchExecutor:
    def _service(self):
        return SimpleNamespace(
            queue_model_download=AsyncMock(return_value=SimpleNamespace(id="dl-1")),
            get_download=lambda download_id: SimpleNamespace(status="completed", downloaded_bytes=None, total_bytes=None),
        )

    def test_a_file_whose_digest_is_already_known_needs_no_checking_and_no_download(self):
        digests = Digests()
        executor = ArtifactsFetchExecutor(object(), Repo(by_sha={SHA: _model()}), digests=digests)

        assert executor._target(_artifact(), None) is None
        assert digests.calls == []

    def test_same_size_files_are_verified_paced_and_a_match_means_no_download(self):
        digests = SettlingDigests(found=_model(), candidates=2)
        service = self._service()
        progress = []
        executor = ArtifactsFetchExecutor(service, Repo(), digests=digests)

        result = executor.execute(_context(_recipe(_artifact(aid="ckpt")), "artifacts.fetch", progress))

        assert result.success is True and result.safe_output["fetched"] == []
        service.queue_model_download.assert_not_called()
        assert [call[2] for call in digests.calls].count("paced") == 1
        assert "inline" not in [call[2] for call in digests.calls]
        assert progress == [
            {"progress_current": 0, "progress_total": 2, "progress_unit": "files"},
            {"progress_current": 2, "progress_total": 2, "progress_unit": "files"},
        ]

    def test_the_verification_is_announced_for_at_most_two_files(self):
        digests = SettlingDigests(found=None, candidates=9)
        progress = []
        executor = ArtifactsFetchExecutor(self._service(), Repo(), digests=digests)

        executor.execute(_context(_recipe(_artifact(aid="ckpt")), "artifacts.fetch", progress))

        assert [item["progress_total"] for item in progress[:2]] == [2, 2]

    def test_when_no_candidate_matches_the_file_is_downloaded(self):
        digests = SettlingDigests(found=None, candidates=2)
        service = self._service()
        executor = ArtifactsFetchExecutor(service, Repo(), digests=digests)

        result = executor.execute(_context(_recipe(_artifact(aid="ckpt")), "artifacts.fetch", []))

        assert result.success is True and [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]
        service.queue_model_download.assert_awaited_once()

    def test_running_the_step_downloads_a_file_nobody_has_without_any_checking(self):
        digests = Digests()
        service = self._service()
        executor = ArtifactsFetchExecutor(service, Repo(), digests=digests)

        result = executor.execute(_context(_recipe(_artifact(aid="ckpt")), "artifacts.fetch", []))

        assert result.success is True and [item["id"] for item in result.safe_output["fetched"]] == ["ckpt"]
        service.queue_model_download.assert_awaited_once()
        assert {call[2] for call in digests.calls} == {"never"}

    def test_a_file_nobody_has_is_still_a_download_target(self):
        executor = ArtifactsFetchExecutor(object(), Repo(), digests=Digests())

        assert executor._target(_artifact(), None) is not None


class TestQueueingAVariantDownload:
    def _variants(self, digests, queue):
        artifact = RecipeArtifact(
            id="dit", kind="checkpoint", model_type="checkpoint", filename="dit.safetensors", display_name="DiT",
            variants=(
                RecipeArtifactVariant(
                    id="v1", label="v1", precision="fp8", filename="v1.safetensors", size_bytes=SIZE,
                    checksum=RecipeChecksum("sha256", "1" * 64), provider_hint={"download_url": "https://example.test/v1"},
                ),
            ),
        )
        recipe = _recipe(artifact)
        catalog = SimpleNamespace(get_recipe=lambda recipe_id: recipe)
        return RecipeSlotVariants(
            catalog, Repo(), download_queue=queue, digests=digests, gpu_profile_provider=GpuProfile
        )

    def test_a_variant_already_on_disk_under_another_name_is_refused_and_found_without_hashing_in_the_request(self):
        digests = Digests(found=_model(filename="renamed.safetensors"))
        queue = SimpleNamespace(queue_model_download=AsyncMock())

        with pytest.raises(VariantDownloadError) as refused:
            self._variants(digests, queue).queue_download("x", "dit", "v1", "admin-1")

        assert refused.value.code == "already_installed"
        assert digests.calls == [("1" * 64, SIZE, "background")]
        queue.queue_model_download.assert_not_called()

    def test_a_variant_nobody_has_is_queued_after_one_background_lookup(self):
        digests = Digests()
        queue = SimpleNamespace(queue_model_download=AsyncMock(return_value=SimpleNamespace(id="dl-1")))

        result = self._variants(digests, queue).queue_download("x", "dit", "v1", "admin-1")

        assert digests.calls == [("1" * 64, SIZE, "background")]
        queue.queue_model_download.assert_awaited_once()
        assert queue.queue_model_download.await_args.kwargs["filename"] == "v1.safetensors"
        assert result == {"download_id": "dl-1", "filename": "v1.safetensors"}


class TestSlotVariants:
    def test_a_listing_never_hashes_in_the_request_thread(self):
        digests = Digests()
        variants = RecipeSlotVariants(SimpleNamespace(), Repo(), digests=digests, gpu_profile_provider=GpuProfile)
        artifact = RecipeArtifact(
            id="dit", kind="checkpoint", model_type="checkpoint", filename="dit.safetensors", display_name="DiT",
            variants=(
                RecipeArtifactVariant(
                    id="v1", label="v1", precision="fp8", filename="v1.safetensors", size_bytes=SIZE,
                    checksum=RecipeChecksum("sha256", "1" * 64),
                ),
            ),
        )

        variants._slot_view(_recipe(artifact), artifact, GpuProfile())

        assert {call[2] for call in digests.calls} == {"background"}

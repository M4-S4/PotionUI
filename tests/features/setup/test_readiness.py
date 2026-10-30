"""The four-facet readiness aggregate and its role filtering.

Every collaborator the manager composes is faked, so these are pure unit tests
of the aggregation logic: which facet flips to which status under which
condition, how the overall verdict combines them, and what an admin sees versus
a regular user.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.features.models.digests import DigestMatch
from src.features.recipes.schema import (
    Recipe,
    RecipeArtifact,
    RecipeArtifactVariant,
    RecipeChecksum,
    RecipePresetRef,
)
from src.features.setup.readiness import (
    DEGRADED,
    NOT_READY,
    READY,
    ReadinessAggregator,
    _format_missing,
)
from src.platform.security.user import AccountType, User

AVAILABILITY = "src.features.models.availability.models_for_engine"


@pytest.fixture(autouse=True)
def _forward_preset_operations_to_manager(monkeypatch):
    """`ReadinessAggregator` calls `src.features.presets.operations.list_presets`
    with the preset-manager collaborator as its leading arg, rather than
    calling `.list_presets()` on it directly. This forwards that call to the
    fake's own method, so `preset_collaborators` here can stay a plain mock with
    `.list_presets`, exactly like the retired manager double."""
    from src.features.presets import operations as preset_operations_module

    def _list_presets(collaborators, *args, **kwargs):
        return collaborators.list_presets(*args, **kwargs)

    monkeypatch.setattr(preset_operations_module, "list_presets", _list_presets)


# --- fakes ------------------------------------------------------------------

def _user(admin=False):
    return User(
        username="u",
        email="u@example.com",
        password_hash="x",
        account_type=AccountType.ADMIN if admin else AccountType.USER,
        id="u1",
    )


class _OkClaimRepository:
    def check_connection(self):
        return None


class _FailClaimRepository:
    def check_connection(self):
        raise RuntimeError("database is locked")


def _backend(name="native-1", engine="native", status="healthy", reason=None, error=None):
    info = {"status": status}
    if reason is not None:
        info["reason"] = reason
    if error is not None:
        info["error"] = error
    backend = MagicMock()
    backend.name = name
    backend.engine = engine
    backend.health_check = AsyncMock(return_value=info)
    return backend


def _manager(
    backends=None,
    presets=None,
    completed=1,
    pending=False,
    instance_claim_repository=None,
    recipe_catalog=None,
    model_repository=None,
    model_indexing_coordinator=None,
    model_digests=None,
):
    backend_registry = MagicMock()
    backend_registry.get_all_backends.return_value = (
        {"native-1": _backend()} if backends is None else backends
    )
    preset_collaborators = MagicMock()
    preset_collaborators.list_presets.return_value = (
        [{"id": "p1", "engine": "native"}] if presets is None else presets
    )
    if model_repository is None:
        model_repository = MagicMock()
        model_repository.get_available_model_ids_for_user.return_value = ["m1"]
    generation_repository = MagicMock()
    generation_repository.count_by_status.return_value = completed
    migration_runner = MagicMock()
    migration_runner.has_pending_migrations.return_value = pending
    return ReadinessAggregator(
        backend_registry=backend_registry,
        preset_collaborators=preset_collaborators,
        model_repository=model_repository,
        generation_repository=generation_repository,
        migration_runner=migration_runner,
        instance_claim_repository=instance_claim_repository or _OkClaimRepository(),
        recipe_catalog=recipe_catalog,
        model_indexing_coordinator=model_indexing_coordinator,
        model_digests=model_digests,
    )


class _RecipeModelRepo:
    def __init__(self, present=()):
        self._present = set(present)

    def get_by_identity(self, model_type, filename):
        return object() if (model_type, filename) in self._present else None

    def get_by_sha256(self, sha256):
        return None


class _RecipeCatalog:
    def __init__(self, recipe):
        self._recipe = recipe

    def get_recipe(self, recipe_id):
        return self._recipe if recipe_id == self._recipe.id else None


def _recipe(artifacts=(), presets=None, recipe_id="r1"):
    return Recipe(
        id=recipe_id,
        schema_version=1,
        version=1,
        name="Test Recipe",
        engine="native",
        category="image",
        artifacts=list(artifacts),
        presets=[RecipePresetRef(preset_id="p1")] if presets is None else list(presets),
    )


def _run(manager, user, recipe_id=None, models_result=None):
    """Evaluate with `models_for_engine` stubbed to control content resolution."""
    result = [{"id": "m1"}] if models_result is None else models_result
    with patch(AVAILABILITY, return_value=result):
        return asyncio.run(manager.evaluate(user, recipe_id=recipe_id))


def _by_area(report):
    return {check.area: check for check in report.checks}


# --- happy path -------------------------------------------------------------

def test_all_green_is_ready():
    report = _run(_manager(), _user(admin=True))
    assert report.overall == READY
    areas = _by_area(report)
    assert areas["service"].code == "SERVICE_OK"
    assert areas["execution"].code == "EXECUTION_READY"
    assert areas["content"].code == "CONTENT_READY"
    assert areas["generation_proven"].code == "GENERATION_PROVEN"
    assert all(c.status == READY for c in report.checks)


def test_report_rows_use_the_doctor_registry_shape():
    report = _run(_manager(), _user(admin=True))
    for check in report.checks:
        assert set(check.model_dump().keys()) == {"area", "status", "code", "message", "action"}


# --- execution facet --------------------------------------------------------

def test_no_backend_is_not_ready():
    report = _run(_manager(backends={}), _user(admin=True))
    execution = _by_area(report)["execution"]
    assert execution.status == NOT_READY
    assert execution.code == "NO_EXECUTION_BACKEND"
    assert report.overall == NOT_READY


def test_degraded_backend_is_degraded():
    backends = {"b": _backend(status="degraded", reason="no CUDA GPU is visible")}
    report = _run(_manager(backends=backends), _user(admin=True))
    execution = _by_area(report)["execution"]
    assert execution.status == DEGRADED
    assert execution.code == "EXECUTION_DEGRADED"
    assert "no CUDA GPU is visible" in execution.message
    # no facet is not_ready -> overall settles at degraded, not not_ready
    assert report.overall == DEGRADED


def test_erroring_backend_is_not_ready():
    backends = {"b": _backend(status="error", error="driver exploded")}
    report = _run(_manager(backends=backends), _user(admin=True))
    execution = _by_area(report)["execution"]
    assert execution.status == NOT_READY
    assert execution.code == "EXECUTION_UNHEALTHY"


def test_one_healthy_among_degraded_wins():
    backends = {
        "bad": _backend(name="bad", status="degraded", reason="misconfigured"),
        "good": _backend(name="good", status="healthy"),
    }
    report = _run(_manager(backends=backends), _user(admin=True))
    execution = _by_area(report)["execution"]
    assert execution.status == READY
    assert "good" in execution.message


# --- content facet ----------------------------------------------------------

def test_no_presets_assigned_is_not_ready():
    report = _run(_manager(presets=[]), _user(admin=True))
    content = _by_area(report)["content"]
    assert content.status == NOT_READY
    assert content.code == "NO_PRESETS_ASSIGNED"
    assert report.overall == NOT_READY
    assert "->" not in content.action
    assert "Admin → Presets" in content.action


def test_presets_without_resolvable_models_is_degraded():
    report = _run(_manager(), _user(admin=True), models_result=[])
    content = _by_area(report)["content"]
    assert content.status == DEGRADED
    assert content.code == "PRESETS_WITHOUT_MODELS"


def test_presets_without_models_says_indexing_in_progress_instead_of_degraded_message():
    coordinator = MagicMock()
    coordinator.status.return_value = {"state": "indexing", "processed": 3, "total": 10}
    report = _run(_manager(model_indexing_coordinator=coordinator), _user(admin=True), models_result=[])
    content = _by_area(report)["content"]
    assert content.status == DEGRADED
    assert content.code == "MODELS_INDEXING"
    assert "3/10" in content.message


def test_presets_without_models_stays_plain_degraded_when_indexing_is_idle():
    coordinator = MagicMock()
    coordinator.status.return_value = {"state": "idle", "processed": 0, "total": 0}
    report = _run(_manager(model_indexing_coordinator=coordinator), _user(admin=True), models_result=[])
    content = _by_area(report)["content"]
    assert content.code == "PRESETS_WITHOUT_MODELS"


def test_unknown_recipe_id_is_not_ready():
    report = _run(_manager(recipe_catalog=_RecipeCatalog(_recipe())), _user(admin=True), recipe_id="does-not-exist")
    content = _by_area(report)["content"]
    assert content.status == NOT_READY
    assert content.code == "RECIPE_NOT_FOUND"


def test_recipe_ready_when_artifacts_indexed_and_presets_installed():
    artifact = RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename="model.safetensors", required=True,
    )
    recipe = _recipe(artifacts=[artifact])
    manager = _manager(
        recipe_catalog=_RecipeCatalog(recipe),
        model_repository=_RecipeModelRepo(present={("checkpoint", "model.safetensors")}),
    )
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == READY
    assert content.code == "RECIPE_READY"
    assert report.overall == READY


def test_recipe_ready_variant_satisfied_by_non_default_variant():
    artifact = RecipeArtifact(
        id="ckpt",
        kind="checkpoint",
        model_type="checkpoint",
        filename="default.safetensors",
        required=True,
        variants=(
            RecipeArtifactVariant(
                id="fp16", label="fp16", precision="fp16", filename="default.safetensors", default=True,
            ),
            RecipeArtifactVariant(
                id="fp8", label="fp8", precision="fp8", filename="default-fp8.safetensors",
            ),
        ),
    )
    recipe = _recipe(artifacts=[artifact])
    manager = _manager(
        recipe_catalog=_RecipeCatalog(recipe),
        model_repository=_RecipeModelRepo(present={("checkpoint", "default-fp8.safetensors")}),
    )
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == READY
    assert content.code == "RECIPE_READY"


def test_recipe_missing_artifact_is_not_ready():
    artifact = RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename="missing.safetensors",
        display_name="Missing Checkpoint", required=True,
    )
    recipe = _recipe(artifacts=[artifact])
    manager = _manager(recipe_catalog=_RecipeCatalog(recipe), model_repository=_RecipeModelRepo())
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == NOT_READY
    assert content.code == "RECIPE_MODELS_MISSING"
    assert "Missing Checkpoint" in content.message
    assert report.overall == NOT_READY


def test_recipe_missing_artifact_says_indexing_in_progress_while_a_scan_is_running():
    artifact = RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename="missing.safetensors",
        display_name="Missing Checkpoint", required=True,
    )
    recipe = _recipe(artifacts=[artifact])
    coordinator = MagicMock()
    coordinator.status.return_value = {"state": "scanning", "processed": 0, "total": 0}
    manager = _manager(
        recipe_catalog=_RecipeCatalog(recipe),
        model_repository=_RecipeModelRepo(),
        model_indexing_coordinator=coordinator,
    )
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == DEGRADED
    assert content.code == "RECIPE_MODELS_INDEXING"
    assert report.overall == DEGRADED


class _Digests:
    def __init__(self, found=None, candidates=0, waiting=None):
        self.found = found
        self.candidates = candidates
        self.waiting = candidates if waiting is None else waiting
        self.calls = []

    def find_by_digest(self, sha256, size, hash_candidates="never", timeout=None):
        self.calls.append((sha256, size, hash_candidates))
        return DigestMatch(self.found, self.candidates, self.waiting)


def _checksummed(size=4096, filename="missing.safetensors"):
    return RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename=filename, display_name="Missing Checkpoint",
        required=True, size_bytes=size, checksum=RecipeChecksum("sha256", "a" * 64),
    )


def _recipe_manager(digests, coordinator=None):
    manager = _manager(
        recipe_catalog=_RecipeCatalog(_recipe(artifacts=[_checksummed()])),
        model_repository=_RecipeModelRepo(),
        model_indexing_coordinator=coordinator,
        model_digests=digests,
    )
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    return manager


def test_a_same_size_file_that_is_not_verified_yet_reads_as_verifying_not_missing():
    digests = _Digests(candidates=2)

    report = _run(_recipe_manager(digests), _user(admin=True), recipe_id="r1")

    content = _by_area(report)["content"]
    assert (content.status, content.code) == (DEGRADED, "RECIPE_MODELS_VERIFYING")
    assert "2 files that may be them are being verified" in content.message
    assert "Missing Checkpoint" in content.message
    assert {call[2] for call in digests.calls} == {"never"}


def test_same_size_files_that_nobody_is_comparing_yet_stay_missing_and_say_so():
    digests = _Digests(candidates=2, waiting=0)

    content = _by_area(_run(_recipe_manager(digests), _user(admin=True), recipe_id="r1"))["content"]

    assert (content.status, content.code) == (NOT_READY, "RECIPE_MODELS_MISSING")
    assert "2 files of the same size exist but have not been compared yet" in content.message


def test_a_single_unchecked_file_is_worded_in_the_singular_while_missing():
    content = _by_area(
        _run(_recipe_manager(_Digests(candidates=1, waiting=0)), _user(admin=True), recipe_id="r1")
    )["content"]

    assert content.code == "RECIPE_MODELS_MISSING"
    assert "1 file of the same size exists but has not been compared yet" in content.message


def test_a_single_unverified_file_is_worded_in_the_singular():
    content = _by_area(_run(_recipe_manager(_Digests(candidates=1)), _user(admin=True), recipe_id="r1"))["content"]

    assert "1 file that may be one of them is being verified" in content.message


def test_a_renamed_file_whose_digest_is_known_counts_as_present():
    found = MagicMock(model_type="checkpoint", is_available=True, filename="renamed.safetensors")

    report = _run(_recipe_manager(_Digests(found=found)), _user(admin=True), recipe_id="r1")

    content = _by_area(report)["content"]
    assert (content.status, content.code) == (READY, "RECIPE_READY")


def test_no_same_size_file_at_all_is_still_missing():
    digests = _Digests(candidates=0)

    content = _by_area(_run(_recipe_manager(digests), _user(admin=True), recipe_id="r1"))["content"]

    assert (content.status, content.code) == (NOT_READY, "RECIPE_MODELS_MISSING")
    assert {call[2] for call in digests.calls} == {"never"}


def test_a_running_scan_is_reported_before_the_verifying_state():
    coordinator = MagicMock()
    coordinator.status.return_value = {"state": "scanning", "processed": 1, "total": 5}

    content = _by_area(
        _run(_recipe_manager(_Digests(candidates=3), coordinator), _user(admin=True), recipe_id="r1")
    )["content"]

    assert content.code == "RECIPE_MODELS_INDEXING"


def test_checking_a_recipe_with_the_real_digest_service_reads_no_file(tmp_path, mock_db, monkeypatch):
    from src.features.models import digests as digests_module
    from src.features.models import indexer as indexer_module
    from tests.features.models.rows_first_support import Harness, _blob

    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", 10)
    harness = Harness(tmp_path)
    harness.put("home", "renamed.safetensors", _blob(1, 4096))
    harness.index()
    reads = []
    real = digests_module.digest_file
    monkeypatch.setattr(digests_module, "digest_file", lambda *args, **kwargs: reads.append(args[0]) or real(*args, **kwargs))
    manager = _recipe_manager(harness.manager)

    content = _by_area(_run(manager, _user(admin=True), recipe_id="r1"))["content"]

    assert content.code == "RECIPE_MODELS_VERIFYING"
    assert reads == [] and harness.queued() == [("renamed.safetensors", "r_home", 0, "scan")]


def test_recipe_optional_artifact_missing_is_still_ready():
    required = RecipeArtifact(
        id="ckpt", kind="checkpoint", model_type="checkpoint", filename="model.safetensors", required=True,
    )
    optional = RecipeArtifact(
        id="lora", kind="lora", model_type="lora", filename="extra.safetensors", required=False,
    )
    recipe = _recipe(artifacts=[required, optional])
    manager = _manager(
        recipe_catalog=_RecipeCatalog(recipe),
        model_repository=_RecipeModelRepo(present={("checkpoint", "model.safetensors")}),
    )
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = True
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == READY
    assert content.code == "RECIPE_READY"


def test_recipe_preset_not_installed_is_not_ready():
    recipe = _recipe(artifacts=[], presets=[RecipePresetRef(preset_id="p1")])
    manager = _manager(recipe_catalog=_RecipeCatalog(recipe), model_repository=_RecipeModelRepo())
    manager.preset_collaborators.db_repo.is_preset_installed.return_value = False
    manager.preset_collaborators.preset_loader.load_preset_by_id.return_value = MagicMock(name="Preset One")
    manager.preset_collaborators.preset_loader.load_preset_by_id.return_value.name = "Preset One"
    report = _run(manager, _user(admin=True), recipe_id="r1")
    content = _by_area(report)["content"]
    assert content.status == NOT_READY
    assert content.code == "RECIPE_PRESET_NOT_INSTALLED"
    assert "Preset One" in content.message
    assert report.overall == NOT_READY


def test_format_missing_caps_at_three_names_then_counts_the_rest():
    text = _format_missing(["a", "b", "c", "d"])
    assert text == "a, b, c and 1 more"


def test_user_visibility_restricts_model_resolution():
    """A regular user's model resolution is scoped to their assigned ids."""
    manager = _manager()
    with patch(AVAILABILITY, return_value=[{"id": "m1"}]) as models_for_engine:
        asyncio.run(manager.evaluate(_user(admin=False)))
    # STRICT scoping: the user's own allowed ids are passed through, not None.
    _, kwargs = models_for_engine.call_args
    assert kwargs["user_allowed_model_ids"] == ["m1"]


def test_admin_model_resolution_is_unrestricted():
    manager = _manager()
    with patch(AVAILABILITY, return_value=[{"id": "m1"}]) as models_for_engine:
        asyncio.run(manager.evaluate(_user(admin=True)))
    _, kwargs = models_for_engine.call_args
    assert kwargs["user_allowed_model_ids"] is None


# --- generation-proven facet ------------------------------------------------

def test_no_completed_generation_is_not_ready():
    report = _run(_manager(completed=0), _user(admin=True))
    proven = _by_area(report)["generation_proven"]
    assert proven.status == NOT_READY
    assert proven.code == "NO_GENERATION_YET"
    assert report.overall == NOT_READY


# --- service facet ----------------------------------------------------------

def test_unreachable_db_is_not_ready():
    report = _run(_manager(instance_claim_repository=_FailClaimRepository()), _user(admin=True))
    service = _by_area(report)["service"]
    assert service.status == NOT_READY
    assert service.code == "DB_UNREACHABLE"
    assert report.overall == NOT_READY


def test_pending_migrations_is_degraded():
    report = _run(_manager(pending=True), _user(admin=True))
    service = _by_area(report)["service"]
    assert service.status == DEGRADED
    assert service.code == "MIGRATIONS_PENDING"


# --- role filtering ---------------------------------------------------------

def test_admin_sees_actions_and_internals():
    report = _run(_manager(backends={}), _user(admin=True))
    execution = _by_area(report)["execution"]
    assert execution.action is not None
    assert "Administration" in execution.action
    assert "backend" in execution.message.lower()


def test_user_gets_status_but_no_admin_action_or_internals():
    backends = {"b": _backend(name="secret-backend", status="degraded", reason="GPU index 3 does not exist")}
    manager = _manager(backends=backends)
    report = _run(manager, _user(admin=False))
    execution = _by_area(report)["execution"]
    # status is still reported to the user...
    assert execution.status == DEGRADED
    # ...but no admin-shaped action and no operator internals leak.
    assert execution.action is None
    assert "secret-backend" not in execution.message
    assert "GPU index 3" not in execution.message
    assert "administrator" in execution.message.lower()


def test_user_and_admin_agree_on_status_and_code():
    """Role filtering changes phrasing, never the verdict."""
    backends = {"b": _backend(status="degraded", reason="whatever")}
    admin_report = _run(_manager(backends=backends), _user(admin=True))
    user_report = _run(_manager(backends=backends), _user(admin=False))
    assert admin_report.overall == user_report.overall
    admin_areas = _by_area(admin_report)
    user_areas = _by_area(user_report)
    for area, admin_check in admin_areas.items():
        assert user_areas[area].status == admin_check.status
        assert user_areas[area].code == admin_check.code

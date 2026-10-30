import pytest

from src.features.models import indexer as indexer_module
from src.features.models.digest_queue_repository import DigestQueueRepository
from src.features.models.digests import DigestManager
from src.features.models.indexer import ModelScanner
from src.features.models.indexing_governor import IndexingGovernor
from src.features.models.read_throttle import ReadThrottle
from src.features.models.type_repository import ModelTypeRepository
from src.platform.database import get_cursor
from src.platform.runtime.model_headers import classify_header as real_classify_header
from src.platform.runtime.model_headers import read_header as real_read_header
from src.platform.settings.repository import SettingRepository
from tests.features.models.fakes import FakeClock, drain_manager
from tests.fixtures.model_index_fixtures import FLUX, LORA, Library, sha_of


@pytest.fixture
def library(tmp_path, mock_db):
    return Library(tmp_path)


@pytest.fixture
def pending(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", 1)
    lib = Library(tmp_path)
    governor = IndexingGovernor(throttle=ReadThrottle(clock=FakeClock().monotonic, sleep=FakeClock().sleep))
    lib.scanner = ModelScanner(lib.scanner.resolver, governor=governor)
    lib.manager = DigestManager(
        DigestQueueRepository(),
        governor,
        lib.scanner.location_path,
        locations_write_lock=lib.scanner.write_lock,
        on_applied=lambda ids: lib.scanner.settle_digested(set(ids)),
        setting_repository=SettingRepository(),
    )
    lib.scanner.attach_digests(lib.manager)
    return lib


def _verdict_rows():
    with get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS n FROM model_header_verdicts")
        return cursor.fetchone()["n"]


def _header_hashes(lib, filename):
    return {row["header_hash"] for row in lib.scanner.locations.list_for_model(lib.model(filename).id)}


def _same_header_other_bytes(lib, name, other):
    source = lib.put(lib.checkpoints, name, FLUX, tag="shared")
    twin = lib.checkpoints / other
    twin.write_bytes(source.read_bytes() + b"\x00" * 16)
    return source, twin


def test_a_new_file_has_its_verdict_as_soon_as_its_header_is_read_before_any_digest(pending):
    pending.put(pending.checkpoints, "flux.safetensors", FLUX)

    pending.index()

    model = pending.model("flux.safetensors")
    assert model.sha256 is None
    verdict = ModelTypeRepository().get_verdicts([model.id])[model.id]
    assert (verdict["status"], verdict["model_type"], verdict["family"]) == ("decided", "diffusion_model", "flux")
    assert len(_header_hashes(pending, "flux.safetensors") - {None}) == 1


def test_a_digest_landing_later_does_not_read_the_header_again(pending, monkeypatch):
    pending.put(pending.checkpoints, "flux.safetensors", FLUX)
    pending.index()
    reads = []
    monkeypatch.setattr(
        indexer_module, "read_header", lambda *args, **kwargs: reads.append(1) or real_read_header(*args, **kwargs)
    )

    drain_manager(pending.manager, max_jobs=5)

    assert pending.model("flux.safetensors").sha256
    assert reads == []
    assert ModelTypeRepository().get_verdicts([pending.model("flux.safetensors").id])


def test_two_files_with_the_same_header_share_one_verdict(library, monkeypatch):
    _same_header_other_bytes(library, "a.safetensors", "b.safetensors")
    classified = []
    monkeypatch.setattr(
        indexer_module, "classify_header", lambda view: classified.append(1) or real_classify_header(view)
    )

    library.index()

    first, second = library.model("a.safetensors"), library.model("b.safetensors")
    assert first.id != second.id and first.sha256 != second.sha256
    assert _header_hashes(library, "a.safetensors") == _header_hashes(library, "b.safetensors")
    verdicts = ModelTypeRepository().get_verdicts([first.id, second.id])
    assert {**verdicts[first.id], "model_id": None} == {**verdicts[second.id], "model_id": None}
    assert classified == [1]
    assert _verdict_rows() == 1


def test_files_with_different_headers_get_their_own_verdicts(library):
    library.put(library.checkpoints, "flux.safetensors", FLUX)
    library.put(library.checkpoints, "lora.safetensors", LORA)

    library.index()

    assert _verdict_rows() == 2
    assert _header_hashes(library, "flux.safetensors") != _header_hashes(library, "lora.safetensors")


def test_a_verdict_is_found_by_digest_through_the_model_that_holds_it(library):
    path = library.put(library.checkpoints, "flux.safetensors", FLUX)
    library.index()

    found = ModelTypeRepository().get_verdicts_by_sha([sha_of(path), "f" * 64])

    assert set(found) == {sha_of(path)}
    assert found[sha_of(path)]["family"] == "flux"


def test_a_digest_keyed_verdict_from_before_the_upgrade_still_settles_the_model(library, monkeypatch):
    path = library.put(library.checkpoints, "flux.safetensors", FLUX)
    library.index()
    model = library.model("flux.safetensors")
    with get_cursor() as cursor:
        cursor.execute("UPDATE model_header_verdicts SET sha256 = ?, header_hash = NULL", (sha_of(path),))
        cursor.execute("UPDATE model_locations SET header_hash = NULL")
    reads = []
    monkeypatch.setattr(
        indexer_module, "read_header", lambda *args, **kwargs: reads.append(1) or real_read_header(*args, **kwargs)
    )

    result = library.index()

    assert ModelTypeRepository().get_verdicts([model.id])[model.id]["family"] == "flux"
    assert (result["classified"], result["indexed"], reads) == (0, 0, [])
    assert library.scanner.count_unindexed()["total"] == 0


def test_a_file_whose_header_changed_is_reclassified_under_its_new_header(library):
    library.put(library.checkpoints, "thing.safetensors", LORA)
    library.index()
    before = _header_hashes(library, "thing.safetensors")
    library.put(library.checkpoints, "thing.safetensors", FLUX, tag="replaced")
    library.set_scan("checkpoint", True)

    library.index()

    assert _header_hashes(library, "thing.safetensors") != before
    model = library.model("thing.safetensors")
    assert ModelTypeRepository().get_verdicts([model.id])[model.id]["family"] == "flux"


def test_an_invalid_header_shares_the_invalid_verdict_with_its_twin_error(library):
    (library.checkpoints / "a.safetensors").write_bytes(b"\x01\x00\x00\x00\x00\x00\x00\x00{")
    (library.checkpoints / "b.safetensors").write_bytes(b"\x01\x00\x00\x00\x00\x00\x00\x00{" + b"\x00" * 9)

    library.index()

    models = [library.model("a.safetensors"), library.model("b.safetensors")]
    verdicts = ModelTypeRepository().get_verdicts([m.id for m in models])
    assert {verdicts[m.id]["status"] for m in models} == {"invalid"}

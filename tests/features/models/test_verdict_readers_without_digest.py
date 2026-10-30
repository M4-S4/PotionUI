import pytest

from src.features.models import indexer as indexer_module
from src.features.organize.core_facts import _base_model_sql
from src.features.organize.item_repository import OrganizeItemRepository
from src.platform.database import get_cursor
from tests.features.models.test_extractable_checkpoint_listing import all_in_one, listing, names
from tests.fixtures.model_index_fixtures import Library


@pytest.fixture
def pending(tmp_path, mock_db, monkeypatch):
    monkeypatch.setattr(indexer_module, "INLINE_HASH_LIMIT", 1)
    lib = Library(tmp_path)
    lib.put(lib.checkpoints, "flux1-dev-fp8.safetensors", all_in_one())
    lib.scanner.attach_digests(_Queue())
    lib.index()
    return lib


class _Queue:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _ids_matching_family(family):
    clause, params = _base_model_sql("is", family, "m")
    with get_cursor() as cursor:
        cursor.execute(f"SELECT m.id FROM models m WHERE {clause}", params)
        return {row["id"] for row in cursor.fetchall()}


def test_the_premise_holds_the_model_has_no_digest_yet(pending):
    assert pending.model("flux1-dev-fp8.safetensors").sha256 is None


def test_the_diffusion_model_picker_lists_an_extractable_checkpoint_that_has_no_digest_yet(pending):
    assert names(listing()) == ["flux1-dev-fp8.safetensors"]


def test_the_base_model_fact_matches_a_model_that_has_no_digest_yet(pending):
    model = pending.model("flux1-dev-fp8.safetensors")

    assert _ids_matching_family("flux") == {model.id}
    assert _ids_matching_family("sdxl") == set()


def test_the_family_options_include_a_model_that_has_no_digest_yet(pending):
    assert OrganizeItemRepository().model_families() == ["flux"]


def test_the_organize_item_carries_the_family_of_a_model_that_has_no_digest_yet(pending):
    model = pending.model("flux1-dev-fp8.safetensors")

    rows = OrganizeItemRepository().models("u1", [model.id], True, False)

    assert [row["family"] for row in rows] == ["flux"]

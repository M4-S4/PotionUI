import sqlite3

from src.features.models.locations_repository import ModelLocationsRepository
from src.features.models.records import Model
from src.features.models.repository import model_repo
from src.platform.filesystem.model_roots_repository import ModelRootRepository

SEEN_BEFORE = "2026-01-01T00:00:00"
SEEN_NOW = "2026-02-02T00:00:00"


def _binding() -> str:
    ModelRootRepository().insert_root("r_scan", "r_scan", "/models/r_scan", "r_scan", "library", False, False, SEEN_BEFORE)
    return ModelRootRepository().insert_binding("r_scan", "checkpoint", "checkpoints", 3000, False)


def _row(binding_id: str, rel_key: str, *, mtime_ns=10, status="present") -> None:
    model = model_repo.create(Model(filename=rel_key, file_size=1, sha256=rel_key, model_type="checkpoint"))
    ModelLocationsRepository().upsert(
        model_id=model.id, binding_id=binding_id, root_id="r_scan", model_type="checkpoint",
        rel_path=rel_key, rel_key=rel_key, size=1, mtime_ns=mtime_ns,
        sha256=rel_key, status=status, seen_at=SEEN_BEFORE,
    )


def _state(binding_id: str, rel_key: str):
    row = ModelLocationsRepository().get(binding_id, rel_key)
    return row["status"], row["mtime_ns"], row["seen_at"]


def test_confirm_present_touches_and_adopts_only_the_listed_rows(mock_db):
    binding_id = _binding()
    _row(binding_id, "touched.safetensors")
    _row(binding_id, "adopted.safetensors", mtime_ns=None)
    _row(binding_id, "other.safetensors")

    ModelLocationsRepository().confirm_present(
        binding_id, ["touched.safetensors"], [("adopted.safetensors", 77)], SEEN_NOW,
    )

    assert _state(binding_id, "touched.safetensors") == ("present", 10, SEEN_NOW)
    assert _state(binding_id, "adopted.safetensors") == ("present", 77, SEEN_NOW)
    assert _state(binding_id, "other.safetensors") == ("present", 10, SEEN_BEFORE)


def test_mark_missing_keeps_every_present_key_past_the_sql_variable_limit(mock_db):
    with mock_db.get_connection() as conn:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 32_766)
    binding_id = _binding()
    _row(binding_id, "present.safetensors")
    _row(binding_id, "absent.safetensors")
    _row(binding_id, "already-missing.safetensors", status="missing")
    present = [f"bulk/{n}.safetensors" for n in range(40_000)] + ["present.safetensors"]

    marked = ModelLocationsRepository().mark_missing_for_binding(binding_id, present, SEEN_NOW)

    assert marked == 1
    assert _state(binding_id, "present.safetensors") == ("present", 10, SEEN_BEFORE)
    assert _state(binding_id, "absent.safetensors") == ("missing", 10, SEEN_NOW)
    assert _state(binding_id, "already-missing.safetensors") == ("missing", 10, SEEN_BEFORE)


def test_mark_missing_with_no_present_keys_marks_the_whole_binding(mock_db):
    binding_id = _binding()
    _row(binding_id, "a.safetensors")
    _row(binding_id, "b.safetensors")

    assert ModelLocationsRepository().mark_missing_for_binding(binding_id, [], SEEN_NOW) == 2
    assert _state(binding_id, "a.safetensors")[0] == "missing"
    assert _state(binding_id, "b.safetensors")[0] == "missing"

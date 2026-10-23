import pytest

from src.features.models import locations_repository as locations_module
from src.features.models.digest_queue_repository import PRIORITY_BACKGROUND, DigestQueueRepository
from src.features.models.locations_repository import (
    DIGEST_CONFLICT,
    DIGEST_DUPLICATE,
    DIGEST_MERGED,
    DIGEST_RECORDED,
    DIGEST_STALE,
    ModelLocationsRepository,
)
from src.features.models.records import Model
from src.features.models.repository import model_repo
from src.platform.database import get_cursor
from tests.features.models.digest_support import DigestLibrary

STAMP = "2026-01-01T00:00:00"


@pytest.fixture
def library(tmp_path, mock_db):
    return DigestLibrary(tmp_path)


@pytest.fixture
def locations(mock_db):
    return ModelLocationsRepository()


@pytest.fixture
def queue(mock_db):
    return DigestQueueRepository()


def _model(item):
    return model_repo.get_by_id(item.model.id, include_providers=False, include_tags=False)


def _seed_dependents(model_id):
    with get_cursor() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO users (id, username, email, password_hash, account_type) "
            "VALUES ('u1', 'u1', 'u1@example.test', 'x', 'USER')"
        )
        cursor.execute("INSERT INTO tags (id, name) VALUES ('tag1', 'style')")
        cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 'tag1')", (model_id,))
        cursor.execute("INSERT INTO model_collections (id, name, user_id) VALUES ('col1', 'Mine', 'u1')")
        cursor.execute("INSERT INTO model_collection_members (collection_id, model_id) VALUES ('col1', ?)", (model_id,))
        cursor.execute("INSERT INTO user_models (id, user_id, model_id) VALUES ('um1', 'u1', ?)", (model_id,))
        cursor.execute("INSERT INTO user_model_meta (user_id, model_id, custom_name) VALUES ('u1', ?, 'Nick')", (model_id,))
        cursor.execute(
            "INSERT INTO user_model_attributes (user_id, model_id, key, value) VALUES ('u1', ?, 'k', 'v')", (model_id,)
        )
        cursor.execute("INSERT INTO user_groups (id, name) VALUES ('g1', 'Group')")
        cursor.execute("INSERT INTO user_group_models (id, group_id, model_id) VALUES ('ugm1', 'g1', ?)", (model_id,))
        cursor.execute("INSERT INTO providers (id, model_id, provider) VALUES ('p1', ?, 'civitai')", (model_id,))
        cursor.execute("INSERT INTO generations (id, form_data) VALUES ('gen1', '{}')")
        cursor.execute("INSERT INTO generation_models (id, generation_id, model_id) VALUES ('gm1', 'gen1', ?)", (model_id,))
        cursor.execute("INSERT INTO backends (id, name, engine) VALUES ('b1', 'B', 'native')")
        cursor.execute("INSERT INTO model_availability (id, model_id, backend_id, ref) VALUES ('av1', ?, 'b1', 'x')", (model_id,))
        cursor.execute(
            "INSERT INTO model_preset_scopes (model_id, preset_id, created_at) VALUES (?, 'preset1', '2026-01-01')",
            (model_id,),
        )
        cursor.execute("INSERT INTO files (id, file_path, file_type) VALUES ('f1', 'a.png', 'image')")
        cursor.execute("INSERT INTO model_files (id, model_id, file_id) VALUES ('mf1', ?, 'f1')", (model_id,))
        cursor.execute(
            "INSERT INTO model_preview_media (id, model_id, url, type) VALUES ('pm1', ?, 'http://x/y.png', 'image')",
            (model_id,),
        )
        cursor.execute(
            "INSERT INTO enhancement_feedback (id, user_id, session_id, message_id, prompt_text, verdict, model_id) "
            "VALUES ('ef1', 'u1', 's', 'm', 'p', 'up', ?)", (model_id,),
        )
        cursor.execute(
            "INSERT INTO generation_costs (id, generation_id, model_id, source, created_at) "
            "VALUES ('gc1', 'gen1', ?, 'x', '2026-01-01')", (model_id,),
        )
    return [
        ("model_tags", "model_id"), ("model_collection_members", "model_id"),
        ("user_models", "model_id"), ("user_model_meta", "model_id"),
        ("user_model_attributes", "model_id"), ("user_group_models", "model_id"),
        ("providers", "model_id"), ("generation_models", "model_id"),
        ("model_availability", "model_id"), ("model_preset_scopes", "model_id"),
        ("model_files", "model_id"), ("model_preview_media", "model_id"),
        ("enhancement_feedback", "model_id"), ("generation_costs", "model_id"),
    ]


def _count(table, column, model_id):
    with get_cursor() as cursor:
        cursor.execute(f'SELECT COUNT(*) AS n FROM "{table}" WHERE "{column}" = ?', (model_id,))
        return cursor.fetchone()["n"]


def _queued(queue, item):
    return queue.get(item.location_id) is not None


class TestGetById:
    def test_returns_the_location_with_its_signature(self, library, locations):
        item = library.add("a.safetensors", b"a" * 10)
        library.set_digest(item, "d" * 64, "q" * 64)

        row = locations.get_by_id(item.location_id)

        assert (row["id"], row["model_id"], row["sha256"], row["quick_sig"], row["status"]) == (
            item.location_id, item.model.id, "d" * 64, "q" * 64, "present",
        )

    def test_an_unknown_location_is_none(self, locations):
        assert locations.get_by_id("missing") is None


class TestUpsertAndSizeTwins:
    def _upsert(self, library, item, **overrides):
        values = dict(
            model_id=item.model.id, binding_id=library.binding.binding_id, root_id="r_home", model_type="lora",
            rel_path=item.path.name, rel_key=item.path.name, size=10, mtime_ns=1, sha256=None, status="present",
            seen_at=STAMP,
        )
        values.update(overrides)
        return ModelLocationsRepository().upsert(**values)

    def test_the_signature_is_stored_and_a_rewrite_of_the_location_clears_it(self, library, locations):
        item = library.add("a.safetensors", b"a" * 10)
        location_id = self._upsert(library, item, sha256="d" * 64, quick_sig="q" * 64)
        assert locations.get_by_id(location_id)["quick_sig"] == "q" * 64

        self._upsert(library, item, size=11, sha256=None)

        row = locations.get_by_id(location_id)
        assert (row["size"], row["sha256"], row["quick_sig"]) == (11, None, None)

    def test_missing_locations_of_one_size_are_listed_only_when_they_hold_a_digest(self, library, locations):
        gone = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        library.set_digest(gone, "d" * 64, "q" * 64)
        library.set_status(gone, "missing")
        never_hashed = library.add("b.safetensors", b"b" * 10)
        library.set_status(never_hashed, "missing")
        here = library.add("c.safetensors", b"c" * 10, sha256="e" * 64)
        other_size = library.add("d.safetensors", b"d" * 11, sha256="f" * 64)
        library.set_status(other_size, "missing")

        rows = locations.list_missing_digested_by_size(10)

        assert [(row["id"], row["quick_sig"]) for row in rows] == [(gone.location_id, "q" * 64)]
        assert here.location_id not in [row["id"] for row in rows]

    def test_present_locations_of_one_size_are_listed_with_their_signature(self, library, locations):
        first = library.add("a.safetensors", b"a" * 10)
        second = library.add("b.safetensors", b"b" * 10)
        library.add("c.safetensors", b"c" * 11)
        gone = library.add("gone.safetensors", b"d" * 10)
        library.set_status(gone, "missing")
        library.set_digest(second, "e" * 64, "q" * 64)

        rows = locations.list_present_by_size(10)

        assert sorted((row["id"], row["quick_sig"]) for row in rows) == sorted(
            [(first.location_id, None), (second.location_id, "q" * 64)]
        )
        assert locations.list_present_by_size(99) == []


class TestApplyDigest:
    def test_one_call_records_the_location_the_model_and_leaves_the_queue(self, library, locations, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, "q" * 64, 99)

        assert (result.outcome, result.model_id) == (DIGEST_RECORDED, item.model.id)
        row = locations.get_by_id(item.location_id)
        assert (row["sha256"], row["quick_sig"]) == ("d" * 64, "q" * 64)
        model = _model(item)
        assert (model.sha256, model.file_size) == ("d" * 64, 99)
        assert not _queued(queue, item)

    def test_a_location_that_is_gone_or_already_digested_only_leaves_the_queue(self, library, locations, queue):
        gone = library.add("gone.safetensors", b"g" * 10)
        done = library.add("done.safetensors", b"d" * 10)
        queue.enqueue([gone.location_id, done.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        library.set_status(gone, "missing")
        library.set_digest(done, "e" * 64)

        first = locations.apply_digest_and_dequeue(gone.location_id, "d" * 64, None, 10)
        second = locations.apply_digest_and_dequeue(done.location_id, "d" * 64, None, 10)

        assert (first.outcome, second.outcome) == (DIGEST_STALE, DIGEST_STALE)
        assert locations.get_by_id(done.location_id)["sha256"] == "e" * 64
        assert _model(gone).sha256 is None and _model(done).sha256 is None
        assert not _queued(queue, gone) and not _queued(queue, done)

    def test_an_unknown_location_is_stale(self, locations):
        assert locations.apply_digest_and_dequeue("nope", "d" * 64, None, 1).outcome == DIGEST_STALE

    def test_a_model_that_already_has_the_digest_is_left_as_it_is(self, library, locations):
        item = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        library.set_digest(item, None)

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, None, 99)

        assert result.outcome == DIGEST_RECORDED
        assert _model(item).file_size == 10

    def test_a_copy_that_differs_from_the_models_digest_becomes_a_conflict(self, library, locations):
        first = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        copy = library.add("a.safetensors", b"b" * 10, rel_path="copies/a.safetensors", model=first.model)

        result = locations.apply_digest_and_dequeue(copy.location_id, "e" * 64, "q" * 64, 10)

        assert (result.outcome, result.model_id, result.model_filename) == (DIGEST_CONFLICT, first.model.id, "a.safetensors")
        row = locations.get_by_id(copy.location_id)
        assert (row["status"], row["sha256"]) == ("conflict", "e" * 64)
        assert _model(first).sha256 == "d" * 64

    def test_the_only_copy_replaces_the_models_old_digest(self, library, locations):
        item = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        library.set_digest(item, None)

        result = locations.apply_digest_and_dequeue(item.location_id, "e" * 64, None, 77)

        assert result.outcome == DIGEST_RECORDED
        model = _model(item)
        assert (model.sha256, model.file_size) == ("e" * 64, 77)

    def test_a_digest_another_model_owns_is_reported_and_everything_else_still_commits(self, library, locations, queue):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        twin = library.add("twin.safetensors", b"o" * 10)
        queue.enqueue([twin.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        result = locations.apply_digest_and_dequeue(twin.location_id, "d" * 64, "q" * 64, 10)

        assert result.outcome == DIGEST_DUPLICATE
        assert (result.model_id, result.model_filename) == (twin.model.id, "twin.safetensors")
        assert (result.owner_model_id, result.owner_filename) == (owner.model.id, "owner.safetensors")
        assert _model(twin).sha256 is None
        assert _model(owner).sha256 == "d" * 64
        row = locations.get_by_id(twin.location_id)
        assert (row["sha256"], row["quick_sig"]) == ("d" * 64, "q" * 64)
        assert not _queued(queue, twin)

    def test_a_replacement_that_collides_with_another_model_is_a_duplicate_too(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        item = library.add("item.safetensors", b"i" * 10, sha256="e" * 64)
        library.set_digest(item, None)

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, None, 10)

        assert (result.outcome, result.owner_model_id) == (DIGEST_DUPLICATE, owner.model.id)
        assert _model(item).sha256 == "e" * 64

    def test_a_digest_whose_owner_has_no_present_copy_joins_the_new_file_to_the_owner(self, library, locations, queue):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        model_repo.mark_unavailable(owner.model.id)
        renamed = library.add("renamed.safetensors", b"o" * 10)
        queue.enqueue([renamed.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        result = locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, "q" * 64, 10)

        assert result.outcome == DIGEST_MERGED
        assert (result.model_id, result.model_filename, result.merged_from) == (
            owner.model.id, "renamed.safetensors", renamed.model.id,
        )
        survivor = _model(owner)
        assert (survivor.filename, survivor.sha256, survivor.is_available, survivor.file_size) == (
            "renamed.safetensors", "d" * 64, True, 10,
        )
        assert model_repo.get_by_id(renamed.model.id, include_providers=False) is None
        row = locations.get_by_id(renamed.location_id)
        assert (row["model_id"], row["status"], row["sha256"], row["quick_sig"]) == (
            owner.model.id, "present", "d" * 64, "q" * 64,
        )
        assert not _queued(queue, renamed)

    def test_the_owners_tags_and_notes_survive_the_join_and_the_new_models_go_away(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        with get_cursor() as cursor:
            cursor.execute("INSERT INTO tags (id, name) VALUES ('t1', 'style')")
            cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 't1')", (owner.model.id,))
            cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 't1')", (renamed.model.id,))

        locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, None, 10)

        with get_cursor() as cursor:
            cursor.execute("SELECT model_id FROM model_tags")
            assert [row["model_id"] for row in cursor.fetchall()] == [owner.model.id]

    def test_everything_hanging_off_the_new_model_moves_to_the_owner(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        seeded = _seed_dependents(renamed.model.id)

        result = locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, None, 10)

        assert result.outcome == DIGEST_MERGED
        for table, column in seeded:
            assert _count(table, column, owner.model.id) == 1, f"{table} did not reach the owner"
            assert _count(table, column, renamed.model.id) == 0, f"{table} was left behind"

    def test_a_column_named_differently_that_references_a_model_moves_too(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        with get_cursor() as cursor:
            cursor.execute(
                "CREATE TABLE zz_refs (id TEXT PRIMARY KEY, source_model TEXT REFERENCES models(id) ON DELETE CASCADE)"
            )
            cursor.execute("INSERT INTO zz_refs (id, source_model) VALUES ('r1', ?)", (renamed.model.id,))

        locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, None, 10)

        assert _count("zz_refs", "source_model", owner.model.id) == 1

    def test_a_dependent_the_owner_already_has_is_not_duplicated_and_the_owners_wins(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        with get_cursor() as cursor:
            cursor.execute("INSERT INTO tags (id, name) VALUES ('same', 'same')")
            cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 'same')", (owner.model.id,))
            cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 'same')", (renamed.model.id,))
            cursor.execute("INSERT INTO tags (id, name) VALUES ('extra', 'extra')")
            cursor.execute("INSERT INTO model_tags (model_id, tag_id) VALUES (?, 'extra')", (renamed.model.id,))

        locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, None, 10)

        with get_cursor() as cursor:
            cursor.execute("SELECT tag_id FROM model_tags WHERE model_id = ? ORDER BY tag_id", (owner.model.id,))
            assert [row["tag_id"] for row in cursor.fetchall()] == ["extra", "same"]
            cursor.execute("SELECT COUNT(*) AS n FROM model_tags")
            assert cursor.fetchone()["n"] == 2

    def test_every_table_that_points_at_a_model_is_known_to_the_join(self, mock_db):
        with get_cursor() as cursor:
            references = set(model_repo._model_references(cursor))

        assert {
            ("model_tags", "model_id"), ("model_collection_members", "model_id"), ("user_models", "model_id"),
            ("user_model_meta", "model_id"), ("user_model_attributes", "model_id"), ("user_group_models", "model_id"),
            ("providers", "model_id"), ("generation_models", "model_id"), ("model_files", "model_id"),
            ("model_availability", "model_id"), ("model_preset_scopes", "model_id"),
            ("model_preview_media", "model_id"), ("prompts", "model_id"), ("model_locations", "model_id"),
            ("generation_costs", "model_id"), ("enhancement_feedback", "model_id"),
        } <= references
        assert not any(table == "models" for table, _ in references)

    def test_a_join_that_fails_part_way_commits_nothing(self, library, locations, queue, monkeypatch):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        seeded = _seed_dependents(renamed.model.id)
        queue.enqueue([renamed.location_id], PRIORITY_BACKGROUND, "scan", STAMP)

        def explode(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(model_repo, "_rename_absorbed", explode)

        with pytest.raises(RuntimeError):
            locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, "q" * 64, 10)

        monkeypatch.undo()
        assert model_repo.get_by_id(renamed.model.id, include_providers=False) is not None
        assert _model(owner).filename == "owner.safetensors"
        row = locations.get_by_id(renamed.location_id)
        assert (row["model_id"], row["sha256"], row["quick_sig"]) == (renamed.model.id, None, None)
        assert _queued(queue, renamed)
        for table, column in seeded:
            assert _count(table, column, renamed.model.id) == 1
            assert _count(table, column, owner.model.id) == 0

    def test_a_model_that_already_has_a_different_digest_is_never_joined(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        item = library.add("renamed.safetensors", b"o" * 10, sha256="e" * 64)
        library.set_digest(item, None)

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, None, 10)

        assert result.outcome == DIGEST_DUPLICATE
        assert _model(item).sha256 == "e" * 64
        assert _model(owner).filename == "owner.safetensors"
        assert locations.get_by_id(item.location_id)["model_id"] == item.model.id

    def test_an_owner_with_a_present_copy_is_never_joined(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        twin = library.add("twin.safetensors", b"o" * 10)

        result = locations.apply_digest_and_dequeue(twin.location_id, "d" * 64, None, 10)

        assert result.outcome == DIGEST_DUPLICATE
        assert model_repo.get_by_id(twin.model.id, include_providers=False) is not None
        assert locations.get_by_id(twin.location_id)["model_id"] == twin.model.id
        assert _model(owner).filename == "owner.safetensors"

    def test_a_new_model_with_more_than_one_location_is_never_joined(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        renamed = library.add("renamed.safetensors", b"o" * 10)
        library.add("copy.safetensors", b"x" * 11, model=renamed.model)

        result = locations.apply_digest_and_dequeue(renamed.location_id, "d" * 64, None, 10)

        assert result.outcome == DIGEST_DUPLICATE
        assert _model(owner).filename == "owner.safetensors"
        assert model_repo.get_by_id(renamed.model.id, include_providers=False) is not None

    def test_a_filename_the_owners_type_already_uses_blocks_the_join(self, library, locations):
        owner = library.add("owner.safetensors", b"o" * 10, sha256="d" * 64)
        library.set_status(owner, "missing")
        taken = model_repo.create(Model(filename="renamed.safetensors", file_size=1, sha256="e" * 64, model_type="lora"))
        renamed = model_repo.create(Model(filename="renamed.safetensors", file_size=10, sha256=None, model_type="vae"))
        item = library.add("renamed.safetensors", b"o" * 10, rel_path="v/renamed.safetensors", model=renamed)

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, None, 10)

        assert result.outcome == DIGEST_DUPLICATE
        assert _model(owner).filename == "owner.safetensors"
        assert model_repo.get_by_id(taken.id, include_providers=False) is not None

    def test_a_failure_in_the_model_step_commits_nothing_and_a_retry_applies(self, library, locations, queue, monkeypatch):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        real = model_repo.update_digest
        calls = []

        def failing_once(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("disk I/O error")
            return real(*args, **kwargs)

        monkeypatch.setattr(locations_module.model_repo, "update_digest", failing_once)

        with pytest.raises(RuntimeError):
            locations.apply_digest_and_dequeue(item.location_id, "d" * 64, "q" * 64, 10)

        assert locations.get_by_id(item.location_id)["sha256"] is None
        assert _model(item).sha256 is None
        assert _queued(queue, item)
        assert [row["location_id"] for row in queue.pending_locations_for_model(item.model.id)] == [item.location_id]

        result = locations.apply_digest_and_dequeue(item.location_id, "d" * 64, "q" * 64, 10)

        assert result.outcome == DIGEST_RECORDED
        assert locations.get_by_id(item.location_id)["sha256"] == "d" * 64
        assert _model(item).sha256 == "d" * 64
        assert not _queued(queue, item)

    def test_a_failure_while_marking_the_conflict_commits_nothing(self, library, locations, queue, monkeypatch):
        first = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        copy = library.add("a.safetensors", b"b" * 10, rel_path="copies/a.safetensors", model=first.model)
        queue.enqueue([copy.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        with get_cursor() as cursor:
            cursor.execute(
                "CREATE TRIGGER block_conflict BEFORE UPDATE OF status ON model_locations "
                "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
            )

        with pytest.raises(Exception):
            locations.apply_digest_and_dequeue(copy.location_id, "e" * 64, None, 10)

        row = locations.get_by_id(copy.location_id)
        assert (row["sha256"], row["status"]) == (None, "present")
        assert _queued(queue, copy)


    def test_a_failure_while_leaving_the_queue_undoes_the_conflict_and_the_digest(self, library, locations, queue):
        first = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        copy = library.add("a.safetensors", b"b" * 10, rel_path="copies/a.safetensors", model=first.model)
        queue.enqueue([copy.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        with get_cursor() as cursor:
            cursor.execute(
                "CREATE TRIGGER block_dequeue BEFORE DELETE ON model_digest_queue "
                "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
            )

        with pytest.raises(Exception):
            locations.apply_digest_and_dequeue(copy.location_id, "e" * 64, "q" * 64, 10)

        row = locations.get_by_id(copy.location_id)
        assert (row["sha256"], row["quick_sig"], row["status"]) == (None, None, "present")
        assert _queued(queue, copy)

    def test_a_failure_while_leaving_the_queue_undoes_the_model_digest_too(self, library, locations, queue):
        item = library.add("a.safetensors", b"a" * 10)
        queue.enqueue([item.location_id], PRIORITY_BACKGROUND, "scan", STAMP)
        with get_cursor() as cursor:
            cursor.execute(
                "CREATE TRIGGER block_dequeue BEFORE DELETE ON model_digest_queue "
                "BEGIN SELECT RAISE(ABORT, 'blocked'); END"
            )

        with pytest.raises(Exception):
            locations.apply_digest_and_dequeue(item.location_id, "d" * 64, None, 10)

        assert locations.get_by_id(item.location_id)["sha256"] is None
        assert _model(item).sha256 is None
        assert _queued(queue, item)


class TestUpdateDigest:
    def test_sets_the_digest_and_the_size(self, library):
        item = library.add("a.safetensors", b"a" * 10)

        assert model_repo.update_digest(item.model.id, sha256="d" * 64, file_size=99) is True

        model = _model(item)
        assert (model.sha256, model.file_size) == ("d" * 64, 99)

    def test_an_unknown_size_keeps_the_stored_one(self, library):
        item = library.add("a.safetensors", b"a" * 10)

        model_repo.update_digest(item.model.id, sha256="d" * 64, file_size=None)

        assert _model(item).file_size == 10

    def test_a_digest_another_model_owns_is_refused(self, library):
        first = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        second = library.add("b.safetensors", b"b" * 10)

        assert model_repo.update_digest(second.model.id, sha256="d" * 64, file_size=10) is False

        assert _model(second).sha256 is None
        assert _model(first).sha256 == "d" * 64

    def test_an_unknown_model_is_refused(self, mock_db):
        assert model_repo.update_digest("nope", sha256="d" * 64, file_size=1) is False

    def test_a_refused_digest_keeps_the_earlier_writes_of_a_shared_transaction(self, library):
        owner = library.add("a.safetensors", b"a" * 10, sha256="d" * 64)
        other = library.add("b.safetensors", b"b" * 10)

        with get_cursor() as cursor:
            cursor.execute("UPDATE models SET description = 'kept' WHERE id = ?", (other.model.id,))
            refused = model_repo.update_digest(other.model.id, sha256="d" * 64, file_size=10, cursor=cursor)
            cursor.execute("UPDATE models SET file_size = 55 WHERE id = ?", (other.model.id,))

        assert refused is False
        model = _model(other)
        assert (model.description, model.file_size, model.sha256) == ("kept", 55, None)
        assert _model(owner).sha256 == "d" * 64

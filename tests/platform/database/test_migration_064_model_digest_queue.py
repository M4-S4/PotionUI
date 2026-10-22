import shutil
from unittest.mock import patch

import pytest

from src.platform.database.database import Database
from tests.fixtures.persistence_base import make_lean_file_database
from tests.platform.database._support import load_migration

_PREREQUISITES = (
    "001_baseline",
    "035_model_roots",
    "036_native_availability_logical_refs",
    "037_drop_model_file_path",
    "042_model_type_classification",
    "043_model_type_assertion_sources",
    "044_model_root_binding_ids",
)

_QUEUE_COLUMNS = {
    "location_id", "priority", "reason", "size", "attempts", "last_error", "not_before", "queued_at",
}


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    repo_root = tmp_path_factory.mktemp("pre064-root")
    path = tmp_path_factory.mktemp("pre064-db") / "pre064.sqlite"
    Database._instance = None
    database = Database()
    database.db_path = path
    database._initialized = True
    make_lean_file_database(database)
    with patch("pathlib.Path.cwd", return_value=repo_root):
        for stem in _PREREQUISITES:
            load_migration(stem, database).up()
    with database.get_connection() as conn:
        conn.execute("PRAGMA journal_mode = DELETE").close()
    Database._instance = None
    return path


@pytest.fixture
def env(template, tmp_path):
    Database._instance = None
    database = Database()
    database.db_path = tmp_path / "test.sqlite"
    database._initialized = True
    shutil.copyfile(template, database.db_path)
    make_lean_file_database(database)
    with patch("pathlib.Path.cwd", return_value=tmp_path):
        yield database, load_migration("064_model_digest_queue", database)
    Database._instance = None


def _seed_library(database, digested):
    with database.get_cursor() as cursor:
        cursor.execute("DELETE FROM model_locations")
        cursor.execute("DELETE FROM model_root_bindings")
        cursor.execute("DELETE FROM model_roots")
        cursor.execute(
            "INSERT INTO model_roots (id, label, path, path_key, kind, read_only, case_insensitive, state) "
            "VALUES ('r1', 'r1', '/lib', '/lib', 'library', 0, 0, 'online')"
        )
        cursor.execute(
            "INSERT INTO model_root_bindings (id, root_id, model_type, subdir, subdir_key, position, is_write, scan_headers) "
            "VALUES ('b1', 'r1', 'lora', 'loras', 'loras', 0, 1, 0)"
        )
        cursor.execute(
            "INSERT INTO models (id, filename, file_size, sha256, model_type, is_available, is_directory) "
            "VALUES ('m1', 'a.safetensors', 5, ?, 'lora', 1, 0)",
            ("a" * 64 if digested else None,),
        )
        cursor.execute(
            "INSERT INTO model_locations "
            "(id, model_id, root_id, model_type, rel_path, rel_key, size, mtime_ns, sha256, status, seen_at, binding_id) "
            "VALUES ('l1', 'm1', 'r1', 'lora', 'a.safetensors', 'a.safetensors', 5, 11, ?, 'present', '2026-01-01', 'b1')",
            ("a" * 64 if digested else None,),
        )


def _columns(database, table):
    with database.get_cursor() as cursor:
        cursor.execute(f"PRAGMA table_info({table})")
        return {row["name"]: row for row in cursor.fetchall()}


def _index_names(database):
    with database.get_cursor() as cursor:
        cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        return {row["name"] for row in cursor.fetchall()}


def _settings(database):
    with database.get_cursor() as cursor:
        cursor.execute(
            "SELECT key, value, value_type, type FROM settings "
            "WHERE key IN ('model_indexing_level', 'model_indexing_paused', 'model_digest_policy') ORDER BY key"
        )
        return [tuple(row) for row in cursor.fetchall()]


def test_adds_a_nullable_signature_column_and_keeps_existing_locations(env):
    database, migration = env
    _seed_library(database, digested=True)

    migration.up()

    columns = _columns(database, "model_locations")
    assert columns["quick_sig"]["notnull"] == 0
    with database.get_cursor() as cursor:
        cursor.execute("SELECT id, sha256, quick_sig, size, mtime_ns FROM model_locations")
        assert [tuple(row) for row in cursor.fetchall()] == [("l1", "a" * 64, None, 5, 11)]


def test_creates_the_queue_table_with_its_columns(env):
    database, migration = env

    migration.up()

    columns = _columns(database, "model_digest_queue")
    assert set(columns) == _QUEUE_COLUMNS
    assert columns["location_id"]["pk"] == 1
    assert columns["priority"]["dflt_value"] == "0"
    assert columns["attempts"]["dflt_value"] == "0"
    assert columns["not_before"]["notnull"] == 0


def test_creates_the_size_and_queue_order_indexes(env):
    database, migration = env

    migration.up()

    assert {"idx_model_locations_size", "idx_models_file_size", "idx_model_digest_queue_order"} <= _index_names(database)
    with database.get_cursor() as cursor:
        cursor.execute("PRAGMA index_xinfo('idx_model_digest_queue_order')")
        keys = [(row["name"], row["desc"]) for row in cursor.fetchall() if row["key"] == 1]
    assert keys == [("priority", 1), ("size", 0)]


def test_a_queue_row_goes_away_with_its_location(env):
    database, migration = env
    _seed_library(database, digested=False)
    migration.up()
    with database.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO model_digest_queue (location_id, reason, size, queued_at) VALUES ('l1', 'scan', 5, '2026-01-01')"
        )

    with database.get_cursor() as cursor:
        cursor.execute("DELETE FROM model_locations WHERE id = 'l1'")
        cursor.execute("SELECT COUNT(*) AS n FROM model_digest_queue")
        assert cursor.fetchone()["n"] == 0


def test_a_fresh_install_starts_with_low_level_unpaused_and_an_unset_policy(env):
    database, migration = env

    migration.up()

    assert _settings(database) == [
        ("model_digest_policy", "unset", "string", "SYSTEM"),
        ("model_indexing_level", "low", "string", "SYSTEM"),
        ("model_indexing_paused", "false", "boolean", "SYSTEM"),
    ]


def test_an_install_that_already_has_digested_models_keeps_hashing_in_the_background(env):
    database, migration = env
    _seed_library(database, digested=True)

    migration.up()

    assert dict((key, value) for key, value, _, _ in _settings(database))["model_digest_policy"] == "background"


def test_an_install_with_only_undigested_models_is_still_unset(env):
    database, migration = env
    _seed_library(database, digested=False)

    migration.up()

    assert dict((key, value) for key, value, _, _ in _settings(database))["model_digest_policy"] == "unset"


def test_existing_setting_values_are_left_alone(env):
    database, migration = env
    with database.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO settings (id, key, value, value_type, description, type) "
            "VALUES ('s1', 'model_indexing_level', 'max', 'string', 'created earlier', 'SYSTEM')"
        )

    migration.up()

    assert dict((key, value) for key, value, _, _ in _settings(database))["model_indexing_level"] == "max"


def test_running_twice_changes_nothing(env):
    database, migration = env
    _seed_library(database, digested=True)
    migration.up()
    with database.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO model_digest_queue (location_id, reason, size, queued_at) VALUES ('l1', 'scan', 5, '2026-01-01')"
        )
        cursor.execute("UPDATE settings SET value = 'normal' WHERE key = 'model_indexing_level'")

    migration.up()

    assert dict((key, value) for key, value, _, _ in _settings(database))["model_indexing_level"] == "normal"
    with database.get_cursor() as cursor:
        cursor.execute("SELECT COUNT(*) AS n FROM model_digest_queue")
        assert cursor.fetchone()["n"] == 1

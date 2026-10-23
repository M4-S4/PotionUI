import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.platform.database.database import Database
from src.platform.util.ids import generate_ulid
from tests.fixtures.persistence_base import make_lean_file_database
from tests.platform.database._support import load_migration

_pre_037_template = None


def _pre_037_template_path(repo_root):
    global _pre_037_template
    if _pre_037_template is None:
        directory = Path(tempfile.mkdtemp(prefix="potionui-pre037-"))
        path = directory / "pre037.sqlite"
        Database._instance = None
        database = Database()
        database.db_path = path
        database._initialized = True
        make_lean_file_database(database)
        with patch("pathlib.Path.cwd", return_value=repo_root):
            load_migration("001_baseline", database).up()
            load_migration("035_model_roots", database).up()
            load_migration("036_native_availability_logical_refs", database).up()
        with database.get_connection() as conn:
            conn.execute("PRAGMA journal_mode = DELETE").close()
        Database._instance = None
        _pre_037_template = path
    return _pre_037_template


class TestMigration037DropModelFilePath(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.repo_root = Path(tempfile.mkdtemp())
        template = _pre_037_template_path(self.repo_root)
        Database._instance = None
        self.db = Database()
        self.db.db_path = Path(self.temp_dir) / "test.sqlite"
        self.db._initialized = True
        shutil.copyfile(template, self.db.db_path)
        make_lean_file_database(self.db)

        self.cwd_patch = patch("pathlib.Path.cwd", return_value=self.repo_root)
        self.cwd_patch.start()

        self.migration = load_migration("037_drop_model_file_path", self.db)

    def tearDown(self):
        self.cwd_patch.stop()
        Database._instance = None

    def _cursor(self):
        return self.db.get_cursor()

    def _has_column(self, table, column) -> bool:
        with self._cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return any(row["name"] == column for row in cursor.fetchall())

    def _insert_model(self, model_type, filename, file_path=None):
        model_id = generate_ulid()
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO models (id, filename, file_path, model_type) VALUES (?, ?, ?, ?)",
                (model_id, filename, file_path, model_type),
            )
        return model_id

    def _insert_location(self, model_id, model_type, rel_path, *, root_id="home", status="present"):
        with self._cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO model_locations (id, model_id, root_id, model_type, rel_path, rel_key, status, seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (generate_ulid(), model_id, root_id, model_type, rel_path, rel_path, status),
            )

    def test_up_drops_the_column_and_its_index(self):
        self._insert_model("lora", "a.safetensors", "models/loras/a.safetensors")

        self.migration.up()

        self.assertFalse(self._has_column("models", "file_path"))
        with self._cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE name = 'idx_models_file_path'")
            self.assertIsNone(cursor.fetchone())

    def test_up_leaves_models_usable_without_file_path(self):
        self.migration.up()

        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO models (id, filename, model_type) VALUES (?, ?, ?)",
                (generate_ulid(), "b.safetensors", "lora"),
            )

    def test_up_is_idempotent_on_a_second_run(self):
        self._insert_model("lora", "c.safetensors", "models/loras/c.safetensors")

        self.migration.up()
        self.migration.up()

        self.assertFalse(self._has_column("models", "file_path"))


if __name__ == "__main__":
    unittest.main()

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from src.platform.database.database import Database

_MIGRATIONS = (
    Path(__file__).resolve().parents[3]
    / "src" / "platform" / "database" / "migrations"
)

_EXPECTED = {
    "idx_generations_created_id": [("created_at", 1), ("id", 1)],
    "idx_generations_status_created_id": [("status", 0), ("created_at", 1), ("id", 1)],
    "idx_generations_user_mode": [("user_id", 0), ("mode", 0), ("created_at", 1), ("id", 1)],
    "idx_generations_user_preset": [("user_id", 0), ("preset_id", 0), ("created_at", 1), ("id", 1)],
}


def _load_migration(stem, database):
    spec = importlib.util.spec_from_file_location(stem, _MIGRATIONS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    module.db = database
    return module


class TestMigration058HistoryIndexes(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Database._instance = None
        self.db = Database()
        self.db.db_path = Path(temp.name) / "test.sqlite"
        self.db._initialized = True
        _load_migration("001_baseline", self.db).up()
        _load_migration("014_generations_user_created_index", self.db).up()
        self.migration = _load_migration("058_history_indexes", self.db)

    def tearDown(self):
        Database._instance = None

    def _key_columns(self, index_name):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA index_xinfo('{index_name}')")
            return [(row["name"], row["desc"]) for row in cursor.fetchall() if row["key"] == 1]

    def _index_names(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'generations'")
            return {row["name"] for row in cursor.fetchall()}

    def test_creates_every_index_with_its_key_order(self):
        self.migration.up()

        for name, columns in _EXPECTED.items():
            self.assertEqual(self._key_columns(name), columns, name)

    def test_running_twice_is_a_no_op(self):
        self.migration.up()
        before = self._index_names()

        self.migration.up()

        self.assertEqual(self._index_names(), before)

    def test_down_drops_only_the_added_indexes(self):
        before = self._index_names()
        self.migration.up()

        self.migration.down()

        self.assertEqual(self._index_names(), before)
        self.assertIn("idx_generations_user_created_id", before)


if __name__ == "__main__":
    unittest.main()

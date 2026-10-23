import unittest

from tests.platform.database._support import MigrationCase

_EXPECTED = {
    "idx_generations_created_id": [("created_at", 1), ("id", 1)],
    "idx_generations_status_created_id": [("status", 0), ("created_at", 1), ("id", 1)],
    "idx_generations_user_mode": [("user_id", 0), ("mode", 0), ("created_at", 1), ("id", 1)],
    "idx_generations_user_preset": [("user_id", 0), ("preset_id", 0), ("created_at", 1), ("id", 1)],
}


class TestMigration058HistoryIndexes(MigrationCase):
    MIGRATION = "058_history_indexes"
    PREREQUISITES = ("001_baseline", "014_generations_user_created_index")

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


if __name__ == "__main__":
    unittest.main()

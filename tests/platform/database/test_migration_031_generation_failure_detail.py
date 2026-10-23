import unittest

from tests.platform.database._support import MigrationCase

_COLUMNS = {
    "error_code",
    "error_user_message",
    "error_detail",
    "failed_pipe_id",
    "failed_pipe_name",
    "failed_at_step",
}


class TestMigration031GenerationFailureDetail(MigrationCase):
    MIGRATION = "031_generation_failure_detail"

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(generations)")
            return {row[1] for row in cursor.fetchall()}

    def test_failure_columns_land_on_generations(self):
        self.assertFalse(_COLUMNS & self._columns())

        self.migration.up()

        self.assertTrue(_COLUMNS <= self._columns())

    def test_existing_rows_read_back_with_no_failure_detail(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, form_data, status, error_message) "
                "VALUES ('g1', '{}', 'failed', 'old failure')"
            )

        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM generations WHERE id = 'g1'")
            row = cursor.fetchone()
        self.assertEqual(row["error_message"], "old failure")
        for column in _COLUMNS:
            self.assertIsNone(row[column])

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = self._columns()

        self.migration.up()

        self.assertEqual(before, self._columns())


if __name__ == "__main__":
    unittest.main()

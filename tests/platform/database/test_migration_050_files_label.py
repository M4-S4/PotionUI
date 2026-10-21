import unittest

from tests.platform.database._support import MigrationCase


class TestMigration050FilesLabel(MigrationCase):
    MIGRATION = "050_files_label"

    def _columns(self, table):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

    def test_label_column_lands_on_files(self):
        self.assertNotIn("label", self._columns("files"))

        self.migration.up()

        self.assertIn("label", self._columns("files"))

    def test_existing_rows_have_no_label(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('user1', 'tester', 'tester@example.com', 'hashed')"
            )
            cursor.execute(
                "INSERT INTO files (id, file_path, file_type, user_id) "
                "VALUES ('f1', 'a.png', 'IMAGE', 'user1')"
            )
            cursor.execute("SELECT label FROM files WHERE id = 'f1'")
            self.assertIsNone(cursor.fetchone()["label"])

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = self._columns("files")

        self.migration.up()

        self.assertEqual(before, self._columns("files"))


if __name__ == "__main__":
    unittest.main()

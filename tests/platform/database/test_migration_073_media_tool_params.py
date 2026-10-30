import sqlite3
import unittest

from tests.platform.database._support import MigrationCase


class TestMigration073MediaToolParams(MigrationCase):
    MIGRATION = "073_media_tool_params"

    def _user(self, user_id="u1"):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, 'x')",
                (user_id, user_id, f"{user_id}@example.com"),
            )

    def _insert(self, user_id, tool_id, params="{}"):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO media_tool_params (user_id, tool_id, params_json, updated_at) VALUES (?, ?, ?, 'now')",
                (user_id, tool_id, params),
            )

    def _tables(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            return {row[0] for row in cursor.fetchall()}

    def test_the_table_is_created_with_the_contract_columns(self):
        self.assertNotIn("media_tool_params", self._tables())

        self.migration.up()

        self.assertIn("media_tool_params", self._tables())
        self.assertEqual(self.columns("media_tool_params"), {"user_id", "tool_id", "params_json", "updated_at"})

    def test_the_primary_key_is_user_and_tool(self):
        self.migration.up()
        self._user()
        self._insert("u1", "trim")

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("u1", "trim")
        self._insert("u1", "crop")

    def test_deleting_a_user_removes_their_rows(self):
        self.migration.up()
        self._user("u1")
        self._user("u2")
        self._insert("u1", "trim")
        self._insert("u2", "trim")

        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = 'u1'")
            cursor.execute("SELECT user_id FROM media_tool_params")
            self.assertEqual([row[0] for row in cursor.fetchall()], ["u2"])

    def test_a_row_for_a_missing_user_is_refused(self):
        self.migration.up()

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("ghost", "trim")

    def test_a_second_run_keeps_the_existing_rows(self):
        self.migration.up()
        self._user()
        self._insert("u1", "trim", '{"start": 1}')

        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT params_json FROM media_tool_params")
            self.assertEqual([row[0] for row in cursor.fetchall()], ['{"start": 1}'])


if __name__ == "__main__":
    unittest.main()

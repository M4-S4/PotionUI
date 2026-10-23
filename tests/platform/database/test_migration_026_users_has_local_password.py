import unittest

from tests.platform.database._support import MigrationCase


class TestMigration026UsersHasLocalPassword(MigrationCase):
    MIGRATION = "026_users_has_local_password"

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(users)")
            return {row[1] for row in cursor.fetchall()}

    def test_column_lands_on_users(self):
        self.assertNotIn("has_local_password", self._columns())

        self.migration.up()

        self.assertIn("has_local_password", self._columns())

    def test_existing_rows_default_to_true(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('user1', 'tester', 'tester@example.com', 'hashed')"
            )

        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT has_local_password FROM users WHERE id = 'user1'")
            row = cursor.fetchone()
            self.assertEqual(row["has_local_password"], 1)

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = self._columns()

        self.migration.up()

        self.assertEqual(self._columns(), before)


if __name__ == "__main__":
    unittest.main()

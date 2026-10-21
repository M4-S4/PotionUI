import unittest

from tests.platform.database._support import MigrationCase


class TestMigration028PromptVariables(MigrationCase):
    MIGRATION = "028_prompt_variables"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('u1', 'u1', 'u1@test.com', 'x')"
            )
            cursor.execute(
                "INSERT INTO prompts (id, user_id, flattened_text) VALUES ('p1', 'u1', 'a fox')"
            )

    def _columns(self):
        with self.db.get_connection() as conn:
            return {row[1] for row in conn.execute("PRAGMA table_info(prompts)").fetchall()}

    def test_adds_the_variables_column(self):
        self.assertNotIn("variables", self._columns())

        self.migration.up()

        self.assertIn("variables", self._columns())

    def test_existing_rows_get_a_null_variables_value(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            row = conn.execute("SELECT variables FROM prompts WHERE id = 'p1'").fetchone()
        self.assertIsNone(row["variables"])

    def test_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertIn("variables", self._columns())


if __name__ == '__main__':
    unittest.main()

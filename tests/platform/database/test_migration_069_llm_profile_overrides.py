import unittest

from tests.platform.database._support import MigrationCase

class TestMigration069LlmProfileOverrides(MigrationCase):
    MIGRATION = "069_llm_profile_overrides"

    def _insert_config(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO llm_configurations (id, name, type, base_url, model, system_message) "
                "VALUES (?, 'name', 'ollama', 'http://localhost', 'model', 'system')",
                (config_id,),
            )

    def _overrides(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT profile_overrides FROM llm_configurations WHERE id = ?", (config_id,))
            return cursor.fetchone()["profile_overrides"]

    def test_column_lands_on_llm_configurations(self):
        self.assertNotIn("profile_overrides", self.columns("llm_configurations"))

        self.migration.up()

        self.assertIn("profile_overrides", self.columns("llm_configurations"))

    def test_existing_configs_start_without_overrides(self):
        self._insert_config("existing")

        self.migration.up()

        self.assertIsNone(self._overrides("existing"))

    def test_new_configs_default_to_no_overrides(self):
        self.migration.up()

        self._insert_config("fresh")

        self.assertIsNone(self._overrides("fresh"))

    def test_a_second_run_changes_nothing_and_keeps_stored_overrides(self):
        self.migration.up()
        self._insert_config("kept")
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE llm_configurations SET profile_overrides = '{\"replay\":\"never\"}' WHERE id = 'kept'")
        before = self.columns("llm_configurations")

        self.migration.up()

        self.assertEqual(before, self.columns("llm_configurations"))
        self.assertEqual(self._overrides("kept"), '{"replay":"never"}')

if __name__ == "__main__":
    unittest.main()

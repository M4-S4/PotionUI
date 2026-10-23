import unittest

from tests.platform.database._support import MigrationCase


class TestMigration065LlmShowThinking(MigrationCase):
    MIGRATION = "065_llm_show_thinking"

    def _insert_config(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO llm_configurations (id, name, type, base_url, model, system_message) "
                "VALUES (?, 'name', 'openai', 'http://localhost', 'model', 'system')",
                (config_id,),
            )

    def _show_thinking(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT show_thinking FROM llm_configurations WHERE id = ?", (config_id,))
            return cursor.fetchone()["show_thinking"]

    def test_column_lands_on_llm_configurations(self):
        self.assertNotIn("show_thinking", self.columns("llm_configurations"))

        self.migration.up()

        self.assertIn("show_thinking", self.columns("llm_configurations"))

    def test_existing_configs_stay_hidden(self):
        self._insert_config("existing")

        self.migration.up()

        self.assertEqual(self._show_thinking("existing"), 0)

    def test_new_configs_default_to_hidden(self):
        self.migration.up()

        self._insert_config("fresh")

        self.assertEqual(self._show_thinking("fresh"), 0)

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = self.columns("llm_configurations")

        self.migration.up()

        self.assertEqual(before, self.columns("llm_configurations"))


if __name__ == "__main__":
    unittest.main()

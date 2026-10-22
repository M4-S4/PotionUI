import json
import unittest

from tests.platform.database._support import MigrationCase


class TestMigration066OllamaThinkDisabled(MigrationCase):
    MIGRATION = "066_ollama_think_disabled"

    def _insert(self, config_id, config_type, options):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO llm_configurations (id, name, type, base_url, model, system_message, provider_options) "
                "VALUES (?, 'name', ?, 'http://localhost', 'model', 'system', ?)",
                (config_id, config_type, options),
            )

    def _raw(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT provider_options FROM llm_configurations WHERE id = ?", (config_id,))
            return cursor.fetchone()[0]

    def _options(self, config_id):
        return json.loads(self._raw(config_id))

    def test_a_config_with_no_options_becomes_disabled(self):
        self._insert("no_options", "ollama", None)

        self.migration.up()

        self.assertEqual(self._options("no_options"), {"think": False})

    def test_empty_options_become_disabled(self):
        self._insert("empty", "ollama", "{}")

        self.migration.up()

        self.assertEqual(self._options("empty"), {"think": False})

    def test_blank_and_json_null_options_become_disabled(self):
        self._insert("blank", "ollama", "")
        self._insert("json_null", "ollama", "null")

        self.migration.up()

        self.assertEqual(self._options("blank"), {"think": False})
        self.assertEqual(self._options("json_null"), {"think": False})

    def test_a_null_think_becomes_disabled_and_other_options_survive(self):
        self._insert("automatic", "ollama", '{"think": null, "keep_alive": "5m", "num_ctx": 8192}')

        self.migration.up()

        self.assertEqual(self._options("automatic"), {"think": False, "keep_alive": "5m", "num_ctx": 8192})

    def test_options_without_a_think_key_gain_it_and_keep_the_rest(self):
        self._insert("other_keys", "ollama", '{"keep_alive": "5m"}')

        self.migration.up()

        self.assertEqual(self._options("other_keys"), {"keep_alive": "5m", "think": False})

    def test_explicit_values_are_untouched(self):
        self._insert("on", "ollama", '{"think": true}')
        self._insert("off", "ollama", '{"think": false}')
        self._insert("level", "ollama", '{"think": "medium"}')

        self.migration.up()

        self.assertEqual(self._raw("on"), '{"think": true}')
        self.assertEqual(self._raw("off"), '{"think": false}')
        self.assertEqual(self._raw("level"), '{"think": "medium"}')

    def test_other_provider_types_are_untouched(self):
        self._insert("openai", "openai", '{"reasoning_effort": null}')
        self._insert("native", "native", None)
        self._insert("openai_empty", "openai", "{}")

        self.migration.up()

        self.assertEqual(self._raw("openai"), '{"reasoning_effort": null}')
        self.assertIsNone(self._raw("native"))
        self.assertEqual(self._raw("openai_empty"), "{}")

    def test_unreadable_options_are_left_alone(self):
        self._insert("broken", "ollama", "{not json")
        self._insert("listed", "ollama", "[1, 2]")

        self.migration.up()

        self.assertEqual(self._raw("broken"), "{not json")
        self.assertEqual(self._raw("listed"), "[1, 2]")

    def test_a_second_run_changes_nothing(self):
        self._insert("automatic", "ollama", '{"think": null}')
        self._insert("empty", "ollama", None)
        self.migration.up()
        before = (self._raw("automatic"), self._raw("empty"))

        self.migration.up()

        self.assertEqual((self._raw("automatic"), self._raw("empty")), before)


if __name__ == "__main__":
    unittest.main()

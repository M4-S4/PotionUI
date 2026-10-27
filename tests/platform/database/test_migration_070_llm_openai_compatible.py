import unittest

from tests.platform.database._support import MigrationCase


class TestMigration070LlmOpenaiCompatible(MigrationCase):
    MIGRATION = "070_llm_openai_compatible"

    def _insert(self, config_id, config_type, base_url, **extra):
        columns = {
            "id": config_id,
            "name": f"name {config_id}",
            "type": config_type,
            "base_url": base_url,
            "model": "model",
            "system_message": "system",
            **extra,
        }
        with self.db.get_cursor() as cursor:
            cursor.execute(
                f"INSERT INTO llm_configurations ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
                tuple(columns.values()),
            )

    def _type(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT type FROM llm_configurations WHERE id = ?", (config_id,))
            return cursor.fetchone()[0]

    def _snapshot(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM llm_configurations ORDER BY id")
            return [tuple(row) for row in cursor.fetchall()]

    def test_configs_on_the_hosted_api_stay_openai(self):
        for number, url in enumerate([
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/",
            "HTTPS://API.OPENAI.COM/v1",
            "https://api.openai.com:443/v1",
            "https://api.openai.com./v1",
            "http://api.openai.com",
            "api.openai.com/v1",
            "  https://api.openai.com/v1  ",
        ]):
            self._insert(f"hosted{number}", "openai", url)

        self.migration.up()

        for number in range(8):
            self.assertEqual(self._type(f"hosted{number}"), "openai")

    def test_configs_on_any_other_server_become_openai_compatible(self):
        urls = {
            "local": "http://localhost:1234/v1",
            "lan": "http://192.168.1.20:8000/v1",
            "router": "https://openrouter.ai/api/v1",
            "azure": "https://resource.openai.azure.com/openai/v1",
            "lookalike": "https://api.openai.com.example.net/v1",
            "suffix": "https://notapi.openai.com/v1",
            "userinfo": "https://api.openai.com@evil.example/v1",
            "schemeless": "localhost:1234/v1",
            "schemeless_lan": "192.168.1.20:8000/v1",
        }
        for config_id, url in urls.items():
            self._insert(config_id, "openai", url)

        self.migration.up()

        for config_id in urls:
            self.assertEqual(self._type(config_id), "openai-compatible", config_id)

    def test_configs_without_a_usable_host_are_left_alone(self):
        self._insert("blank", "openai", "")
        self._insert("spaces", "openai", "   ")
        self._insert("no_host", "openai", "https://")
        self._insert("broken", "openai", "http://[::1")

        self.migration.up()

        for config_id in ("blank", "spaces", "no_host", "broken"):
            self.assertEqual(self._type(config_id), "openai", config_id)

    def test_other_provider_types_are_untouched(self):
        self._insert("ollama", "ollama", "http://localhost:11434")
        self._insert("native", "native", "")
        self._insert("deepseek", "deepseek", "https://api.deepseek.com")
        self._insert("compatible", "openai-compatible", "http://localhost:1234/v1")

        self.migration.up()

        self.assertEqual(
            [self._type(config_id) for config_id in ("ollama", "native", "deepseek", "compatible")],
            ["ollama", "native", "deepseek", "openai-compatible"],
        )

    def test_only_the_type_changes(self):
        self._insert(
            "keeps", "openai", "http://localhost:1234/v1",
            api_key="secret", provider_options='{"top_p": 0.9}', temperature=0.3, max_tokens=500, enabled=0,
        )
        before = self._snapshot()

        self.migration.up()

        after = self._snapshot()
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(llm_configurations)")
            names = [row[1] for row in cursor.fetchall()]
        changed = [name for name, old, new in zip(names, before[0], after[0]) if old != new]
        self.assertEqual(changed, ["type"])

    def test_a_second_run_changes_nothing(self):
        self._insert("hosted", "openai", "https://api.openai.com/v1")
        self._insert("local", "openai", "http://localhost:1234/v1")
        self.migration.up()
        once = self._snapshot()

        self.migration.up()

        self.assertEqual(self._snapshot(), once)
        self.assertEqual((self._type("hosted"), self._type("local")), ("openai", "openai-compatible"))

    def test_an_empty_table_is_fine(self):
        self.migration.up()

        self.assertEqual(self._snapshot(), [])


if __name__ == "__main__":
    unittest.main()

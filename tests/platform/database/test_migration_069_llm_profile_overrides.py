import json
import sqlite3
import unittest

from tests.platform.database._support import MigrationCase


class TestMigration069LlmProfileOverrides(MigrationCase):
    MIGRATION = "069_llm_profile_overrides"
    PREREQUISITES = ("001_baseline", "065_llm_show_thinking")

    def _insert(self, config_id, *, type="ollama", temperature=0.7, max_tokens=1000, options=None, overrides=None, **extra):
        values = dict(
            id=config_id, name=f"name-{config_id}", type=type, base_url="http://localhost", api_key="enc:secret",
            model="model", system_message="system", temperature=temperature, max_tokens=max_tokens, timeout=45,
            supports_vision=1, provider_options=options if options is None or isinstance(options, str) else json.dumps(options),
            disable_system_prompt=1, memory_reflection=0, show_thinking=1,
        )
        values.update(extra)
        if overrides is not None:
            values["profile_overrides"] = json.dumps(overrides)
        names = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self.db.get_cursor() as cursor:
            cursor.execute(f"INSERT INTO llm_configurations ({names}) VALUES ({marks})", tuple(values.values()))

    def _row(self, config_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM llm_configurations WHERE id = ?", (config_id,))
            return dict(cursor.fetchone())

    def _options(self, config_id):
        raw = self._row(config_id)["provider_options"]
        return json.loads(raw) if raw else None

    def _overrides(self, config_id):
        raw = self._row(config_id)["profile_overrides"]
        return json.loads(raw) if raw else None

    def _notnull(self, column):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(llm_configurations)")
            return {row["name"]: bool(row["notnull"]) for row in cursor.fetchall()}[column]

    def _names(self, kind):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = ? AND tbl_name = 'llm_configurations'", (kind,))
            return {row["name"] for row in cursor.fetchall()}

    def test_the_overrides_column_lands_and_the_sampling_columns_allow_null(self):
        self.assertTrue(self._notnull("temperature") and self._notnull("max_tokens"))
        self.assertNotIn("profile_overrides", self.columns("llm_configurations"))

        self.migration.up()

        self.assertIn("profile_overrides", self.columns("llm_configurations"))
        self.assertFalse(self._notnull("temperature"))
        self.assertFalse(self._notnull("max_tokens"))
        self.assertTrue(self._notnull("timeout"))

    def test_rows_still_at_the_old_default_temperature_become_null_and_others_are_kept(self):
        self._insert("default", temperature=0.7, max_tokens=1000)
        self._insert("custom", temperature=0.3, max_tokens=2500)
        self._insert("zero", temperature=0.0, max_tokens=1)

        self.migration.up()

        self.assertIsNone(self._row("default")["temperature"])
        self.assertEqual(self._row("default")["max_tokens"], 1000)
        self.assertEqual(self._row("custom")["temperature"], 0.3)
        self.assertEqual(self._row("custom")["max_tokens"], 2500)
        self.assertEqual(self._row("zero")["temperature"], 0.0)

    def test_every_other_column_survives_the_rebuild(self):
        self._insert("keep", type="openai", options={"seed": 7, "top_p": 0.9})
        before = self._row("keep")

        self.migration.up()
        after = self._row("keep")

        for column in before:
            if column in ("temperature", "updated_at"):
                continue
            self.assertEqual(after[column], before[column], column)
        self.assertIsNone(after["profile_overrides"])

    def test_assignments_to_users_and_groups_survive_the_rebuild(self):
        self._insert("shared")
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES ('u1', 'user', 'user@example.com', 'x')"
            )
            cursor.execute("INSERT INTO user_groups (id, name) VALUES ('g1', 'group')")
            cursor.execute("INSERT INTO user_llms (id, user_id, llm_config_id) VALUES ('a1', 'u1', 'shared')")
            cursor.execute("INSERT INTO user_group_llms (id, group_id, llm_config_id) VALUES ('a2', 'g1', 'shared')")

        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT id, user_id, llm_config_id FROM user_llms")
            self.assertEqual([tuple(row) for row in cursor.fetchall()], [("a1", "u1", "shared")])
            cursor.execute("SELECT id, group_id, llm_config_id FROM user_group_llms")
            self.assertEqual([tuple(row) for row in cursor.fetchall()], [("a2", "g1", "shared")])
            cursor.execute("DELETE FROM llm_configurations WHERE id = 'shared'")
            cursor.execute("SELECT COUNT(*) FROM user_llms")
            self.assertEqual(cursor.fetchone()[0], 0)

    def test_the_scratch_tables_do_not_survive(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%069'")
            self.assertEqual(cursor.fetchall(), [])

    def test_indexes_and_the_updated_at_trigger_are_recreated(self):
        self.migration.up()

        self.assertEqual(
            self._names("index") & {"idx_llm_configurations_enabled", "idx_llm_configurations_vision"},
            {"idx_llm_configurations_enabled", "idx_llm_configurations_vision"},
        )
        self.assertEqual(self._names("trigger"), {"update_llm_configurations_updated_at"})

    def test_new_rows_can_leave_the_sampling_columns_empty(self):
        self.migration.up()

        self._insert("fresh", temperature=None, max_tokens=None)

        self.assertIsNone(self._row("fresh")["temperature"])
        self.assertIsNone(self._row("fresh")["max_tokens"])
        self.assertIsNone(self._row("fresh")["profile_overrides"])

    def test_a_second_run_changes_nothing_and_keeps_an_explicit_default_temperature(self):
        self._insert("moved", options={"context_window": 8192})
        self.migration.up()
        self._insert("explicit", temperature=0.7)
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE llm_configurations SET profile_overrides = '{\"replay\":\"never\"}' WHERE id = 'explicit'")
        before = (self._row("explicit"), self._row("moved"))

        self.migration.up()

        self.assertEqual((self._row("explicit"), self._row("moved")), before)
        self.assertEqual(self._row("explicit")["temperature"], 0.7)

    def test_it_also_runs_without_the_show_thinking_column(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("ALTER TABLE llm_configurations DROP COLUMN show_thinking")
            cursor.execute(
                "INSERT INTO llm_configurations (id, name, type, base_url, model, system_message) "
                "VALUES ('old', 'n', 'ollama', 'http://x', 'm', 's')"
            )

        self.migration.up()

        self.assertEqual(self._row("old")["show_thinking"], 0)

    def test_a_failure_part_way_leaves_the_original_table_untouched(self):
        self._insert("safe", temperature=0.7)
        with self.db.get_cursor() as cursor:
            cursor.execute("DROP TABLE user_group_llms")

        with self.assertRaises(sqlite3.OperationalError):
            self.migration.up()

        self.assertEqual(self._row("safe")["temperature"], 0.7)
        self.assertTrue(self._notnull("temperature"))
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%069'")
            self.assertEqual(cursor.fetchall(), [])

    def test_ollama_context_options_become_a_requested_size_and_a_budget(self):
        self._insert("window", type="ollama", options={"context_window": 16384, "think": False})
        self._insert("numctx", type="ollama", options={"num_ctx": 32768})
        self._insert("both", type="ollama", options={"context_window": 8192, "num_ctx": 4096})

        self.migration.up()

        self.assertEqual(self._overrides("window"), {"context_request": 16384, "context_window": 16384})
        self.assertEqual(self._options("window"), {"think": False})
        self.assertEqual(self._overrides("numctx"), {"context_request": 32768, "context_window": 32768})
        self.assertIsNone(self._options("numctx"))
        self.assertEqual(self._overrides("both"), {"context_request": 8192, "context_window": 8192})
        self.assertIsNone(self._options("both"))

    def test_other_types_only_get_a_budget_and_keep_a_num_ctx_that_means_nothing_to_them(self):
        self._insert("hosted", type="openai", options={"context_window": 65536, "num_ctx": 123, "seed": 1})

        self.migration.up()

        self.assertEqual(self._overrides("hosted"), {"context_window": 65536})
        self.assertEqual(self._options("hosted"), {"num_ctx": 123, "seed": 1})

    def test_forcing_prompt_tools_becomes_a_profile_override_and_a_false_flag_is_just_dropped(self):
        self._insert("forced", type="openai", options={"force_prompt_tools": True})
        self._insert("unforced", type="openai", options={"force_prompt_tools": False, "seed": 2})
        self._insert("strange", type="openai", options={"force_prompt_tools": "yes"})

        self.migration.up()

        self.assertEqual(self._overrides("forced"), {"prompt_tools": True})
        self.assertIsNone(self._options("forced"))
        self.assertIsNone(self._overrides("unforced"))
        self.assertEqual(self._options("unforced"), {"seed": 2})
        self.assertIsNone(self._overrides("strange"))
        self.assertIsNone(self._options("strange"))

    def test_existing_overrides_win_over_moved_values(self):
        self.migration.up()
        self._insert("mixed", type="ollama", options={"context_window": 8192}, overrides={"context_window": 4096, "replay": "never"})

        self.migration.up()

        self.assertEqual(self._overrides("mixed"), {"context_request": 8192, "context_window": 4096, "replay": "never"})

    def test_unusable_sizes_are_dropped_not_moved(self):
        self._insert("tiny", type="ollama", options={"context_window": 512})
        self._insert("flag", type="ollama", options={"context_window": True, "num_ctx": "8192"})

        self.migration.up()

        for config_id in ("tiny", "flag"):
            self.assertIsNone(self._overrides(config_id))
            self.assertIsNone(self._options(config_id))

    def test_provider_options_that_are_not_an_object_are_left_alone(self):
        self._insert("broken", options="{not json")
        self._insert("list", options="[1, 2]")

        self.migration.up()

        self.assertEqual(self._row("broken")["provider_options"], "{not json")
        self.assertEqual(self._row("list")["provider_options"], "[1, 2]")


if __name__ == "__main__":
    unittest.main()

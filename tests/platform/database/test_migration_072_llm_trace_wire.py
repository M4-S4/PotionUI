import unittest

from tests.platform.database._support import MigrationCase

NEW_COLUMNS = ("adapter", "request_body", "response_raw", "usage", "cost", "error")


class TestMigration072LlmTraceWire(MigrationCase):
    MIGRATION = "072_llm_trace_wire"

    def _insert_session(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES ('u1', 'u1', 'u1@example.com', 'x')"
            )
            cursor.execute(
                "INSERT INTO chat_sessions (id, user_id, mode, name, status) "
                "VALUES ('s1', 'u1', 'generation', 'n', 'active')"
            )

    def _insert_trace(self, trace_id):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO chat_llm_call_traces (id, session_id, provider, model, response_text, prompt_tokens) "
                "VALUES (?, 's1', 'openai', 'gpt', 'hello', 12)",
                (trace_id,),
            )

    def _trace(self, trace_id):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT * FROM chat_llm_call_traces WHERE id = ?", (trace_id,))
            return cursor.fetchone()

    def test_the_wire_columns_land_on_the_traces_table(self):
        for name in NEW_COLUMNS:
            self.assertNotIn(name, self.columns("chat_llm_call_traces"))

        self.migration.up()

        for name in NEW_COLUMNS:
            self.assertIn(name, self.columns("chat_llm_call_traces"))

    def test_existing_traces_keep_their_data_and_get_empty_wire_fields(self):
        self._insert_session()
        self._insert_trace("old")

        self.migration.up()

        row = self._trace("old")
        self.assertEqual((row["response_text"], row["prompt_tokens"]), ("hello", 12))
        for name in NEW_COLUMNS:
            self.assertIsNone(row[name])

    def test_new_rows_store_each_wire_field(self):
        self._insert_session()
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO chat_llm_call_traces "
                "(id, session_id, adapter, request_body, response_raw, usage, cost, error) "
                "VALUES ('new', 's1', 'ollama-chat', '{\"a\":1}', 'raw', '{\"input_tokens\":3}', 0.25, '{\"kind\":\"x\"}')"
            )

        row = self._trace("new")
        self.assertEqual(
            (row["adapter"], row["request_body"], row["response_raw"], row["usage"], row["cost"], row["error"]),
            ("ollama-chat", '{"a":1}', "raw", '{"input_tokens":3}', 0.25, '{"kind":"x"}'),
        )

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = self.columns("chat_llm_call_traces")

        self.migration.up()

        self.assertEqual(before, self.columns("chat_llm_call_traces"))


if __name__ == "__main__":
    unittest.main()

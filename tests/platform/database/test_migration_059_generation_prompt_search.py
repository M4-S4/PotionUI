import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from src.platform.database.database import Database

_MIGRATIONS = (
    Path(__file__).resolve().parents[3]
    / "src" / "platform" / "database" / "migrations"
)

_TRIGGERS = {
    "generation_prompt_text_insert",
    "generation_prompt_text_update",
    "generation_prompt_text_delete",
}

def _load_migration(stem, database):
    spec = importlib.util.spec_from_file_location(stem, _MIGRATIONS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    module.db = database
    return module

class TestMigration059GenerationPromptSearch(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.addCleanup(Database.close_pool)
        Database._instance = None
        self.db = Database()
        self.db.db_path = Path(temp.name) / "test.sqlite"
        self.db._initialized = True
        _load_migration("001_baseline", self.db).up()
        self.migration = _load_migration("059_generation_prompt_search", self.db)

    def tearDown(self):
        Database._instance = None

    def _insert(self, generation_id, form_data, prompt_state=None):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, form_data, prompt_state) VALUES (?, ?, ?)",
                (
                    generation_id,
                    form_data if isinstance(form_data, str) else json.dumps(form_data),
                    None if prompt_state is None else json.dumps(prompt_state),
                ),
            )

    def _execute(self, sql, params=()):
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, params)

    def _texts(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT generation_id, text FROM generation_prompt_text")
            return {row["generation_id"]: row["text"] for row in cursor.fetchall()}

    def _schema(self, kind):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = ? AND name LIKE 'generation_prompt%'", (kind,))
            return {row["name"] for row in cursor.fetchall()}

    def test_backfill_collects_prompt_strings_from_both_columns(self):
        self._insert("plain", {"prompt": "A Lighthouse", "negative_prompt": "blurry"})
        self._insert("both", {"nested": {"positive_prompt": "tall tower"}}, {"prompt": "at dusk", "negativePrompt": "fog"})
        self._insert("none", {"steps": 20})
        self._insert("bad", "not json")

        self.migration.up()

        texts = self._texts()
        self.assertEqual(texts["plain"], "A Lighthouse")
        self.assertEqual(sorted(texts["both"].split("\x1f")), ["at dusk", "tall tower"])
        self.assertIsNone(texts["none"])
        self.assertIsNone(texts["bad"])

    def test_insert_trigger_fills_the_side_table(self):
        self.migration.up()

        self._insert("new", {"prompt": "a red fox"}, {"promptSegments": [{"prompt": "snowy field"}]})

        self.assertEqual(sorted(self._texts()["new"].split("\x1f")), ["a red fox", "snowy field"])

    def test_update_trigger_follows_form_data_and_prompt_state(self):
        self.migration.up()
        self._insert("g1", {"prompt": "a red fox"})

        self._execute("UPDATE generations SET form_data = ? WHERE id = ?", (json.dumps({"prompt": "a blue whale"}), "g1"))
        self.assertEqual(self._texts()["g1"], "a blue whale")

        self._execute("UPDATE generations SET prompt_state = ? WHERE id = ?", (json.dumps({"prompt": "deep sea"}), "g1"))
        self.assertEqual(sorted(self._texts()["g1"].split("\x1f")), ["a blue whale", "deep sea"])

    def test_update_of_other_columns_keeps_the_side_row(self):
        self.migration.up()
        self._insert("g1", {"prompt": "a red fox"})
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT id FROM generation_prompt_text WHERE generation_id = 'g1'")
            before = cursor.fetchone()[0]

        self._execute("UPDATE generations SET status = 'completed' WHERE id = 'g1'")

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT id, text FROM generation_prompt_text WHERE generation_id = 'g1'")
            row = cursor.fetchone()
        self.assertEqual((row[0], row[1]), (before, "a red fox"))

    def test_cascade_delete_removes_the_side_row(self):
        self.migration.up()
        self._insert("gone", {"prompt": "a red fox"})
        self._insert("kept", {"prompt": "a red panda"})

        self._execute("DELETE FROM generations WHERE id = 'gone'")

        self.assertEqual(self._texts(), {"kept": "a red panda"})

    def test_delete_without_foreign_keys_still_removes_the_side_row(self):
        self.migration.up()
        self._insert("gone", {"prompt": "a red fox"})

        conn = sqlite3.connect(self.db.db_path)
        try:
            conn.execute("DELETE FROM generations WHERE id = 'gone'")
            conn.commit()
        finally:
            conn.close()

        self.assertEqual(self._texts(), {})

    def test_running_twice_is_a_no_op(self):
        self._insert("g1", {"prompt": "a red fox"})
        self.migration.up()

        self.migration.up()

        self.assertEqual(self._texts(), {"g1": "a red fox"})

    def test_up_creates_only_the_side_table_and_its_triggers(self):
        self.migration.up()

        self.assertEqual(self._schema("table"), {"generation_prompt_text"})
        self.assertEqual(self._schema("trigger"), _TRIGGERS)

    def test_down_drops_the_table_and_triggers(self):
        self.migration.up()

        self.migration.down()

        self.assertEqual(self._schema("trigger"), set())
        self.assertEqual(self._schema("table"), set())
        self._insert("after", {"prompt": "a red fox"})


if __name__ == "__main__":
    unittest.main()

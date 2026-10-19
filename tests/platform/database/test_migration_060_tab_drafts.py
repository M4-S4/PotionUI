import importlib.util
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


def _load_migration(stem, database):
    spec = importlib.util.spec_from_file_location(stem, _MIGRATIONS / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    module.db = database
    return module


class TestMigration060TabDrafts(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        Database._instance = None
        self.db = Database()
        self.db.db_path = Path(self.temp_dir) / "test.sqlite"
        self.db._initialized = True
        with self.db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE applied_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration_name TEXT UNIQUE NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        _load_migration("001_baseline", self.db).up()
        self.migration = _load_migration("060_tab_drafts", self.db)

    def tearDown(self):
        Database._instance = None

    def _tables(self):
        with self.db.get_connection() as conn:
            return {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }

    def _insert_user(self, user_id):
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
                (user_id, user_id, f"{user_id}@example.com", "h"),
            )
            conn.commit()

    def _insert_draft(self, user_id, tab_id):
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO tab_drafts (user_id, tab_id, doc, rev, created_at, updated_at) "
                "VALUES (?, ?, '{}', 1, 'a', 'a')",
                (user_id, tab_id),
            )
            conn.commit()

    def test_baseline_alone_has_no_table(self):
        self.assertNotIn("tab_drafts", self._tables())

    def test_creates_the_table_with_the_documented_columns(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            columns = {row[1]: row for row in conn.execute("PRAGMA table_info(tab_drafts)")}
        self.assertEqual(
            set(columns),
            {"user_id", "tab_id", "doc", "rev", "variable_count", "created_at", "updated_at"},
        )
        for name in columns:
            self.assertEqual(columns[name][3], 1, name)

    def test_primary_key_is_user_then_tab(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            keyed = sorted(
                (row[5], row[1]) for row in conn.execute("PRAGMA table_info(tab_drafts)") if row[5]
            )
        self.assertEqual(keyed, [(1, "user_id"), (2, "tab_id")])

    def test_updated_at_is_indexed_for_the_sweep(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            columns = [row[2] for row in conn.execute("PRAGMA index_info(idx_tab_drafts_updated_at)")]
        self.assertEqual(columns, ["updated_at"])

    def test_same_tab_id_may_belong_to_two_users_but_not_twice_to_one(self):
        self.migration.up()
        self._insert_user("u1")
        self._insert_user("u2")

        self._insert_draft("u1", "tab-1")
        self._insert_draft("u2", "tab-1")
        with self.assertRaises(sqlite3.IntegrityError):
            self._insert_draft("u1", "tab-1")

    def test_a_draft_needs_an_existing_user_and_goes_with_it(self):
        self.migration.up()
        self._insert_user("u1")
        self._insert_draft("u1", "tab-1")

        with self.db.get_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with self.assertRaises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO tab_drafts (user_id, tab_id, doc, rev, created_at, updated_at) "
                    "VALUES ('ghost', 't', '{}', 1, 'a', 'a')"
                )
            conn.execute("DELETE FROM users WHERE id = 'u1'")
            conn.commit()
            remaining = conn.execute("SELECT COUNT(*) FROM tab_drafts").fetchone()[0]
        self.assertEqual(remaining, 0)

    def test_a_table_without_the_count_column_gets_it_backfilled(self):
        self._insert_user("u1")
        with self.db.get_connection() as conn:
            conn.execute(
                "CREATE TABLE tab_drafts (user_id TEXT NOT NULL REFERENCES users (id) ON DELETE CASCADE, "
                "tab_id TEXT NOT NULL, doc TEXT NOT NULL, rev INTEGER NOT NULL DEFAULT 1, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (user_id, tab_id))"
            )
            conn.execute(
                "INSERT INTO tab_drafts VALUES ('u1', 'a', '{\"variables\":{\"x\":{},\"y\":{}}}', 1, 'a', 'a')"
            )
            conn.execute("INSERT INTO tab_drafts VALUES ('u1', 'b', '{}', 1, 'a', 'a')")
            conn.commit()

        self.migration.up()

        with self.db.get_connection() as conn:
            counts = dict(conn.execute("SELECT tab_id, variable_count FROM tab_drafts").fetchall())
        self.assertEqual(counts, {"a": 2, "b": 0})

    def test_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertIn("tab_drafts", self._tables())

    def test_down_removes_it(self):
        self.migration.up()
        self.migration.down()

        self.assertNotIn("tab_drafts", self._tables())

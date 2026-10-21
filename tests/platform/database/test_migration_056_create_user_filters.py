import sqlite3

from tests.platform.database._support import MigrationCase


class TestMigration056CreateUserFilters(MigrationCase):
    MIGRATION = "056_create_user_filters"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON")
            for user_id in ("u1", "u2"):
                cursor.execute(
                    "INSERT INTO users (id, username, email, password_hash, account_type) VALUES (?, ?, ?, 'x', 'USER')",
                    (user_id, user_id, f"{user_id}@example.test"),
                )

    def _insert(self, filter_id, owner="u1", name="Ember"):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO user_filters (id, owner_id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')",
                (filter_id, owner, name),
            )

    def _names(self, kind):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT name FROM sqlite_master WHERE type = ?", (kind,))
            return {row["name"] for row in cursor.fetchall()}

    def test_creates_the_table_with_the_expected_columns(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(user_filters)")
            columns = {row["name"]: row for row in cursor.fetchall()}
        assert set(columns) == {
            "id", "owner_id", "name", "description", "group_name", "intensity", "steps_json",
            "schema_version", "created_at", "updated_at",
        }
        assert columns["group_name"]["dflt_value"] == "'Mine'"
        assert columns["intensity"]["dflt_value"] == "100"
        assert columns["steps_json"]["dflt_value"] == "'[]'"

    def test_creates_both_indexes(self):
        self.migration.up()

        assert {"idx_user_filters_owner_name", "idx_user_filters_owner"} <= self._names("index")

    def test_the_same_name_for_one_owner_is_rejected_ignoring_case(self):
        self.migration.up()
        self._insert("f1", name="Ember")

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("f2", name="EMBER")

    def test_the_name_may_repeat_across_owners(self):
        self.migration.up()
        self._insert("f1")

        self._insert("f2", owner="u2")

    def test_defaults_fill_in_a_minimal_row(self):
        self.migration.up()
        self._insert("f1")

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT group_name, intensity, steps_json, schema_version FROM user_filters")
            row = cursor.fetchone()
        assert (row["group_name"], row["intensity"], row["steps_json"], row["schema_version"]) == ("Mine", 100, "[]", 1)

    def test_deleting_a_user_removes_their_filters(self):
        self.migration.up()
        self._insert("f1")
        self._insert("f2", owner="u2")

        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = 'u1'")

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT id FROM user_filters")
            assert [row["id"] for row in cursor.fetchall()] == ["f2"]

    def test_a_row_for_an_unknown_owner_is_rejected(self):
        self.migration.up()

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("f1", owner="ghost")

    def test_up_twice_is_harmless(self):
        self.migration.up()
        self._insert("f1")

        self.migration.up()

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("f2")

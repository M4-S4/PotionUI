import sqlite3

from tests.platform.database._support import MigrationCase


class TestMigration048GenerationIdempotencyKey(MigrationCase):
    MIGRATION = "048_generation_idempotency_key"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            for user_id in ("u1", "u2"):
                cursor.execute(
                    "INSERT INTO users (id, username, email, password_hash, account_type) VALUES (?, ?, ?, 'x', 'USER')",
                    (user_id, user_id, f"{user_id}@example.test"),
                )

    def _insert(self, generation_id, user_id="u1", key=None):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, preset_id, form_data, user_id, status, progress, mode, idempotency_key) "
                "VALUES (?, 'p', '{}', ?, 'pending', 0, 'txt2img', ?)",
                (generation_id, user_id, key),
            )

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(generations)")
            return {row["name"] for row in cursor.fetchall()}

    def test_adds_the_column_and_keeps_existing_rows_keyless(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generations (id, preset_id, form_data, user_id, status, progress, mode) "
                "VALUES ('old', 'p', '{}', 'u1', 'completed', 0, 'txt2img')"
            )
        assert "idempotency_key" not in self._columns()

        self.migration.up()

        assert {"idempotency_key", "idempotency_fingerprint"} <= self._columns()
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT idempotency_key FROM generations WHERE id = 'old'")
            assert cursor.fetchone()["idempotency_key"] is None

    def test_the_same_key_twice_for_one_user_is_rejected(self):
        self.migration.up()
        self._insert("g1", key="k")

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("g2", key="k")

    def test_two_users_may_share_a_key(self):
        self.migration.up()

        self._insert("g1", user_id="u1", key="k")
        self._insert("g2", user_id="u2", key="k")

    def test_any_number_of_keyless_rows_is_allowed(self):
        self.migration.up()

        self._insert("g1")
        self._insert("g2")

    def test_up_twice_is_harmless(self):
        self.migration.up()
        self._insert("g1", key="k")

        self.migration.up()

        with self.assertRaises(sqlite3.IntegrityError):
            self._insert("g2", key="k")

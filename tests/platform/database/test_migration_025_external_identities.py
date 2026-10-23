import unittest

from tests.platform.database._support import MigrationCase

_KEYS = (
    "external_login_auto_create",
    "external_login_default_group",
    "external_login_link_by_email",
)


class TestMigration025ExternalIdentities(MigrationCase):
    MIGRATION = "025_external_identities"

    def _settings(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT key, value, value_type, type FROM settings WHERE key IN (?, ?, ?)",
                _KEYS,
            )
            return {row["key"]: dict(row) for row in cursor.fetchall()}

    def _table_exists(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='external_identities'"
            )
            return cursor.fetchone() is not None

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(external_identities)")
            return {row["name"] for row in cursor.fetchall()}

    def test_the_table_is_created_with_its_columns(self):
        self.assertFalse(self._table_exists())

        self.migration.up()

        self.assertTrue(self._table_exists())
        self.assertEqual(
            self._columns(),
            {"id", "issuer", "subject", "user_id", "created_at", "last_login_at"},
        )

    def test_issuer_and_subject_are_unique_together(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash, account_type) "
                "VALUES ('u1', 'one', 'one@example.com', 'h', 'USER')"
            )
            cursor.execute(
                "INSERT INTO external_identities (id, issuer, subject, user_id, created_at) "
                "VALUES ('e1', 'iss', 'sub', 'u1', '2026-01-01T00:00:00+00:00')"
            )

        with self.assertRaises(Exception):
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO external_identities (id, issuer, subject, user_id, created_at) "
                    "VALUES ('e2', 'iss', 'sub', 'u1', '2026-01-01T00:00:00+00:00')"
                )

    def test_the_three_settings_are_seeded_as_system_settings(self):
        self.assertEqual(self._settings(), {})

        self.migration.up()

        seeded = self._settings()
        self.assertEqual(set(seeded), set(_KEYS))
        self.assertEqual(seeded["external_login_auto_create"]["value"], "false")
        self.assertEqual(seeded["external_login_link_by_email"]["value"], "false")
        self.assertEqual(seeded["external_login_default_group"]["value"], "")
        self.assertEqual({row["type"] for row in seeded.values()}, {"SYSTEM"})

    def test_a_second_run_is_harmless_and_leaves_edits_alone(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE settings SET value = 'true' WHERE key = 'external_login_auto_create'"
            )

        self.migration.up()

        seeded = self._settings()
        self.assertEqual(seeded["external_login_auto_create"]["value"], "true")
        self.assertEqual(len(seeded), 3)
        self.assertTrue(self._table_exists())


if __name__ == "__main__":
    unittest.main()

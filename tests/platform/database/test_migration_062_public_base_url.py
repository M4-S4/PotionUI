from tests.platform.database._support import MigrationCase


class TestMigration062PublicBaseUrl(MigrationCase):
    MIGRATION = "062_public_base_url"

    def _row(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT value, value_type, type FROM settings WHERE key = 'public_base_url'")
            row = cursor.fetchone()
            return tuple(row) if row else None

    def test_up_seeds_an_empty_system_setting(self):
        self.migration.up()

        self.assertEqual(self._row(), ("", "string", "SYSTEM"))

    def test_up_keeps_an_existing_value(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = 'https://a.test' WHERE key = 'public_base_url'")
        self.migration.up()

        self.assertEqual(self._row()[0], "https://a.test")

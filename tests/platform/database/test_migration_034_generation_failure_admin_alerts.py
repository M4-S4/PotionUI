from tests.platform.database._support import MigrationCase

_KEYS = ("notify_admins_on_generation_failure", "notify_admins_on_generation_failure_categories")


class TestMigration034GenerationFailureAdminAlerts(MigrationCase):
    MIGRATION = "034_generation_failure_admin_alerts"

    def _rows(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT key, value, value_type, type FROM settings WHERE key IN (?, ?) ORDER BY key", _KEYS
            )
            return [tuple(row) for row in cursor.fetchall()]

    def test_up_seeds_disabled_system_settings(self):
        self.migration.up()

        self.assertEqual(self._rows(), [
            ("notify_admins_on_generation_failure", "false", "boolean", "SYSTEM"),
            ("notify_admins_on_generation_failure_categories", "[]", "json", "SYSTEM"),
        ])

    def test_up_is_idempotent_and_keeps_existing_values(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = 'true' WHERE key = ?", (_KEYS[0],))
        self.migration.up()

        rows = self._rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][1], "true")

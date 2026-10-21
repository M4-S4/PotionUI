"""023 seeds the three backup settings - idempotently, since the migration
runner replays nothing but a second `up()` must still be harmless.
"""

import unittest

from tests.platform.database._support import MigrationCase

_KEYS = ("backup_destination", "backup_retention", "backup_default_tier")


class TestMigration023BackupSettings(MigrationCase):
    MIGRATION = "023_backup_settings"

    def _settings(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT key, value, value_type, type FROM settings WHERE key IN (?, ?, ?)", _KEYS
            )
            return {row["key"]: dict(row) for row in cursor.fetchall()}

    def test_the_three_settings_are_seeded_as_system_settings(self):
        self.assertEqual(self._settings(), {})

        self.migration.up()

        seeded = self._settings()
        self.assertEqual(set(seeded), set(_KEYS))
        self.assertEqual(seeded["backup_destination"]["value"], "backups")
        self.assertEqual(seeded["backup_retention"]["value"], "7")
        self.assertEqual(seeded["backup_default_tier"]["value"], "config")
        self.assertEqual({row["type"] for row in seeded.values()}, {"SYSTEM"})

    def test_a_second_run_leaves_an_edited_value_alone(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "UPDATE settings SET value = '/srv/backups' WHERE key = 'backup_destination'"
            )

        self.migration.up()

        seeded = self._settings()
        self.assertEqual(seeded["backup_destination"]["value"], "/srv/backups")
        self.assertEqual(len(seeded), 3)


if __name__ == "__main__":
    unittest.main()

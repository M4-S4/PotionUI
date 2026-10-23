"""021 seeds the three retention settings - idempotently, since the migration
runner replays nothing but a second `up()` must still be harmless."""

import unittest

from tests.platform.database._support import MigrationCase

_SETTING_KEYS = (
    "tmp_retention_days",
    "run_report_retention_days",
    "llm_trace_retention_days",
)


class TestMigration021Housekeeping(MigrationCase):
    MIGRATION = "021_housekeeping"

    def _settings(self):
        placeholders = ",".join("?" for _ in _SETTING_KEYS)
        with self.db.get_cursor() as cursor:
            cursor.execute(
                f"SELECT key, value, value_type, type FROM settings WHERE key IN ({placeholders})",
                _SETTING_KEYS,
            )
            return {row["key"]: row for row in cursor.fetchall()}

    def test_nothing_is_seeded_before_the_migration_runs(self):
        self.assertEqual(self._settings(), {})

    def test_the_three_windows_are_seeded_as_admin_settings(self):
        self.migration.up()

        rows = self._settings()
        self.assertEqual(set(rows), set(_SETTING_KEYS))
        for row in rows.values():
            self.assertEqual(row["type"], "SYSTEM")
            self.assertEqual(row["value_type"], "integer")
        self.assertEqual(rows["tmp_retention_days"]["value"], "7")
        self.assertEqual(rows["run_report_retention_days"]["value"], "30")
        self.assertEqual(rows["llm_trace_retention_days"]["value"], "7")

    def test_the_seeded_defaults_are_what_the_worker_reads(self):
        from src.features.housekeeping.settings import DEFAULTS, load_retention
        from src.platform.settings.records import _typed_value, SettingValueType

        self.migration.up()
        rows = self._settings()

        class _Settings:
            @staticmethod
            def get_setting(key, default=None, user_id=None):
                row = rows.get(key)
                if row is None:
                    return default
                return _typed_value(row["value"], SettingValueType(row["value_type"]))

        self.assertEqual(load_retention(_Settings()), DEFAULTS)

    def test_a_second_run_changes_nothing(self):
        self.migration.up()

        self.migration.up()

        self.assertEqual(len(self._settings()), len(_SETTING_KEYS))

    def test_a_second_run_does_not_reset_an_edited_window(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = '0' WHERE key = 'tmp_retention_days'")

        self.migration.up()

        self.assertEqual(self._settings()["tmp_retention_days"]["value"], "0")


if __name__ == "__main__":
    unittest.main()

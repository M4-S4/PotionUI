from tests.platform.database._support import MigrationCase


class TestMigration063PresetPickerState(MigrationCase):
    MIGRATION = "063_preset_picker_state"

    def _rows(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT key, value, value_type, type FROM settings "
                "WHERE key IN ('preset_picker_pinned', 'preset_picker_recent') ORDER BY key"
            )
            return [tuple(row) for row in cursor.fetchall()]

    def test_up_seeds_both_lists_as_empty_per_user_json_settings(self):
        self.migration.up()

        self.assertEqual(
            self._rows(),
            [
                ("preset_picker_pinned", "[]", "json", "USER"),
                ("preset_picker_recent", "[]", "json", "USER"),
            ],
        )

    def test_up_is_idempotent_and_keeps_existing_values(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = '[\"sdxl\"]' WHERE key = 'preset_picker_pinned'")
        self.migration.up()

        self.assertEqual(len(self._rows()), 2)
        self.assertEqual(self._rows()[0][1], '["sdxl"]')

"""020 adds the thumbnail-profile column to `files` and `uploads` and seeds the
five thumbnail settings - both idempotently, since the migration runner replays
nothing but a second `up()` must still be harmless.
"""

import unittest

from tests.platform.database._support import MigrationCase

_SETTING_KEYS = (
    "thumbnail_sizes",
    "thumbnail_video_fps",
    "thumbnail_video_seconds",
    "thumbnail_video_quality",
    "thumbnail_image_quality",
)


class TestMigration020ThumbnailProfiles(MigrationCase):
    MIGRATION = "020_thumbnail_profiles"

    def _columns(self, table):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

    def _settings(self):
        placeholders = ",".join("?" for _ in _SETTING_KEYS)
        with self.db.get_cursor() as cursor:
            cursor.execute(
                f"SELECT key, value, value_type, type FROM settings WHERE key IN ({placeholders})",
                _SETTING_KEYS,
            )
            return cursor.fetchall()

    def test_the_profile_column_lands_on_both_tables(self):
        self.assertNotIn("thumbnail_profile", self._columns("files"))
        self.assertNotIn("thumbnail_profile", self._columns("uploads"))

        self.migration.up()

        self.assertIn("thumbnail_profile", self._columns("files"))
        self.assertIn("thumbnail_profile", self._columns("uploads"))

    def test_existing_rows_read_back_with_a_null_profile(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO files (id, file_path, file_type) VALUES ('f1', 'a/0.png', 'IMAGE')"
            )
            cursor.execute("SELECT thumbnail_profile FROM files WHERE id = 'f1'")
            self.assertIsNone(cursor.fetchone()["thumbnail_profile"])

    def test_the_five_settings_are_seeded_as_admin_settings(self):
        self.migration.up()

        rows = {row["key"]: row for row in self._settings()}
        self.assertEqual(set(rows), set(_SETTING_KEYS))
        for row in rows.values():
            self.assertEqual(row["type"], "SYSTEM")
        self.assertEqual(rows["thumbnail_sizes"]["value"], '["medium"]')
        self.assertEqual(rows["thumbnail_sizes"]["value_type"], "json")
        self.assertEqual(rows["thumbnail_video_fps"]["value"], "12")
        self.assertEqual(rows["thumbnail_video_seconds"]["value"], "3")
        self.assertEqual(rows["thumbnail_video_quality"]["value"], "50")
        self.assertEqual(rows["thumbnail_image_quality"]["value"], "85")
        for key in _SETTING_KEYS[1:]:
            self.assertEqual(rows[key]["value_type"], "integer")

    def test_the_seeded_defaults_load_as_the_balanced_profile(self):
        from src.features.generation.thumbnail_profile import PROFILES, load_thumbnail_profile
        from src.platform.settings.records import _typed_value, SettingValueType

        self.migration.up()

        rows = {row["key"]: row for row in self._settings()}

        class _Settings:
            @staticmethod
            def get_setting(key, default=None, user_id=None):
                row = rows.get(key)
                if row is None:
                    return default
                return _typed_value(row["value"], SettingValueType(row["value_type"]))

        self.assertEqual(load_thumbnail_profile(_Settings()), PROFILES["balanced"])

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before_files = self._columns("files")
        before_uploads = self._columns("uploads")

        self.migration.up()

        self.assertEqual(self._columns("files"), before_files)
        self.assertEqual(self._columns("uploads"), before_uploads)
        self.assertEqual(len(self._settings()), len(_SETTING_KEYS))

    def test_a_second_run_does_not_reset_an_edited_setting(self):
        self.migration.up()
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = '8' WHERE key = 'thumbnail_video_fps'")

        self.migration.up()

        rows = {row["key"]: row for row in self._settings()}
        self.assertEqual(rows["thumbnail_video_fps"]["value"], "8")


if __name__ == "__main__":
    unittest.main()

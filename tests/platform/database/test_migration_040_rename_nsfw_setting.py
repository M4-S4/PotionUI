from tests.platform.database._support import MigrationCase


class TestMigration040RenameNsfwSetting(MigrationCase):
    MIGRATION = "040_rename_nsfw_setting"

    def _row(self, key):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT id, value, value_type, type, description FROM settings WHERE key = ?",
                (key,),
            )
            return cursor.fetchone()

    def test_up_renames_row_keeping_value_type_and_scope(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE settings SET value = 'true' WHERE key = 'nsfw_filter'")
        self.migration.up()

        self.assertIsNone(self._row("nsfw_filter"))
        row = self._row("nsfw")
        self.assertEqual(row["value"], "true")
        self.assertEqual(row["value_type"], "boolean")
        self.assertEqual(row["type"], "USER")
        self.assertEqual(row["description"], "Allow NSFW content generation")

    def test_up_keeps_existing_user_overrides(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("SELECT id FROM settings WHERE key = 'nsfw_filter'")
            setting_id = cursor.fetchone()["id"]
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) VALUES ('u1', 'u1', 'u1@x', 'h')"
            )
            cursor.execute(
                "INSERT INTO user_settings (id, user_id, setting_id, value) VALUES ('us1', 'u1', ?, 'true')",
                (setting_id,),
            )
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT us.value FROM user_settings us JOIN settings s ON s.id = us.setting_id "
                "WHERE s.key = 'nsfw' AND us.user_id = 'u1'"
            )
            self.assertEqual(cursor.fetchone()["value"], "true")

    def test_up_is_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertIsNotNone(self._row("nsfw"))
        self.assertIsNone(self._row("nsfw_filter"))

    def test_up_drops_legacy_row_when_target_already_exists(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO settings (id, key, value, value_type, description, type) "
                "VALUES ('setting_nsfw', 'nsfw', 'true', 'boolean', 'x', 'USER')"
            )
        self.migration.up()

        self.assertIsNone(self._row("nsfw_filter"))
        self.assertEqual(self._row("nsfw")["value"], "true")

from tests.platform.database._support import MigrationCase


class TestMigration039SessionPinned(MigrationCase):
    MIGRATION = "039_session_pinned"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA foreign_keys = OFF")
            cursor.execute(
                "INSERT INTO sessions (id, user_id, preset_id, name, data) VALUES ('old', 'u', 'p', 'Old', '{}')"
            )

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(sessions)")
            return {row["name"] for row in cursor.fetchall()}

    def test_existing_sessions_come_through_unpinned(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT pinned, pinned_at FROM sessions WHERE id = 'old'")
            row = cursor.fetchone()
        self.assertEqual(row["pinned"], 0)
        self.assertIsNone(row["pinned_at"])

    def test_up_is_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertLessEqual({"pinned", "pinned_at"}, self._columns())

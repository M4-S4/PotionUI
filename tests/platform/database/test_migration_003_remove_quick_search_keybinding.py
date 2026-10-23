"""003 deletes the `quick_search` `keybinding_defaults` row for databases that
already ran `001_baseline.py` before it stopped seeding that row. The
interesting behavior is the cascade: a user who had customized `quick_search`
has a matching `user_keybindings` row, and that row must disappear with it
rather than being left pointing at a deleted default.
"""

import unittest

from tests.platform.database._support import MigrationCase


class TestMigration003RemoveQuickSearchKeybinding(MigrationCase):
    MIGRATION = "003_remove_quick_search_keybinding"

    def pre_state(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO keybinding_defaults
                    (id, key, modifiers, label, category, context, description, enabled, source, sort_order)
                VALUES ('quick_search', '/', '', 'Quick Search', 'general', 'global', 'Open quick search dialog', 1, 'system', 2)
                """
            )
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('u1', 'u1', 'u1@test.com', 'x')"
            )
            cursor.executemany(
                "INSERT INTO user_keybindings (user_id, action_id, key, modifiers) VALUES (?, ?, ?, ?)",
                [
                    ("u1", "quick_search", "k", "ctrl"),
                    ("u1", "show_help", "h", "ctrl"),
                ],
            )

    def test_removes_the_default_row(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM keybinding_defaults WHERE id = 'quick_search'"
            ).fetchone()
        self.assertIsNone(row)

    def test_cascades_to_user_overrides(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            action_ids = {
                row[0] for row in conn.execute(
                    "SELECT action_id FROM user_keybindings WHERE user_id = 'u1'"
                ).fetchall()
            }
        self.assertNotIn("quick_search", action_ids)
        self.assertIn("show_help", action_ids)

    def test_idempotent(self):
        self.migration.up()
        self.migration.up()

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT id FROM keybinding_defaults WHERE id = 'quick_search'"
            ).fetchone()
        self.assertIsNone(row)


if __name__ == '__main__':
    unittest.main()

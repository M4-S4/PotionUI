import unittest

from tests.platform.database._support import MigrationCase

_TABLES = ("prompt_segments", "saved_segments", "segment_template_segments")


class TestMigration030SegmentResources(MigrationCase):
    MIGRATION = "030_segment_resources"

    def _columns(self, table):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

    def test_resources_column_lands_on_all_three_tables(self):
        for table in _TABLES:
            self.assertNotIn("resources", self._columns(table))

        self.migration.up()

        for table in _TABLES:
            self.assertIn("resources", self._columns(table))

    def test_existing_rows_read_back_with_an_empty_map(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('user1', 'tester', 'tester@example.com', 'hashed')"
            )
            cursor.execute(
                "INSERT INTO segment_categories (id, user_id, name) VALUES ('cat1', 'user1', 'Category')"
            )
            cursor.execute(
                "INSERT INTO saved_segments (id, user_id, category_id, name) VALUES ('s1', 'user1', 'cat1', 'Segment')"
            )

        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT resources FROM saved_segments WHERE id = 's1'")
            self.assertEqual(cursor.fetchone()["resources"], "{}")

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = {table: self._columns(table) for table in _TABLES}

        self.migration.up()

        for table in _TABLES:
            self.assertEqual(self._columns(table), before[table])


if __name__ == "__main__":
    unittest.main()

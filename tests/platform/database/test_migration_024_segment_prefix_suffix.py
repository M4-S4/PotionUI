"""024 adds `prefix`/`suffix` to the three RichSegment child tables -
idempotently, since the migration runner replays nothing but a second
`up()` must still be harmless.
"""

import unittest

from tests.platform.database._support import MigrationCase

_TABLES = ("prompt_segments", "saved_segments", "segment_template_segments")


class TestMigration024SegmentPrefixSuffix(MigrationCase):
    MIGRATION = "024_segment_prefix_suffix"

    def _columns(self, table):
        with self.db.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table})")
            return {row[1] for row in cursor.fetchall()}

    def test_prefix_and_suffix_columns_land_on_all_three_tables(self):
        for table in _TABLES:
            self.assertNotIn("prefix", self._columns(table))
            self.assertNotIn("suffix", self._columns(table))

        self.migration.up()

        for table in _TABLES:
            self.assertIn("prefix", self._columns(table))
            self.assertIn("suffix", self._columns(table))

    def test_existing_rows_read_back_with_null_prefix_and_suffix(self):
        self.migration.up()

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (id, username, email, password_hash) "
                "VALUES ('user1', 'tester', 'tester@example.com', 'hashed')"
            )
            cursor.execute(
                "INSERT INTO segment_categories (id, user_id, name) "
                "VALUES ('cat1', 'user1', 'Category')"
            )
            cursor.execute(
                "INSERT INTO saved_segments (id, user_id, category_id, name) "
                "VALUES ('s1', 'user1', 'cat1', 'Segment')"
            )
            cursor.execute(
                "SELECT prefix, suffix FROM saved_segments WHERE id = 's1'"
            )
            row = cursor.fetchone()
            self.assertIsNone(row["prefix"])
            self.assertIsNone(row["suffix"])

    def test_a_second_run_changes_nothing(self):
        self.migration.up()
        before = {table: self._columns(table) for table in _TABLES}

        self.migration.up()

        for table in _TABLES:
            self.assertEqual(self._columns(table), before[table])


if __name__ == "__main__":
    unittest.main()

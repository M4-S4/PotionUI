"""007 adds the liveness columns to `provisioned_compute` and seeds the
heartbeat interval setting - both idempotently, since the migration runner
replays nothing but a second `up()` must still be harmless.
"""

from tests.platform.database._support import MigrationCase


class TestMigration007ProvisionedComputeLiveness(MigrationCase):
    MIGRATION = "007_provisioned_compute_liveness"
    PREREQUISITES = ("001_baseline", "004_provisioned_compute")

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(provisioned_compute)")
            return {row[1]: row for row in cursor.fetchall()}

    def _interval_setting(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT value, value_type, type FROM settings WHERE key = 'provisioning.status_interval_seconds'"
            )
            return cursor.fetchall()

    def test_adds_the_columns_with_progress_defaulting_to_an_empty_list(self):
        self.migration.up()

        columns = self._columns()
        self.assertIn("status_detail", columns)
        self.assertIn("status_checked_at", columns)
        self.assertIn("progress", columns)

        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO provisioned_compute (id, provider_id, handle, profile_name, status) "
                "VALUES ('r1', 'fake', '', 'p', 'provisioning')"
            )
            cursor.execute("SELECT progress, status_detail, status_checked_at FROM provisioned_compute WHERE id = 'r1'")
            row = cursor.fetchone()
        self.assertEqual(row["progress"], "[]")
        self.assertIsNone(row["status_detail"])
        self.assertIsNone(row["status_checked_at"])

    def test_seeds_the_interval_setting_once(self):
        self.migration.up()

        rows = self._interval_setting()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "15")
        self.assertEqual(rows[0]["value_type"], "integer")
        self.assertEqual(rows[0]["type"], "SYSTEM")

    def test_second_run_is_a_no_op(self):
        self.migration.up()
        before = set(self._columns())

        self.migration.up()

        self.assertEqual(set(self._columns()), before)
        self.assertEqual(len(self._interval_setting()), 1)

from tests.platform.database._support import MigrationCase


class TestMigration053Plans(MigrationCase):
    MIGRATION = "053_plans"

    def _query(self, sql, params=()):
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def _columns(self, table):
        return {row["name"] for row in self._query(f"PRAGMA table_info({table})")}

    def test_up_creates_the_tables_columns_unlimited_plan_and_settings(self):
        self.migration.up()

        self.assertEqual(
            self._query("SELECT id, name, limits_json, is_system FROM plans"),
            [{"id": "unlimited", "name": "Unlimited", "limits_json": "[]", "is_system": 1}],
        )
        self.assertIn("plan_id", self._columns("users"))
        self.assertIn("plan_id", self._columns("user_groups"))
        self.assertTrue({"user_id", "kind", "units", "ref_id", "refunded_at"} <= self._columns("limit_events"))
        settings = {row["key"]: row["value"] for row in self._query("SELECT key, value FROM settings WHERE key LIKE 'plans_%'")}
        self.assertEqual(settings, {
            "plans_exempt_admins": "true",
            "plans_day_timezone": "UTC",
            "plans_contact_line": "Ask your admin for more.",
        })

    def test_up_is_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertEqual(len(self._query("SELECT id FROM plans")), 1)

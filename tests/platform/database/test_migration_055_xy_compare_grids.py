from tests.platform.database._support import MigrationCase


class TestMigration055XyCompareGrids(MigrationCase):
    MIGRATION = "055_xy_compare_grids"

    def _columns(self, table):
        with self.db.get_connection() as conn:
            return {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}

    def _object_exists(self, kind, name):
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?", (kind, name)
            ).fetchone() is not None

    def test_baseline_has_no_grid_columns_or_table(self):
        self.assertNotIn("grid_id", self._columns("generations"))
        self.assertFalse(self._object_exists("table", "generation_grids"))

    def test_up_adds_nullable_grid_columns_to_generations(self):
        self.migration.up()

        columns = self._columns("generations")
        for name in ("grid_id", "grid_x", "grid_y", "axis_values"):
            self.assertIn(name, columns)
            self.assertEqual(columns[name][3], 0)
            self.assertIsNone(columns[name][4])

    def test_up_creates_the_grids_table_and_the_grid_index(self):
        self.migration.up()

        self.assertEqual(
            set(self._columns("generation_grids")),
            {"id", "user_id", "preset_id", "tab_id", "x_axis", "y_axis", "lock_seed", "base_request", "seeds", "created_at"},
        )
        self.assertTrue(self._object_exists("index", "idx_generations_grid"))
        with self.db.get_connection() as conn:
            indexed = [row[2] for row in conn.execute("PRAGMA index_info(idx_generations_grid)")]
        self.assertEqual(indexed[0], "grid_id")

    def test_up_seeds_the_confirm_threshold_setting_at_24(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT value, value_type, type FROM settings WHERE key = 'compare_confirm_above'"
            ).fetchone()
        self.assertEqual(tuple(row), ("24", "integer", "SYSTEM"))

    def test_up_keeps_an_admins_threshold_on_rerun(self):
        self.migration.up()
        with self.db.get_connection() as conn:
            conn.execute("UPDATE settings SET value = '50' WHERE key = 'compare_confirm_above'")
            conn.commit()

        self.migration.up()

        with self.db.get_connection() as conn:
            rows = conn.execute("SELECT value FROM settings WHERE key = 'compare_confirm_above'").fetchall()
        self.assertEqual([tuple(row) for row in rows], [("50",)])

    def test_up_is_idempotent(self):
        self.migration.up()
        self.migration.up()

        self.assertIn("grid_id", self._columns("generations"))

    def test_existing_generations_survive_with_null_grid_fields(self):
        with self.db.get_connection() as conn:
            conn.execute(
                "INSERT INTO generations (id, preset_id, form_data, user_id, status) VALUES ('g', 'p', '{}', NULL, 'completed')"
            )
            conn.commit()

        self.migration.up()

        with self.db.get_connection() as conn:
            row = conn.execute("SELECT grid_id, grid_x, grid_y, axis_values FROM generations WHERE id = 'g'").fetchone()
        self.assertEqual(tuple(row), (None, None, None, None))

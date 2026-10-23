from tests.platform.database._support import MigrationCase


class TestMigration047GenerationCosts(MigrationCase):
    MIGRATION = "047_generation_costs"

    def _columns(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("PRAGMA table_info(generation_costs)")
            return {row["name"]: row for row in cursor.fetchall()}

    def _insert(self, cost_id="c1", generation_id="gone"):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO generation_costs (id, generation_id, source, created_at) VALUES (?, ?, 'provider', 't')",
                (cost_id, generation_id),
            )

    def _count(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM generation_costs")
            return cursor.fetchone()["n"]

    def test_creates_the_cost_table_with_its_columns(self):
        self.migration.up()

        assert set(self._columns()) == {
            "id", "generation_id", "backend_id", "model_id", "user_id", "amount_usd", "source", "detail", "created_at",
        }
        assert self._columns()["amount_usd"]["type"] == "TEXT"

    def test_a_cost_row_does_not_need_its_generation_so_it_outlives_it(self):
        self.migration.up()

        self._insert(generation_id="never-existed")

        assert self._count() == 1

    def test_a_row_may_have_no_amount(self):
        self.migration.up()
        self._insert()

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT amount_usd, detail FROM generation_costs")
            row = cursor.fetchone()
        assert row["amount_usd"] is None and row["detail"] == "{}"

    def test_up_twice_keeps_the_rows(self):
        self.migration.up()
        self._insert()

        self.migration.up()

        assert self._count() == 1

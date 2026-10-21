"""014 adds the composite index the history list orders by. `001_baseline.py` is
a frozen snapshot that only indexes `generations(user_id)`, so this migration is
what puts the index there on a fresh install as well as an upgrade.
"""

from tests.platform.database._support import MigrationCase

_INDEX = "idx_generations_user_created_id"


class TestMigration014GenerationsUserCreatedIndex(MigrationCase):
    MIGRATION = "014_generations_user_created_index"

    def _index_sql(self):
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (_INDEX,),
            ).fetchone()
        return row[0] if row else None

    def test_baseline_alone_has_no_such_index(self):
        self.assertIsNone(self._index_sql())

    def test_creates_the_index_over_user_created_and_id(self):
        self.migration.up()

        sql = self._index_sql()
        self.assertIsNotNone(sql)
        normalized = " ".join(sql.split()).lower()
        self.assertIn("on generations (user_id, created_at desc, id desc)", normalized)

    def test_index_columns_are_in_key_order(self):
        self.migration.up()

        with self.db.get_connection() as conn:
            columns = [row[2] for row in conn.execute(f"PRAGMA index_info({_INDEX})")]
        self.assertEqual(columns, ["user_id", "created_at", "id"])

    def test_idempotent(self):
        self.migration.up()
        self.migration.up()

        with self.db.get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = ?",
                (_INDEX,),
            ).fetchone()[0]
        self.assertEqual(count, 1)

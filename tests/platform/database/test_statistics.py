import tempfile
import threading
import unittest
from pathlib import Path

from src.platform.database.database import Database
from src.platform.database.statistics import (
    HOUSEKEEPING_ANALYSIS_LIMIT,
    analyze,
    analyze_if_missing,
    has_statistics,
    refresh_statistics,
)


class TestPlannerStatistics(unittest.TestCase):

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        Database._instance = None
        self.addCleanup(setattr, Database, "_instance", None)
        self.db = Database()
        self.db.db_path = Path(temp.name) / "test.sqlite"
        self.db._initialized = True
        with self.db.get_cursor() as cursor:
            cursor.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, kind TEXT)")
            cursor.execute("CREATE INDEX idx_items_kind ON items (kind)")
        self._add_rows(200)

    def _add_rows(self, count):
        with self.db.get_cursor() as cursor:
            cursor.executemany(
                "INSERT INTO items (kind) VALUES (?)", [(f"k{i % 4}",) for i in range(count)]
            )

    def _stat(self):
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT stat FROM sqlite_stat1 WHERE idx = 'idx_items_kind'"
            ).fetchone()[0]

    def _schema_version(self):
        with self.db.get_connection() as conn:
            return conn.execute("PRAGMA schema_version").fetchone()[0]

    def test_analyzes_when_the_statistics_table_is_missing(self):
        with self.db.get_connection() as conn:
            assert has_statistics(conn) is False

        assert analyze_if_missing(self.db) is True

        assert self._stat() == "200 50"

    def test_does_not_analyze_again_once_statistics_exist(self):
        analyze_if_missing(self.db)
        self._add_rows(200)

        assert analyze_if_missing(self.db) is False

        assert self._stat() == "200 50"

    def test_refresh_analyzes_again_over_existing_statistics(self):
        analyze_if_missing(self.db)
        self._add_rows(200)

        refresh_statistics(self.db)

        assert self._stat() == "400 100"

    def test_only_the_analyze_that_creates_the_statistics_table_changes_the_schema(self):
        before = self._schema_version()

        analyze_if_missing(self.db)
        created = self._schema_version()
        refresh_statistics(self.db)

        assert created == before + 1
        assert self._schema_version() == created

    def test_the_housekeeping_limit_does_not_stay_on_the_connection(self):
        with self.db.get_connection() as conn:
            analyze(conn, HOUSEKEEPING_ANALYSIS_LIMIT)
            assert conn.execute("PRAGMA analysis_limit").fetchone()[0] == HOUSEKEEPING_ANALYSIS_LIMIT

            analyze(conn)

            assert conn.execute("PRAGMA analysis_limit").fetchone()[0] == 0

    def test_runs_while_a_reader_holds_a_snapshot(self):
        with self.db.get_connection() as reader:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM items").fetchall()

            refresh_statistics(self.db)

            reader.rollback()
        assert self._stat() == "200 50"

    def test_waits_for_a_writer_instead_of_failing(self):
        finished = threading.Event()
        errors = []

        def run():
            try:
                refresh_statistics(self.db)
            except Exception as e:
                errors.append(e)
            finally:
                finished.set()

        with self.db.get_connection() as writer:
            writer.execute("BEGIN IMMEDIATE")
            thread = threading.Thread(target=run)
            thread.start()
            assert finished.wait(0.2) is False
            writer.commit()
        thread.join(5)

        assert finished.is_set()
        assert errors == []
        assert self._stat() == "200 50"


if __name__ == "__main__":
    unittest.main()

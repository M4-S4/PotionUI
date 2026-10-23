from unittest.mock import patch

import pytest

from src.bootstrap import app as bootstrap_app
from tests.fixtures.persistence_base import PersistenceTestBase


class _ContainerNotBuilt(Exception):
    pass


def test_create_app_collects_statistics_right_after_the_migrations():
    calls = []

    with patch.object(bootstrap_app, "run_migrations_sync", lambda: calls.append("migrations")), \
            patch.object(bootstrap_app, "ensure_database_statistics", lambda: calls.append("statistics")), \
            patch.object(bootstrap_app, "apply_startup_env_overrides", lambda: calls.append("overrides")), \
            patch.object(bootstrap_app, "build_container", side_effect=_ContainerNotBuilt):
        with pytest.raises(_ContainerNotBuilt):
            bootstrap_app.create_app()

    assert calls == ["migrations", "statistics", "overrides"]


class TestEnsureDatabaseStatistics(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        with self.db.get_cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS sqlite_stat1")
        self.create_test_user()
        self._add_generations(0, 3)

    def _add_generations(self, start, stop):
        for i in range(start, stop):
            self.create_test_generation(generation_id=f"gen-{i}")

    def _generation_rows_seen(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "SELECT stat FROM sqlite_stat1 WHERE idx = 'idx_generations_user_created_id'"
            )
            return int(cursor.fetchone()[0].split()[0])

    def test_collects_statistics_when_the_database_has_none(self):
        bootstrap_app.ensure_database_statistics()

        self.assertEqual(self._generation_rows_seen(), 3)

    def test_leaves_statistics_that_already_exist_alone(self):
        bootstrap_app.ensure_database_statistics()
        self._add_generations(3, 5)

        bootstrap_app.ensure_database_statistics()

        self.assertEqual(self._generation_rows_seen(), 3)

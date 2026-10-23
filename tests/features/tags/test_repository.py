"""Tests for TagRepository's batched generation-tag lookup."""
import importlib
import json
import os
import sys
from contextlib import contextmanager

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tests.fixtures.persistence_base import PersistenceTestBase
from src.features.tags.repository import TagRepository


def _count_cursor_executes(db_instance, fn):
    """Run `fn()` and return how many `cursor.execute` calls it issued.

    `sqlite3.Cursor` is a C type and refuses attribute patching, so the count
    is taken by wrapping the `Database.get_cursor()` context manager instead
    of the cursor class itself.
    """
    counter = {"n": 0}
    original_get_cursor = db_instance.get_cursor

    class _CountingCursor:
        def __init__(self, cursor):
            self._cursor = cursor

        def execute(self, *args, **kwargs):
            counter["n"] += 1
            return self._cursor.execute(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    @contextmanager
    def counting_get_cursor():
        with original_get_cursor() as cursor:
            yield _CountingCursor(cursor)

    db_instance.get_cursor = counting_get_cursor
    try:
        fn()
    finally:
        db_instance.get_cursor = original_get_cursor
    return counter["n"]


_MEASURED_TAG_STATISTICS = (
    ("generation_tags", "idx_generation_tags_tag_id", "563 9"),
    ("generation_tags", "idx_generation_tags_generation_id", "563 2"),
    ("generation_tags", "sqlite_autoindex_generation_tags_1", "563 2 1"),
    ("tags", "idx_tags_type_user", "80 27 14"),
    ("tags", "idx_tags_name_type_user", "80 1 1 1"),
    ("tags", "idx_tags_user_id", "80 16"),
    ("tags", "idx_tags_name", "80 1"),
    ("tags", "sqlite_autoindex_tags_2", "80 1"),
    ("tags", "sqlite_autoindex_tags_1", "80 1"),
)


class TestTagRepositoryGenerationTagsBulk(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        # PersistenceTestBase only redirects a fixed module list to the test
        # database; tags.repository isn't one of them, so redirect it here.
        self.tags_module = importlib.import_module("src.features.tags.repository")
        self.tags_module.db = self.db

        self.repo = TagRepository()
        self.test_user_id = self.create_test_user()

    def _create_generation_tag(self, name):
        return self.repo.create_tag(name, type="GENERATION", user_id=self.test_user_id)

    def test_get_generation_tags_bulk_empty_for_generation_with_no_tags(self):
        gen_with_tag = self.create_test_generation(generation_id="gen_tagged", user_id=self.test_user_id)
        gen_without_tag = self.create_test_generation(generation_id="gen_untagged", user_id=self.test_user_id)
        tag = self._create_generation_tag("keeper")
        self.repo.add_tag_to_generation(gen_with_tag, tag.id)

        bulk = self.repo.get_generation_tags_bulk([gen_with_tag, gen_without_tag])

        self.assertEqual([t.id for t in bulk[gen_with_tag]], [tag.id])
        self.assertEqual(bulk[gen_without_tag], [])

    def test_get_generation_tags_bulk_matches_single_lookup(self):
        gen_a = self.create_test_generation(generation_id="gen_a", user_id=self.test_user_id)
        gen_b = self.create_test_generation(generation_id="gen_b", user_id=self.test_user_id)
        tag1 = self._create_generation_tag("alpha")
        tag2 = self._create_generation_tag("beta")
        self.repo.add_tag_to_generation(gen_a, tag1.id)
        self.repo.add_tag_to_generation(gen_a, tag2.id)
        self.repo.add_tag_to_generation(gen_b, tag2.id)

        expected_a = [t.id for t in self.repo.get_generation_tags(gen_a)]
        expected_b = [t.id for t in self.repo.get_generation_tags(gen_b)]

        bulk = self.repo.get_generation_tags_bulk([gen_a, gen_b])

        self.assertEqual([t.id for t in bulk[gen_a]], expected_a)
        self.assertEqual([t.id for t in bulk[gen_b]], expected_b)

    def test_get_generation_tags_bulk_issues_constant_query_count(self):
        """The regression guard: query count must not scale with generation
        count, which is exactly the N+1 the batching replaced."""

        def make_tagged_generation(gen_id):
            generation_id = self.create_test_generation(generation_id=gen_id, user_id=self.test_user_id)
            tag = self._create_generation_tag(f"tag_{gen_id}")
            self.repo.add_tag_to_generation(generation_id, tag.id)
            return generation_id

        small_ids = [make_tagged_generation(f"small_{i}") for i in range(3)]
        large_ids = [make_tagged_generation(f"large_{i}") for i in range(40)]

        small_query_count = _count_cursor_executes(
            self.db, lambda: self.repo.get_generation_tags_bulk(small_ids)
        )
        large_query_count = _count_cursor_executes(
            self.db, lambda: self.repo.get_generation_tags_bulk(large_ids)
        )

        self.assertEqual(small_query_count, large_query_count)
        self.assertEqual(small_query_count, 1)

    def _plan_of_bulk_lookup(self, generation_ids):
        statements = []
        original_get_cursor = self.db.get_cursor

        class _RecordingCursor:
            def __init__(self, cursor):
                self._cursor = cursor

            def execute(self, sql, params=()):
                statements.append((sql, list(params)))
                return self._cursor.execute(sql, params)

            def __getattr__(self, name):
                return getattr(self._cursor, name)

        @contextmanager
        def recording_get_cursor():
            with original_get_cursor() as cursor:
                yield _RecordingCursor(cursor)

        self.db.get_cursor = recording_get_cursor
        try:
            self.repo.get_generation_tags_bulk(generation_ids)
        finally:
            self.db.get_cursor = original_get_cursor
        sql, params = statements[0]
        with self.db.get_cursor() as cursor:
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}", params)
            return [row[3] for row in cursor.fetchall()]

    def test_bulk_lookup_starts_from_the_page_under_measured_statistics(self):
        with self.db.get_cursor() as cursor:
            cursor.execute("ANALYZE")
            cursor.execute("DELETE FROM sqlite_stat1 WHERE tbl IN ('tags', 'generation_tags')")
            cursor.executemany(
                "INSERT INTO sqlite_stat1 (tbl, idx, stat) VALUES (?, ?, ?)", _MEASURED_TAG_STATISTICS
            )

        plan = self._plan_of_bulk_lookup([f"gen_{i}" for i in range(50)])

        self.assertTrue(plan[0].startswith("SEARCH gt"), plan)
        self.assertFalse(any(step.startswith("SCAN t") for step in plan), plan)


class TestTagRepositoryTimestampContract(PersistenceTestBase):
    """A tag's `created_at` must carry a UTC offset wherever it leaves the
    backend - a naive value is parsed as local time by the browser and shows
    a fresh generation as hours old for anyone not in UTC."""

    def setUp(self):
        super().setUp()
        self.tags_module = importlib.import_module("src.features.tags.repository")
        self.tags_module.db = self.db
        self.repo = TagRepository()
        self.test_user_id = self.create_test_user()

    def test_create_tag_returns_offset_aware_created_at(self):
        tag = self.repo.create_tag("regression-tag", type="GENERATION", user_id=self.test_user_id)

        self.assertIsNotNone(tag.created_at.tzinfo)
        wire_value = json.loads(tag.model_dump_json())["created_at"]
        self.assertTrue(wire_value.endswith("+00:00") or wire_value.endswith("Z"))

    def test_row_with_offset_less_stored_timestamp_reads_as_utc(self):
        """A legacy row written before this contract (or via sqlite
        CURRENT_TIMESTAMP) has no offset in `created_at`; it must still be
        read as UTC and re-emitted with an explicit +00:00."""
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO tags (id, name, type, user_id, created_at) VALUES (?, ?, ?, ?, ?)",
                ("tag_legacy", "legacy-tag", "GENERATION", self.test_user_id, "2024-01-01T12:00:00"),
            )

        tag = self.repo.get_tag_by_id("tag_legacy")

        self.assertEqual(tag.created_at.isoformat(), "2024-01-01T12:00:00+00:00")


if __name__ == '__main__':
    import unittest
    unittest.main()

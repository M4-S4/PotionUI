import unittest

from src.features.media_tools.repository import MediaToolParamsRepository
from tests.fixtures.persistence_base import PersistenceTestBase


class TestMediaToolParamsRepository(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        self.repo = MediaToolParamsRepository()
        self.user_id = self.create_test_user()
        self.other_user_id = self.create_test_user("other_user", "other", "other@example.com")

    def tearDown(self):
        self.remove_scratch_files()
        super().tearDown()

    def test_nothing_is_remembered_until_a_run_saves_it(self):
        self.assertIsNone(self.repo.get(self.user_id, "trim"))
        self.assertEqual(self.repo.get_many(self.user_id, ["trim", "crop"]), {})

    def test_save_then_get_round_trips_json_values(self):
        params = {"start": 1.5, "end": 4, "keep_pitch": True, "fill": "#aabbcc", "anchor": None}

        self.repo.save(self.user_id, "trim", params)

        self.assertEqual(self.repo.get(self.user_id, "trim"), params)

    def test_saving_again_replaces_the_previous_values_without_a_second_row(self):
        self.repo.save(self.user_id, "trim", {"start": 1})
        self.repo.save(self.user_id, "trim", {"start": 2})

        self.assertEqual(self.repo.get(self.user_id, "trim"), {"start": 2})
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS count FROM media_tool_params")
            self.assertEqual(cursor.fetchone()["count"], 1)

    def test_values_are_kept_per_user_and_per_tool(self):
        self.repo.save(self.user_id, "trim", {"start": 1})
        self.repo.save(self.other_user_id, "trim", {"start": 9})
        self.repo.save(self.user_id, "crop", {"x": 3})

        self.assertEqual(self.repo.get(self.user_id, "trim"), {"start": 1})
        self.assertEqual(self.repo.get(self.other_user_id, "trim"), {"start": 9})
        self.assertEqual(self.repo.get(self.user_id, "crop"), {"x": 3})

    def test_get_many_returns_only_the_requested_tools_of_that_user(self):
        self.repo.save(self.user_id, "trim", {"start": 1})
        self.repo.save(self.user_id, "crop", {"x": 3})
        self.repo.save(self.user_id, "flip", {"axis": "vertical"})
        self.repo.save(self.other_user_id, "trim", {"start": 9})

        found = self.repo.get_many(self.user_id, ["trim", "flip", "speed"])

        self.assertEqual(found, {"trim": {"start": 1}, "flip": {"axis": "vertical"}})
        self.assertEqual(self.repo.get_many(self.user_id, []), {})

    def test_a_row_with_unreadable_json_reads_as_empty_params(self):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO media_tool_params (user_id, tool_id, params_json, updated_at) VALUES (?, ?, ?, ?)",
                (self.user_id, "trim", "{broken", "2026-01-01T00:00:00+00:00"),
            )

        self.assertEqual(self.repo.get(self.user_id, "trim"), {})

    def test_deleting_the_user_removes_their_remembered_params(self):
        self.repo.save(self.user_id, "trim", {"start": 1})
        self.repo.save(self.other_user_id, "trim", {"start": 9})

        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = ?", (self.user_id,))

        self.assertIsNone(self.repo.get(self.user_id, "trim"))
        self.assertEqual(self.repo.get(self.other_user_id, "trim"), {"start": 9})


if __name__ == "__main__":
    unittest.main()

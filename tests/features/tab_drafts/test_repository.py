from datetime import datetime, timedelta, timezone

from src.features.tab_drafts.errors import TabDraftLimitReached
from src.features.tab_drafts.manager import TabDraftManager
from src.features.tab_drafts.repository import TabDraftRepository
from tests.fixtures.persistence_base import PersistenceTestBase

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def text(value):
    return {"type": "text", "value": value}


def set_op(name, definition):
    op = {"op": "set", "name": name}
    op.update({("pinned_index" if k == "pinnedIndex" else k): v for k, v in definition.items()})
    return op


class TabDraftStoreTestBase(PersistenceTestBase):

    def setUp(self):
        super().setUp()
        self.repository = TabDraftRepository()
        self.create_test_user("u1", "u1", "u1@example.com")
        self.create_test_user("u2", "u2", "u2@example.com")


class TestRepository(TabDraftStoreTestBase):

    def test_first_write_creates_the_row_at_rev_one(self):
        draft = self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0)

        self.assertEqual(draft.rev, 1)
        self.assertEqual(draft.created_at, T0)
        self.assertEqual(draft.updated_at, T0)
        self.assertEqual(self.repository.get("u1", "t1").doc, {"variables": {"a": text("x")}})

    def test_every_write_bumps_rev_and_keeps_created_at(self):
        self.repository.replace("u1", "t1", {"variables": {}}, now=T0)
        second = self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0 + timedelta(hours=1))
        third = self.repository.mutate("u1", "t1", lambda doc: doc, now=T0 + timedelta(hours=2))

        self.assertEqual((second.rev, third.rev), (2, 3))
        stored = self.repository.get("u1", "t1")
        self.assertEqual(stored.rev, 3)
        self.assertEqual(stored.created_at, T0)
        self.assertEqual(stored.updated_at, T0 + timedelta(hours=2))

    def test_mutate_sees_the_stored_doc_and_stores_what_it_returns(self):
        self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0)

        def add(doc):
            return {"variables": {**doc["variables"], "b": text("y")}}

        draft = self.repository.mutate("u1", "t1", add, now=T0)

        self.assertEqual(set(draft.doc["variables"]), {"a", "b"})

    def test_mutate_without_create_leaves_a_missing_row_missing(self):
        result = self.repository.mutate("u1", "t1", lambda doc: doc, now=T0, create=False)

        self.assertIsNone(result)
        self.assertIsNone(self.repository.get("u1", "t1"))

    def test_a_failing_mutation_rolls_back_and_leaves_rev_alone(self):
        self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0)

        def boom(doc):
            raise RuntimeError("no")

        with self.assertRaises(RuntimeError):
            self.repository.mutate("u1", "t1", boom, now=T0)

        stored = self.repository.get("u1", "t1")
        self.assertEqual(stored.rev, 1)
        self.assertEqual(stored.doc, {"variables": {"a": text("x")}})

    def test_a_failing_mutation_on_a_new_draft_creates_nothing(self):
        def boom(doc):
            raise RuntimeError("no")

        with self.assertRaises(RuntimeError):
            self.repository.mutate("u1", "t1", boom, now=T0)

        self.assertIsNone(self.repository.get("u1", "t1"))

    def test_other_users_do_not_see_the_draft(self):
        self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0)

        self.assertIsNone(self.repository.get("u2", "t1"))
        self.assertEqual(self.repository.list_summaries("u2"), [])
        self.assertFalse(self.repository.delete("u2", "t1"))
        self.assertIsNotNone(self.repository.get("u1", "t1"))

    def test_two_users_may_hold_the_same_tab_id(self):
        self.repository.replace("u1", "t1", {"variables": {"a": text("one")}}, now=T0)
        self.repository.replace("u2", "t1", {"variables": {"a": text("two")}}, now=T0)

        self.assertEqual(self.repository.get("u1", "t1").doc["variables"]["a"]["value"], "one")
        self.assertEqual(self.repository.get("u2", "t1").doc["variables"]["a"]["value"], "two")

    def test_writing_as_one_user_never_touches_the_others_row(self):
        self.repository.replace("u1", "t1", {"variables": {"a": text("one")}}, now=T0)
        self.repository.mutate("u2", "t1", lambda doc: {"variables": {"b": text("two")}}, now=T0)

        self.assertEqual(self.repository.get("u1", "t1").rev, 1)
        self.assertEqual(set(self.repository.get("u1", "t1").doc["variables"]), {"a"})

    def test_summaries_carry_counts_and_order_newest_first(self):
        self.repository.replace("u1", "old", {"variables": {"a": text("x")}}, now=T0)
        self.repository.replace(
            "u1", "new", {"variables": {"a": text("x"), "b": text("y")}}, now=T0 + timedelta(days=1)
        )
        self.repository.replace("u1", "empty", {"variables": {}}, now=T0 - timedelta(days=1))
        self.repository.mutate(
            "u1", "old", lambda doc: {"variables": {"a": text("x"), "b": text("y"), "c": text("z")}}, now=T0
        )
        self.repository.mutate("u1", "old", lambda doc: {"variables": {"a": text("x")}}, now=T0)

        summaries = self.repository.list_summaries("u1")

        self.assertEqual(
            [(s.tab_id, s.variable_count, s.rev) for s in summaries],
            [("new", 2, 1), ("old", 1, 3), ("empty", 0, 1)],
        )
        self.assertEqual(summaries[0].updated_at, T0 + timedelta(days=1))

    def test_delete_removes_only_that_row(self):
        self.repository.replace("u1", "t1", {"variables": {}}, now=T0)
        self.repository.replace("u1", "t2", {"variables": {}}, now=T0)

        self.assertTrue(self.repository.delete("u1", "t1"))
        self.assertFalse(self.repository.delete("u1", "t1"))

        self.assertEqual([s.tab_id for s in self.repository.list_summaries("u1")], ["t2"])

    def test_sweep_deletes_rows_not_updated_since_the_cutoff(self):
        self.repository.replace("u1", "stale", {"variables": {}}, now=T0)
        self.repository.replace("u2", "stale", {"variables": {}}, now=T0 + timedelta(days=1))
        self.repository.replace("u1", "fresh", {"variables": {}}, now=T0 + timedelta(days=40))

        removed = self.repository.delete_not_updated_since(T0 + timedelta(days=10))

        self.assertEqual(removed, 2)
        self.assertEqual([s.tab_id for s in self.repository.list_summaries("u1")], ["fresh"])
        self.assertEqual(self.repository.list_summaries("u2"), [])

    def test_sweep_keeps_a_row_exactly_at_the_cutoff(self):
        self.repository.replace("u1", "edge", {"variables": {}}, now=T0)

        self.assertEqual(self.repository.delete_not_updated_since(T0), 0)
        self.assertEqual(self.repository.delete_not_updated_since(T0 + timedelta(seconds=1)), 1)

    def test_deleting_the_user_removes_their_drafts(self):
        self.repository.replace("u1", "t1", {"variables": {}}, now=T0)

        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE id = 'u1'")

        self.assertIsNone(self.repository.get("u1", "t1"))


class TestManagerSweep(TabDraftStoreTestBase):

    def test_sweep_uses_a_thirty_day_window_from_the_injected_clock(self):
        clock = [T0]
        manager = TabDraftManager(self.repository, clock=lambda: clock[0])
        manager.replace("u1", "t1", {"variables": {"a": text("x")}})
        clock[0] = T0 + timedelta(days=29)
        manager.replace("u1", "t2", {"variables": {}})

        self.assertEqual(manager.sweep(now=T0 + timedelta(days=30)), 0)
        self.assertEqual(manager.sweep(now=T0 + timedelta(days=30, seconds=1)), 1)
        self.assertEqual([s.tab_id for s in manager.list("u1")], ["t2"])
        self.assertEqual(manager.sweep(now=T0 + timedelta(days=60)), 1)
        self.assertEqual(manager.list("u1"), [])


class TestDraftCap(TabDraftStoreTestBase):

    def test_creating_a_row_past_the_cap_raises_and_stores_nothing(self):
        for i in range(3):
            self.repository.replace("u1", f"t{i}", {"variables": {}}, now=T0, max_rows=3)

        with self.assertRaises(TabDraftLimitReached) as raised:
            self.repository.replace("u1", "t3", {"variables": {}}, now=T0, max_rows=3)

        self.assertEqual(raised.exception.limit, 3)
        self.assertIsNone(self.repository.get("u1", "t3"))
        self.assertEqual(len(self.repository.list_summaries("u1")), 3)

    def test_existing_rows_can_still_be_written_at_the_cap(self):
        for i in range(3):
            self.repository.replace("u1", f"t{i}", {"variables": {}}, now=T0, max_rows=3)

        draft = self.repository.replace("u1", "t1", {"variables": {"a": text("x")}}, now=T0, max_rows=3)

        self.assertEqual(draft.rev, 2)

    def test_the_cap_is_per_user(self):
        for i in range(3):
            self.repository.replace("u1", f"t{i}", {"variables": {}}, now=T0, max_rows=3)

        self.repository.replace("u2", "t0", {"variables": {}}, now=T0, max_rows=3)

    def test_deleting_frees_a_slot(self):
        for i in range(3):
            self.repository.replace("u1", f"t{i}", {"variables": {}}, now=T0, max_rows=3)
        self.repository.delete("u1", "t0")

        self.repository.replace("u1", "t9", {"variables": {}}, now=T0, max_rows=3)

    def test_variable_count_is_kept_on_every_write(self):
        self.repository.replace("u1", "t", {"variables": {"a": text("x"), "b": text("y")}}, now=T0)
        self.assertEqual(self.repository.list_summaries("u1")[0].variable_count, 2)
        self.repository.mutate("u1", "t", lambda doc: {"variables": {}}, now=T0)
        self.assertEqual(self.repository.list_summaries("u1")[0].variable_count, 0)

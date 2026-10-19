import threading
from unittest.mock import patch

from src.features.tab_drafts import validation
from src.features.tab_drafts.errors import (
    TabDraftInvalid,
    TabDraftLimitReached,
    TabDraftNotFound,
    TabDraftTooLarge,
)
from src.features.tab_drafts.manager import TabDraftManager
from tests.features.tab_drafts.test_repository import T0, TabDraftStoreTestBase, set_op, text


def setop(name, definition):
    return set_op(name, definition)


def removeop(name):
    return {"op": "remove", "name": name}


def choice(*options, **extra):
    return {"type": "choice", "options": list(options), "mode": "shuffle", **extra}


def number(**extra):
    return {"type": "number", "min": 0, "max": 10, "step": 1, "mode": "shuffle", **extra}


class ManagerTestBase(TabDraftStoreTestBase):

    def setUp(self):
        super().setUp()
        self.manager = TabDraftManager(self.repository, clock=lambda: T0)

    def failures(self, call):
        with self.assertRaises(TabDraftInvalid) as raised:
            call()
        return raised.exception.operations


class TestOperations(ManagerTestBase):

    def test_the_first_op_creates_the_draft(self):
        draft = self.manager.apply_operations("u1", "t1", [setop("hair", text("red"))])

        self.assertEqual(draft.rev, 1)
        self.assertEqual(draft.doc, {"variables": {"hair": text("red")}})
        self.assertEqual(self.manager.get("u1", "t1").doc, draft.doc)

    def test_each_batch_bumps_rev_once(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("1")), setop("b", text("2"))])
        draft = self.manager.apply_operations("u1", "t1", [setop("c", text("3"))])

        self.assertEqual(draft.rev, 2)
        self.assertEqual(set(draft.doc["variables"]), {"a", "b", "c"})

    def test_set_replaces_the_whole_definition(self):
        self.manager.apply_operations("u1", "t1", [setop("a", choice("x", "y"))])
        draft = self.manager.apply_operations("u1", "t1", [setop("a", text("now text"))])

        self.assertEqual(draft.doc["variables"]["a"], text("now text"))

    def test_remove_drops_the_variable_and_ignores_an_unknown_one(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("1")), setop("b", text("2"))])
        draft = self.manager.apply_operations("u1", "t1", [removeop("a"), removeop("ghost")])

        self.assertEqual(set(draft.doc["variables"]), {"b"})

    def test_last_write_wins_per_name_and_other_names_merge(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("base")), setop("b", text("base"))])
        self.manager.apply_operations("u1", "t1", [setop("a", text("from-device-1"))])
        draft = self.manager.apply_operations("u1", "t1", [setop("b", text("from-device-2"))])

        self.assertEqual(draft.doc["variables"]["a"], text("from-device-1"))
        self.assertEqual(draft.doc["variables"]["b"], text("from-device-2"))

    def test_within_a_batch_the_later_op_wins(self):
        draft = self.manager.apply_operations(
            "u1", "t1", [setop("a", text("1")), setop("a", text("2")), setop("b", text("x")), removeop("b")]
        )

        self.assertEqual(draft.doc["variables"], {"a": text("2")})

    def test_a_stale_base_rev_is_not_an_error(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])
        self.manager.apply_operations("u1", "t1", [setop("b", text("2"))])

        draft = self.manager.apply_operations("u1", "t1", [setop("c", text("3"))])

        self.assertEqual(set(draft.doc["variables"]), {"a", "b", "c"})

    def test_concurrent_writers_never_lose_each_others_variables(self):
        errors = []

        def write(index):
            try:
                self.manager.apply_operations("u1", "t1", [setop(f"v{index}", text(str(index)))])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        draft = self.manager.get("u1", "t1")
        self.assertEqual(errors, [])
        self.assertEqual(set(draft.doc["variables"]), {f"v{i}" for i in range(8)})
        self.assertEqual(draft.rev, 8)

    def test_a_bad_op_rejects_the_whole_batch_with_a_per_op_error(self):
        self.manager.apply_operations("u1", "t1", [setop("keep", text("1"))])

        failures = self.failures(
            lambda: self.manager.apply_operations(
                "u1",
                "t1",
                [setop("good", text("ok")), setop("bad name", text("x")), removeop("keep")],
            )
        )

        self.assertEqual([f["index"] for f in failures], [1])
        self.assertEqual(failures[0]["name"], "bad name")
        stored = self.manager.get("u1", "t1")
        self.assertEqual(stored.rev, 1)
        self.assertEqual(set(stored.doc["variables"]), {"keep"})

    def test_a_definition_error_is_reported_against_its_op_and_stores_nothing(self):
        failures = self.failures(
            lambda: self.manager.apply_operations(
                "u1", "t1", [setop("fine", text("ok")), setop("wide", number(max=100000))]
            )
        )

        self.assertEqual([(f["index"], f["name"], f["op"]) for f in failures], [(1, "wide", "set")])
        self.assertIn("limit", failures[0]["errors"][0])
        self.assertIsNone(self.manager.find("u1", "t1"))

    def test_malformed_ops_are_each_reported(self):
        failures = self.failures(
            lambda: self.manager.apply_operations(
                "u1", "t1", ["nope", {"op": "explode", "name": "a"}, {"op": "set", "name": "a", "type": "x"}, {"op": "remove"}]
            )
        )

        self.assertEqual([f["index"] for f in failures], [0, 1, 2, 3])

    def test_a_batch_of_exactly_max_operations_is_accepted(self):
        ops = [setop(f"v{i}", text(str(i))) for i in range(validation.MAX_OPERATIONS)]

        draft = self.manager.apply_operations("u1", "t1", ops)

        self.assertEqual(len(draft.doc["variables"]), validation.MAX_OPERATIONS)

    def test_the_flat_wire_shape_builds_each_variable_type(self):
        draft = self.manager.apply_operations(
            "u1",
            "t1",
            [
                {"op": "set", "name": "t", "type": "text", "value": "hi"},
                {"op": "set", "name": "bare"},
                {"op": "set", "name": "c", "type": "choice", "options": ["a", "b"], "mode": "pin", "pinned_index": 1},
                {"op": "set", "name": "s", "type": "choice", "options": ["a", "b"], "pinned_index": 1},
                {"op": "set", "name": "n", "type": "number", "min": 1, "max": 5, "value": 3, "reason": "why"},
            ],
        )

        variables = draft.doc["variables"]
        self.assertEqual(variables["t"], {"type": "text", "value": "hi"})
        self.assertEqual(variables["bare"], {"type": "text", "value": ""})
        self.assertEqual(variables["c"], {"type": "choice", "options": ["a", "b"], "mode": "pin", "pinnedIndex": 1})
        self.assertIsNone(variables["s"]["pinnedIndex"])
        self.assertEqual(
            variables["n"],
            {"type": "number", "mode": "shuffle", "min": 1, "max": 5, "step": 1, "value": 3},
        )

    def test_a_flat_choice_without_options_is_rejected_against_its_op(self):
        failures = self.failures(
            lambda: self.manager.apply_operations("u1", "t1", [{"op": "set", "name": "c", "type": "choice"}])
        )

        self.assertEqual([(f["index"], f["name"]) for f in failures], [(0, "c")])

    def test_an_unknown_flat_type_is_rejected_against_its_op(self):
        failures = self.failures(
            lambda: self.manager.apply_operations("u1", "t1", [{"op": "set", "name": "m", "type": "mystery"}])
        )

        self.assertEqual([(f["index"], f["name"]) for f in failures], [(0, "m")])

    def test_ops_must_be_a_non_empty_bounded_list(self):
        for ops in (None, [], "x", {"op": "set"}):
            with self.assertRaises(TabDraftInvalid):
                self.manager.apply_operations("u1", "t1", ops)
        with self.assertRaises(TabDraftInvalid):
            self.manager.apply_operations(
                "u1", "t1", [removeop("a")] * (validation.MAX_OPERATIONS + 1)
            )

    def test_removing_a_variable_a_condition_reads_is_rejected(self):
        self.manager.apply_operations(
            "u1",
            "t1",
            [
                setop("hair", choice("red", "blue")),
                setop("eyes", choice("green", {"text": "teal", "when": {"var": "hair", "values": ["red"]}})),
            ],
        )

        failures = self.failures(lambda: self.manager.apply_operations("u1", "t1", [removeop("hair")]))

        self.assertEqual(len(failures), 1)
        self.assertIn("hair", failures[0]["errors"][0])
        self.assertIn("hair", self.manager.get("u1", "t1").doc["variables"])

    def test_preexisting_problems_do_not_block_an_unrelated_op(self):
        self.repository.replace(
            "u1", "t1", {"variables": {"legacy": number(max=100000)}}, now=T0
        )

        draft = self.manager.apply_operations("u1", "t1", [setop("fresh", text("x"))])

        self.assertEqual(set(draft.doc["variables"]), {"legacy", "fresh"})

    def test_variables_helper_reads_the_current_map(self):
        self.assertEqual(self.manager.variables("u1", "t1"), {})
        self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])

        self.assertEqual(self.manager.variables("u1", "t1"), {"a": text("1")})


class TestValidationLimits(ManagerTestBase):

    def test_number_grid_limits_are_enforced(self):
        self.assertTrue(self.failures(lambda: self.manager.apply_operations("u1", "t1", [setop("n", number(max=1000))])))
        self.assertTrue(self.failures(lambda: self.manager.apply_operations("u1", "t1", [setop("n", number(step=0))])))
        self.assertTrue(self.failures(lambda: self.manager.apply_operations("u1", "t1", [setop("n", number(min=5, max=1))])))
        self.manager.apply_operations("u1", "t1", [setop("n", number(max=199))])

    def test_choice_options_are_capped_at_two_hundred(self):
        self.manager.apply_operations(
            "u1", "t1", [setop("ok", choice(*[f"o{i}" for i in range(validation.MAX_OPTIONS)]))]
        )

        failures = self.failures(
            lambda: self.manager.apply_operations(
                "u1", "t1", [setop("many", choice(*[f"o{i}" for i in range(validation.MAX_OPTIONS + 1)]))]
            )
        )

        self.assertIn(str(validation.MAX_OPTIONS), failures[0]["errors"][0])

    def test_option_text_is_capped_at_three_hundred_characters(self):
        self.manager.apply_operations(
            "u1", "t1", [setop("ok", choice("x" * validation.MAX_OPTION_CHARS, "y"))]
        )

        self.assertTrue(
            self.failures(
                lambda: self.manager.apply_operations(
                    "u1", "t1", [setop("long", choice("x" * (validation.MAX_OPTION_CHARS + 1), "y"))]
                )
            )
        )
        self.assertTrue(
            self.failures(
                lambda: self.manager.apply_operations(
                    "u1",
                    "t1",
                    [setop("long2", choice({"text": "x" * (validation.MAX_OPTION_CHARS + 1)}, "y"))],
                )
            )
        )

    def test_text_value_is_capped_at_two_thousand_characters(self):
        self.manager.apply_operations("u1", "t1", [setop("ok", text("x" * validation.MAX_TEXT_CHARS))])

        failures = self.failures(
            lambda: self.manager.apply_operations(
                "u1", "t1", [setop("long", text("x" * (validation.MAX_TEXT_CHARS + 1)))]
            )
        )

        self.assertEqual(failures[0]["name"], "long")

    def test_document_size_is_bounded_for_ops(self):
        per_variable = choice(*["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS)
        names = [f"v{i}" for i in range(6)]
        ops = [setop(name, per_variable) for name in names]

        with self.assertRaises(TabDraftTooLarge) as raised:
            self.manager.apply_operations("u1", "t1", ops)

        self.assertEqual(raised.exception.limit, validation.MAX_DOC_BYTES)
        self.assertGreater(raised.exception.size, validation.MAX_DOC_BYTES)
        self.assertIsNone(self.manager.find("u1", "t1"))

    def test_document_size_is_bounded_for_replace(self):
        per_variable = choice(*["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS)
        doc = {"variables": {f"v{i}": per_variable for i in range(6)}}

        with self.assertRaises(TabDraftTooLarge):
            self.manager.replace("u1", "t1", doc)

    def test_a_document_just_under_the_bound_is_stored(self):
        per_variable = choice(*["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS)
        self.manager.replace("u1", "t1", {"variables": {f"v{i}": per_variable for i in range(3)}})

        self.assertEqual(self.manager.get("u1", "t1").rev, 1)

    def test_a_growing_draft_cannot_cross_the_bound_one_batch_at_a_time(self):
        per_variable = choice(*["w" * validation.MAX_OPTION_CHARS] * validation.MAX_OPTIONS)
        stored = 0
        with self.assertRaises(TabDraftTooLarge):
            for i in range(10):
                self.manager.apply_operations("u1", "t1", [setop(f"v{i}", per_variable)])
                stored = i + 1

        self.assertEqual(len(self.manager.get("u1", "t1").doc["variables"]), stored)
        self.assertLessEqual(validation.doc_size(self.manager.get("u1", "t1").doc), validation.MAX_DOC_BYTES)


class TestDraftCap(ManagerTestBase):

    def test_the_cap_is_two_hundred_drafts_per_user(self):
        self.assertEqual(validation.MAX_DRAFTS_PER_USER, 200)

    def test_a_new_draft_past_the_cap_is_refused_for_ops_and_replace(self):
        with patch.object(validation, "MAX_DRAFTS_PER_USER", 2):
            self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])
            self.manager.replace("u1", "t2", {"variables": {}})

            with self.assertRaises(TabDraftLimitReached) as raised:
                self.manager.apply_operations("u1", "t3", [setop("a", text("1"))])
            with self.assertRaises(TabDraftLimitReached):
                self.manager.replace("u1", "t3", {"variables": {}})

            self.assertEqual(raised.exception.limit, 2)
            self.assertIsNone(self.manager.find("u1", "t3"))

    def test_existing_drafts_and_other_users_are_unaffected_at_the_cap(self):
        with patch.object(validation, "MAX_DRAFTS_PER_USER", 2):
            self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])
            self.manager.apply_operations("u1", "t2", [setop("a", text("1"))])

            self.manager.apply_operations("u1", "t2", [setop("b", text("2"))])
            self.manager.apply_operations("u2", "t1", [setop("a", text("1"))])

            self.assertEqual(len(self.manager.list("u1")), 2)

    def test_a_deleted_draft_frees_its_slot(self):
        with patch.object(validation, "MAX_DRAFTS_PER_USER", 1):
            self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])
            self.manager.delete("u1", "t1")

            self.manager.apply_operations("u1", "t2", [setop("a", text("1"))])


class TestReplace(ManagerTestBase):

    def test_replace_creates_then_overwrites_and_bumps_rev(self):
        first = self.manager.replace("u1", "t1", {"variables": {"a": text("1")}})
        second = self.manager.replace("u1", "t1", {"variables": {"b": text("2")}})

        self.assertEqual((first.rev, second.rev), (1, 2))
        self.assertEqual(set(second.doc["variables"]), {"b"})

    def test_a_missing_variables_key_is_an_empty_map(self):
        self.assertEqual(self.manager.replace("u1", "t1", {}).doc, {"variables": {}})

    def test_names_are_trimmed_and_collisions_rejected(self):
        draft = self.manager.replace("u1", "t1", {"variables": {" a ": text("1")}})
        self.assertEqual(set(draft.doc["variables"]), {"a"})
        with self.assertRaises(TabDraftInvalid):
            self.manager.replace("u1", "t1", {"variables": {"a": text("1"), " a": text("2")}})

    def test_invalid_documents_are_rejected_and_store_nothing(self):
        for doc in (
            [],
            {"variables": []},
            {"variables": {"bad name": text("x")}},
            {"variables": {"a": {"type": "mystery"}}},
            {"variables": {"a": text("x")}, "layout": {}},
        ):
            with self.assertRaises(TabDraftInvalid):
                self.manager.replace("u1", "t1", doc)
        self.assertIsNone(self.manager.find("u1", "t1"))


class TestScoping(ManagerTestBase):

    def test_get_and_delete_of_another_users_tab_are_not_found(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])

        with self.assertRaises(TabDraftNotFound):
            self.manager.get("u2", "t1")
        with self.assertRaises(TabDraftNotFound):
            self.manager.delete("u2", "t1")
        self.assertEqual(self.manager.get("u1", "t1").rev, 1)

    def test_ops_and_replace_by_another_user_never_touch_the_owners_row(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("owner"))])

        self.manager.apply_operations("u2", "t1", [setop("a", text("intruder"))])
        self.manager.replace("u2", "t1", {"variables": {}})

        owner = self.manager.get("u1", "t1")
        self.assertEqual(owner.rev, 1)
        self.assertEqual(owner.doc["variables"]["a"], text("owner"))
        self.assertEqual([s.tab_id for s in self.manager.list("u2")], ["t1"])

    def test_a_new_draft_under_a_taken_tab_id_starts_empty_at_rev_one(self):
        self.manager.apply_operations("u1", "t1", [setop("secret", text("owner"))])
        self.manager.apply_operations("u1", "t1", [setop("more", text("owner"))])

        draft = self.manager.apply_operations("u2", "t1", [setop("mine", text("u2"))])

        self.assertEqual(draft.rev, 1)
        self.assertEqual(draft.doc, {"variables": {"mine": text("u2")}})
        self.assertEqual(self.manager.get("u2", "t1").doc, draft.doc)

    def test_list_only_shows_the_users_own_drafts(self):
        self.manager.apply_operations("u1", "mine", [setop("a", text("1"))])
        self.manager.apply_operations("u2", "theirs", [setop("a", text("1"))])

        self.assertEqual([s.tab_id for s in self.manager.list("u1")], ["mine"])

    def test_delete_then_get_is_not_found(self):
        self.manager.apply_operations("u1", "t1", [setop("a", text("1"))])
        self.manager.delete("u1", "t1")

        with self.assertRaises(TabDraftNotFound):
            self.manager.get("u1", "t1")

    def test_tab_ids_are_validated(self):
        for tab_id in ("", "a/b", "x" * 65, "has space", "../etc"):
            with self.assertRaises(TabDraftInvalid):
                self.manager.get("u1", tab_id)
            with self.assertRaises(TabDraftInvalid):
                self.manager.apply_operations("u1", tab_id, [removeop("a")])
        self.manager.apply_operations("u1", "3f2b8c1e-aaaa-4bbb-8ccc-0123456789ab", [removeop("a")])

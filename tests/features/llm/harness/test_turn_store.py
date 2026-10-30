import sqlite3
import unittest
from dataclasses import replace
from typing import Dict, List, Optional

from src.features.llm.harness import codec
from src.features.llm.harness.errors import HarnessError, RateLimited
from src.features.llm.harness.events import FinishReason, Usage
from src.features.llm.harness.messages import (
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.recovery import INTERRUPTED_REASON, interrupted_tool_step
from src.features.llm.harness.store import SqliteTurnStore
from src.features.llm.harness.turn_repository import TurnStoreError
from src.features.llm.harness.turn_state import (
    DecisionKind,
    Notice,
    NoticeKind,
    PendingCall,
    StepRecord,
    StepStatus,
    TurnState,
    TurnStatus,
)
from tests.fixtures.llm_turns import ask_step, final_step, result_step, state, user_step
from tests.fixtures.persistence_base import PersistenceTestBase

ANTHROPIC = Origin(adapter="anthropic-messages", provider="anthropic", model="claude-sonnet-5-5", credential="0f3a9c1d2e4b5a6c")
RAW_BLOCKS = [
    {"type": "thinking", "thinking": "", "signature": "EqQBCkYIBxgCKkA="},
    {"type": "text", "text": "Checking.", "citations": None},
    {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {"q": "ż", "deep": {"x": [1, 2.5, None, True]}}},
]


class ReferenceTurnStore:
    def __init__(self) -> None:
        self._steps: List[StepRecord] = []
        self._states: Dict[str, TurnState] = {}
        self._order: List[str] = []

    def steps(self, session_id: str) -> List[StepRecord]:
        return sorted((step for step in self._steps if step.session_id == session_id), key=lambda step: step.seq)

    def turn_steps(self, turn_id: str) -> List[StepRecord]:
        return sorted((step for step in self._steps if step.turn_id == turn_id), key=lambda step: step.seq)

    def next_seq(self, session_id: str) -> int:
        return max((step.seq for step in self._steps if step.session_id == session_id), default=-1) + 1

    def append(self, step: StepRecord) -> None:
        state = self._states.get(step.turn_id)
        if state is None or state.session_id != step.session_id:
            raise TurnStoreError("unknown turn")
        if step.seq != self.next_seq(step.session_id):
            raise TurnStoreError("out of order")
        if any(existing.id == step.id for existing in self._steps):
            raise TurnStoreError("duplicate id")
        self._steps.append(step)

    def save_state(self, state: TurnState) -> None:
        existing = self._states.get(state.turn_id)
        if existing is not None and existing.session_id != state.session_id:
            raise TurnStoreError("another session")
        if existing is None:
            self._order.append(state.turn_id)
        self._states[state.turn_id] = state

    def load_state(self, turn_id: str) -> Optional[TurnState]:
        return self._states.get(turn_id)

    def open_turn(self, session_id: str) -> Optional[TurnState]:
        for turn_id in reversed(self._order):
            state = self._states[turn_id]
            if state.session_id == session_id and state.status in (TurnStatus.RUNNING, TurnStatus.AWAITING_DECISION):
                return state
        return None

    def latest_turn(self, session_id: str) -> Optional[TurnState]:
        for turn_id in reversed(self._order):
            if self._states[turn_id].session_id == session_id:
                return self._states[turn_id]
        return None

    def fail_interrupted(self) -> int:
        running = [item for item in self._states.values() if item.status is TurnStatus.RUNNING]
        for item in running:
            tool_step = interrupted_tool_step(
                self.turn_steps(item.turn_id),
                item,
                step_id=f"swept-{item.turn_id}",
                seq=self.next_seq(item.session_id),
                status=StepStatus.FAILED,
                reason=INTERRUPTED_REASON,
            )
            if tool_step is not None:
                self.append(tool_step)
            self.save_state(replace(item, status=TurnStatus.FAILED, error=HarnessError("x", code="interrupted").to_json()))
        return len(running)


class TurnStoreContract:
    def new_store(self):
        raise NotImplementedError

    def add_session(self, session_id: str) -> None:
        pass

    def reopen(self):
        return self.store

    def setUp(self):
        super().setUp()
        self.store = self.new_store()
        self.add_session("s1")

    def test_an_empty_session_has_no_steps_and_starts_at_one(self):
        self.assertEqual(self.store.steps("s1"), [])
        self.assertEqual(self.store.next_seq("s1"), 0)
        self.assertIsNone(self.store.open_turn("s1"))
        self.assertIsNone(self.store.load_state("missing"))

    def test_a_saved_state_loads_back_equal(self):
        saved = state(
            status=TurnStatus.AWAITING_DECISION,
            rounds_used=2,
            pending=(PendingCall(call_id="c2", name="delete_prompt", decision=DecisionKind.APPROVAL, preview={"before": ["a"]}),),
            gathered=(ToolResultPart(call_id="c1", name="search", content=(TextPart(text="2 hits"),)),),
            failed_calls={"search:{}": "no results"},
            tool_failures={"search": 1},
            applied_calls=frozenset({"a", "b"}),
            loaded_tool_groups=("gallery",),
            notices=(Notice(kind=NoticeKind.TRUNCATED, message="cut", step_seq=3, detail={"n": 1}),),
            error=RateLimited("slow", status=429, retry_after=2.5).to_json(),
        )
        self.store.save_state(saved)

        self.assertEqual(self.reopen().load_state("t1"), saved)

    def test_saving_a_state_again_replaces_it(self):
        self.store.save_state(state(rounds_used=0))
        self.store.save_state(state(rounds_used=3, status=TurnStatus.COMPLETED))

        loaded = self.store.load_state("t1")
        self.assertEqual((loaded.rounds_used, loaded.status), (3, TurnStatus.COMPLETED))

    def test_steps_come_back_in_order_with_every_field(self):
        self.store.save_state(state())
        first = user_step(0)
        second = ask_step(
            1, [("c1", "search", {"q": "cats"}), ("c2", "search", {"q": "dogs"})], text="Looking.",
            usage=Usage(input_tokens=10, output_tokens=5, cached_input_tokens=8, cost=0.0004), finish_raw="tool_use",
        )
        third = result_step(2, [("c1", "search", "2 hits", False), ("c2", "search", "boom", True)])
        fourth = final_step(3, "Done.", usage=Usage(input_tokens=30))
        for step in (first, second, third, fourth):
            self.store.append(step)

        stored = self.reopen().steps("s1")
        self.assertEqual(stored, [first, second, third, fourth])
        self.assertEqual(self.store.next_seq("s1"), 4)
        self.assertEqual(self.store.turn_steps("t1"), stored)

    def test_opaque_provider_data_survives_byte_for_byte(self):
        self.store.save_state(state())
        original = StepRecord(
            id="a0", session_id="s1", turn_id="t1", seq=0,
            message=message(
                "m1", "assistant",
                ReasoningPart(opaque={"signature": "EqQBCkYIBxgCKkA="}),
                ReasoningPart(redacted=True, opaque={"data": "EmwKAhgB"}),
                ReasoningPart(text="timed", duration_ms=1840),
                TextPart(text="Checking.", phase="commentary"),
                ToolCallPart(call_id="c1", name="search", args=RAW_BLOCKS[2]["input"], args_raw='{"q": "ż"}', wire_id="toolu_01", opaque={"thought_signature": "abc"}),
                origin=ANTHROPIC, raw=RAW_BLOCKS,
            ),
            status=StepStatus.INCOMPLETE, usage=Usage(input_tokens=1, raw={"input_tokens": 1}),
            finish=FinishReason.LENGTH, finish_raw="max_tokens", created_at="2026-10-26T10:00:01+00:00",
        )
        self.store.append(original)

        (stored,) = self.reopen().steps("s1")
        self.assertEqual(stored, original)
        self.assertEqual(codec.dumps(stored.message), codec.dumps(original.message))
        self.assertTrue(stored.message.replays_raw_to(ANTHROPIC))

    def test_a_step_needs_a_known_turn_of_its_own_session(self):
        self.add_session("s2")
        self.store.save_state(state())

        with self.assertRaises(TurnStoreError):
            self.store.append(user_step(0, turn_id="nope"))
        with self.assertRaises(TurnStoreError):
            self.store.append(user_step(0, session_id="s2"))
        self.assertEqual(self.store.steps("s1"), [])

    def test_steps_are_append_only_and_contiguous(self):
        self.store.save_state(state())
        self.store.append(user_step(0))

        with self.assertRaises(TurnStoreError):
            self.store.append(user_step(0, text="rewritten"))
        with self.assertRaises(TurnStoreError):
            self.store.append(final_step(2, "skipped a seq"))
        self.assertEqual(self.store.steps("s1"), [user_step(0)])

    def test_a_step_id_is_unique(self):
        self.store.save_state(state())
        self.store.append(user_step(0))

        with self.assertRaises(TurnStoreError):
            self.store.append(replace(user_step(1), id="s1.u0"))

    def test_sessions_keep_separate_sequences(self):
        self.add_session("s2")
        self.store.save_state(state())
        self.store.save_state(state(turn_id="t2", session_id="s2"))
        self.store.append(user_step(0))
        self.store.append(user_step(0, turn_id="t2", session_id="s2", text="other"))
        self.store.append(final_step(1, "answer"))

        self.assertEqual([step.seq for step in self.store.steps("s1")], [0, 1])
        self.assertEqual([step.seq for step in self.store.steps("s2")], [0])
        self.assertEqual(self.store.next_seq("s2"), 1)

    def test_a_turn_cannot_move_to_another_session(self):
        self.add_session("s2")
        self.store.save_state(state())

        with self.assertRaises(TurnStoreError):
            self.store.save_state(state(session_id="s2"))
        self.assertEqual(self.store.load_state("t1").session_id, "s1")

    def test_only_a_running_or_waiting_turn_is_open(self):
        self.store.save_state(state(turn_id="t1", status=TurnStatus.COMPLETED))
        self.assertIsNone(self.store.open_turn("s1"))

        self.store.save_state(state(turn_id="t2", status=TurnStatus.AWAITING_DECISION))
        self.assertEqual(self.store.open_turn("s1").turn_id, "t2")

        self.store.save_state(state(turn_id="t2", status=TurnStatus.CANCELLED))
        self.store.save_state(state(turn_id="t3", status=TurnStatus.RUNNING))
        self.store.save_state(state(turn_id="t4", status=TurnStatus.FAILED))
        self.assertEqual(self.store.open_turn("s1").turn_id, "t3")

    def test_the_latest_turn_is_the_newest_of_its_session_whatever_its_status(self):
        self.assertIsNone(self.store.latest_turn("s1"))
        self.store.save_state(state(turn_id="t1", status=TurnStatus.AWAITING_DECISION))
        self.store.save_state(state(turn_id="t2", status=TurnStatus.COMPLETED))
        self.store.save_state(state(turn_id="t1", status=TurnStatus.CANCELLED))

        self.assertEqual(self.store.latest_turn("s1").turn_id, "t2")
        self.assertIsNone(self.store.latest_turn("s2"))

    def test_the_newest_open_turn_wins(self):
        self.store.save_state(state(turn_id="t1", status=TurnStatus.RUNNING))
        self.store.save_state(state(turn_id="t2", status=TurnStatus.AWAITING_DECISION))

        self.assertEqual(self.store.open_turn("s1").turn_id, "t2")

    def test_cancelling_mid_stream_keeps_the_partial_step_and_closes_the_tool_batch(self):
        self.store.save_state(state())
        self.store.append(user_step(0))
        self.store.append(ask_step(1, [("c1", "search", {"q": "cats"})], text="Looking"))
        running = state(gathered=(ToolResultPart(call_id="c1", name="search", content=(TextPart(text="2 hits"),)),))
        self.store.save_state(running)

        steps = self.store.steps("s1")
        closing = interrupted_tool_step(
            steps, running, step_id="r3", seq=self.store.next_seq("s1"), status=StepStatus.CANCELLED, reason="Cancelled."
        )
        self.store.append(closing)
        partial = final_step(3, "Found 2 cats, but", status=StepStatus.CANCELLED, finish=None)
        self.store.append(partial)
        self.store.save_state(replace(running, status=TurnStatus.CANCELLED))

        stored = self.reopen().steps("s1")
        self.assertEqual([step.status for step in stored], [StepStatus.COMPLETED, StepStatus.COMPLETED, StepStatus.CANCELLED, StepStatus.CANCELLED])
        self.assertEqual(stored[2].message.tool_results()[0].text(), "2 hits")
        self.assertEqual(stored[3].message.text(), "Found 2 cats, but")
        self.assertIsNone(self.store.open_turn("s1"))

    def test_a_provider_error_keeps_the_partial_step_and_the_error(self):
        self.store.save_state(state())
        self.store.append(user_step(0))
        self.store.append(final_step(1, "The answer is", status=StepStatus.FAILED, finish=None))
        failed = state(status=TurnStatus.FAILED, error=RateLimited("slow down", status=429, retry_after=4.0).to_json())
        self.store.save_state(failed)

        loaded = self.reopen().load_state("t1")
        self.assertEqual(HarnessError.from_json(loaded.error), RateLimited("slow down", status=429, retry_after=4.0))
        self.assertEqual(self.reopen().steps("s1")[1].message.text(), "The answer is")
        self.assertEqual(self.reopen().steps("s1")[1].status, StepStatus.FAILED)

    def test_a_turn_waiting_for_a_decision_resumes_from_the_stored_state(self):
        self.store.save_state(state())
        self.store.append(user_step(0))
        self.store.append(ask_step(1, [("c1", "search", {}), ("c2", "delete_prompt", {"id": 7})], text="Deleting."))
        waiting = state(
            status=TurnStatus.AWAITING_DECISION,
            rounds_used=1,
            pending=(PendingCall(call_id="c2", name="delete_prompt", decision=DecisionKind.APPROVAL, preview={"title": "old"}),),
            gathered=(ToolResultPart(call_id="c1", name="search", content=(TextPart(text="found"),)),),
            applied_calls=frozenset({"search:{}"}),
        )
        self.store.save_state(waiting)

        resumed = self.reopen().open_turn("s1")
        self.assertEqual(resumed, waiting)
        done = replace(
            resumed,
            status=TurnStatus.RUNNING,
            pending=(),
            gathered=resumed.gathered + (ToolResultPart(call_id="c2", name="delete_prompt", content=(TextPart(text="deleted"),)),),
        )
        self.store.save_state(done)
        self.store.append(result_step(2, [(item.call_id, item.name, item.text(), False) for item in done.gathered]))
        self.store.append(final_step(3, "Deleted it."))
        self.store.save_state(replace(done, status=TurnStatus.COMPLETED))

        self.assertEqual([step.message.role for step in self.store.steps("s1")], [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT])
        self.assertIsNone(self.store.open_turn("s1"))

    def test_the_restart_sweep_fails_running_turns_and_closes_their_tool_batch(self):
        self.add_session("s2")
        self.store.save_state(state())
        self.store.append(user_step(0))
        self.store.append(ask_step(1, [("c1", "search", {}), ("c2", "rename", {})]))
        self.store.save_state(state(gathered=(ToolResultPart(call_id="c1", name="search", content=(TextPart(text="found"),)),)))
        self.store.save_state(state(turn_id="t2", session_id="s2"))
        self.store.append(user_step(0, turn_id="t2", session_id="s2"))
        self.store.save_state(state(turn_id="t3", status=TurnStatus.AWAITING_DECISION))

        swept = self.reopen().fail_interrupted()

        self.assertEqual(swept, 2)
        reopened = self.reopen()
        first = reopened.load_state("t1")
        self.assertEqual(first.status, TurnStatus.FAILED)
        self.assertEqual(HarnessError.from_json(first.error).code, "interrupted")
        closing = reopened.steps("s1")[-1]
        self.assertEqual(closing.message.role, Role.TOOL)
        self.assertEqual(closing.status, StepStatus.FAILED)
        self.assertEqual(
            [(item.call_id, item.text(), item.is_error) for item in closing.message.tool_results()],
            [("c1", "found", False), ("c2", INTERRUPTED_REASON, True)],
        )
        self.assertEqual(reopened.load_state("t2").status, TurnStatus.FAILED)
        self.assertEqual(len(reopened.steps("s2")), 1)
        self.assertEqual(reopened.load_state("t3").status, TurnStatus.AWAITING_DECISION)

    def test_a_second_sweep_finds_nothing(self):
        self.store.save_state(state())
        self.store.append(user_step(0))
        self.store.append(ask_step(1, [("c1", "search", {})]))

        self.assertEqual(self.store.fail_interrupted(), 1)
        steps_after_first = self.store.steps("s1")

        self.assertEqual(self.store.fail_interrupted(), 0)
        self.assertEqual(self.store.steps("s1"), steps_after_first)

    def test_stored_history_re_renders_to_the_same_bytes(self):
        self.store.save_state(state())
        history = [
            user_step(0),
            ask_step(1, [("c1", "search", {"q": "ż", "n": [1, 2.5]})], text="Looking.", usage=Usage(input_tokens=3)),
            result_step(2, [("c1", "search", "2 hits", False)]),
            final_step(3, "Done."),
        ]
        for step in history:
            self.store.append(step)

        def render(steps):
            return codec.dumps([step.message for step in steps])

        self.assertEqual(render(self.reopen().steps("s1")), render(history))


class TestReferenceTurnStore(TurnStoreContract, unittest.TestCase):
    def new_store(self):
        return ReferenceTurnStore()


class TestSqliteTurnStore(TurnStoreContract, PersistenceTestBase):
    in_memory = True

    def new_store(self):
        return SqliteTurnStore()

    def add_session(self, session_id):
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO chat_sessions (id, user_id, mode, name, status) VALUES (?, 'u1', 'assistant', 'chat', 'active')",
                (session_id,),
            )

    def reopen(self):
        return SqliteTurnStore()

    def test_a_turn_needs_an_existing_session(self):
        with self.assertRaises(TurnStoreError):
            self.store.save_state(state(session_id="ghost"))

    def test_the_database_refuses_to_rewrite_a_step(self):
        self.store.save_state(state())
        self.store.append(user_step(0))

        with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
            with self.db.get_cursor() as cursor:
                cursor.execute("UPDATE chat_steps SET parts = '[]'")
        self.assertEqual(self.store.steps("s1"), [user_step(0)])

    def test_deleting_the_session_removes_its_turns_and_steps(self):
        self.store.save_state(state())
        self.store.append(user_step(0))

        with self.db.get_cursor() as cursor:
            cursor.execute("DELETE FROM chat_sessions WHERE id = 's1'")

        self.assertEqual(self.store.steps("s1"), [])
        self.assertIsNone(self.store.load_state("t1"))

    def test_steps_and_states_load_in_one_batch_by_turn(self):
        self.add_session("s2")
        self.store.save_state(state())
        self.store.save_state(state(turn_id="t2", session_id="s2", status=TurnStatus.COMPLETED))
        self.store.append(user_step(0))
        self.store.append(final_step(1, "answer"))
        self.store.append(user_step(0, turn_id="t2", session_id="s2"))

        grouped = self.store.steps_by_turn(["t1", "t2", "t1", "missing"])
        loaded = self.store.states(["t1", "t2", "missing"])

        self.assertEqual({turn: [step.seq for step in steps] for turn, steps in grouped.items()}, {"t1": [0, 1], "t2": [0]})
        self.assertEqual({turn: item.status for turn, item in loaded.items()}, {"t1": TurnStatus.RUNNING, "t2": TurnStatus.COMPLETED})
        self.assertEqual(self.store.steps_by_turn([]), {})

    def test_the_sweep_marks_a_turn_with_unreadable_state_failed(self):
        self.store.save_state(state())
        with self.db.get_cursor() as cursor:
            cursor.execute("UPDATE chat_turns SET state_json = '{broken'")

        self.assertEqual(self.store.fail_interrupted(), 1)

        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT status FROM chat_turns WHERE id = 't1'")
            self.assertEqual(cursor.fetchone()["status"], "failed")
        self.assertEqual(self.store.fail_interrupted(), 0)

    def test_a_failed_append_rolls_back_cleanly(self):
        self.store.save_state(state())
        self.store.append(user_step(0))

        with self.assertRaises(TurnStoreError):
            self.store.append(replace(user_step(1), id="s1.u0"))

        self.store.append(final_step(1, "still appendable"))
        self.assertEqual([step.seq for step in self.store.steps("s1")], [0, 1])


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace

from src.features.chat import decisions, projection
from src.features.chat.repository import ChatRepository
from src.features.llm.harness.events import Usage
from src.features.llm.harness.messages import ReasoningPart, TextPart, ToolCallPart, ToolResultPart, message
from src.features.llm.harness.store import SqliteTurnStore
from src.features.llm.harness.turn_state import (
    DecisionKind,
    Notice,
    NoticeKind,
    PendingCall,
    StepRecord,
    StepStatus,
    TurnStatus,
)
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN
from tests.fixtures.llm_turns import result_step, state, user_step
from tests.fixtures.persistence_base import PersistenceTestBase


class TestStepsBecomeChatRows(PersistenceTestBase):
    in_memory = True

    def setUp(self):
        super().setUp()
        self.store = SqliteTurnStore()
        self.chat = ChatRepository()
        with self.db.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO chat_sessions (id, user_id, mode, name, status) VALUES ('s1', 'u1', 'assistant', 'chat', 'active')"
            )

    def assistant_with_reasoning(self, seq, text, reasoning, usage):
        return StepRecord(
            id=f"s1.a{seq}", session_id="s1", turn_id="t1", seq=seq,
            message=message(f"ma{seq}", "assistant", ReasoningPart(text=reasoning), TextPart(text=text), origin=SCRIPTED_ORIGIN),
            status=StepStatus.COMPLETED, usage=usage,
        )

    def test_a_paused_and_resumed_turn_projects_into_two_rows_that_read_back_with_reasoning(self):
        self.store.save_state(state())
        self.store.append(user_step(0, text="rename it"))
        user_row = self.chat.add_message("s1", "user", "rename it", turn_id="t1")
        asks = StepRecord(
            id="s1.a1", session_id="s1", turn_id="t1", seq=1,
            message=message(
                "ma2", "assistant", ReasoningPart(text="plan"), TextPart(text="Renaming. "),
                ToolCallPart(call_id="c1", name="rename_prompt", args={"name": "b"}), origin=SCRIPTED_ORIGIN,
            ),
            status=StepStatus.COMPLETED, usage=Usage(input_tokens=10, output_tokens=2),
        )
        self.store.append(asks)
        waiting = state(
            status=TurnStatus.AWAITING_DECISION,
            rounds_used=1,
            pending=(PendingCall(call_id="c1", name="rename_prompt", decision=DecisionKind.APPROVAL, preview={"after": "b"}),),
        )
        self.store.save_state(waiting)
        steps = self.store.turn_steps("t1")
        first_segment = steps[1:]
        first_row = self.chat.add_message(
            "s1", "assistant", projection.segment_text(first_segment),
            metadata=projection.assistant_metadata(first_segment, waiting, context_metadata={"form_state": {}}),
            turn_id="t1",
        )

        resumed = replace(waiting, status=TurnStatus.RUNNING, pending=(), gathered=(
            ToolResultPart(call_id="c1", name="rename_prompt", content=(TextPart(text="Renamed."),)),
        ))
        self.store.save_state(resumed)
        self.store.append(result_step(2, [("c1", "rename_prompt", "Renamed.", False)]))
        self.store.append(self.assistant_with_reasoning(3, "Done.", "wrap up", Usage(input_tokens=30, output_tokens=4)))
        finished = replace(resumed, status=TurnStatus.COMPLETED, rounds_used=2, notices=(
            Notice(kind=NoticeKind.REASONING_DROPPED, message="dropped", step_seq=3),
        ))
        self.store.save_state(finished)
        second_segment = self.store.turn_steps("t1")[3:]
        second_row = self.chat.add_message(
            "s1", "assistant", projection.segment_text(second_segment),
            metadata=projection.assistant_metadata(second_segment, finished, extra={"continued_from": {"message_id": first_row.id, "tool_index": 0}}),
            turn_id="t1",
        )

        stored = self.chat.get_messages("s1")
        by_turn = self.store.steps_by_turn(sorted({item.turn_id for item in stored}))
        states = self.store.states(list(by_turn))
        shown = projection.present(stored, by_turn, states, show_thinking=True)
        hidden = projection.present(stored, by_turn, states, show_thinking=False)

        self.assertEqual([item.id for item in shown], [user_row.id, first_row.id, second_row.id])
        self.assertEqual(shown[0].reasoning, None)
        self.assertEqual([block.text for block in shown[1].reasoning], ["plan"])
        self.assertEqual([block.text for block in shown[2].reasoning], ["wrap up"])
        self.assertTrue(all(item.reasoning is None for item in hidden))
        self.assertEqual((shown[1].usage.input_tokens, shown[2].usage.input_tokens), (10, 30))
        self.assertIsNone(shown[1].notices)
        self.assertEqual([item.kind for item in shown[2].notices], ["reasoning_dropped"])
        self.assertEqual(shown[1].content, "Renaming. ")
        self.assertEqual(shown[1].tool_executions[0]["approval_status"], "pending")
        self.assertEqual(shown[1].tool_executions[0]["preview"], {"after": "b"})
        self.assertIsNone(shown[2].tool_executions)
        self.assertEqual({item.turn_id for item in shown}, {"t1"})

        call_ids = decisions.execution_call_ids(first_row.metadata)
        self.assertEqual(call_ids, ["c1"])
        self.assertEqual(decisions.decision_for(waiting, call_ids, 0, True).call_id, "c1")

    def test_the_turn_id_of_a_row_survives_storage(self):
        self.store.save_state(state())

        row = self.chat.add_message("s1", "user", "hi", turn_id="t1")
        plain = self.chat.add_message("s1", "user", "no turn")

        self.assertEqual(self.chat.get_message(row.id).turn_id, "t1")
        self.assertIsNone(self.chat.get_message(plain.id).turn_id)
        with self.db.get_cursor() as cursor:
            cursor.execute("SELECT turn_id FROM chat_messages WHERE id = ?", (row.id,))
            self.assertEqual(cursor.fetchone()["turn_id"], "t1")

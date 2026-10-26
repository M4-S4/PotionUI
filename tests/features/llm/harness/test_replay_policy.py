from typing import Any, List, Sequence

import pytest

from src.features.llm.harness.events import Finish, FinishReason, OpaqueDelta, StreamEvent
from src.features.llm.harness.messages import (
    LEGACY_ORIGIN,
    CompactionPart,
    Message,
    Origin,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    message,
)
from src.features.llm.harness.profile import ReplayPolicy
from src.features.llm.harness.replay import PLACEHOLDER_SIGNATURE, missing_reasoning, policy_replay
from src.features.llm.harness.turn_state import NoticeKind, StepRecord, StepStatus
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, assemble, scripted_profile, text_step, tool_step

ANTHROPIC = Origin(adapter="anthropic-messages", provider="anthropic", model="claude-opus-5-5", credential="key-a")
OTHER_MODEL = Origin(adapter="anthropic-messages", provider="anthropic", model="claude-sonnet-5-5", credential="key-a")
OTHER_KEY = Origin(adapter="anthropic-messages", provider="anthropic", model="claude-opus-5-5", credential="key-b")
GEMINI = Origin(adapter="gemini-interactions", provider="gemini", model="gemini-3.5-flash", credential="key-g")
OLLAMA = Origin(adapter="ollama-chat", provider="ollama", model="qwen3.6:27b", credential="local")

TURN_ONE = "turn-1"
TURN_TWO = "turn-2"


def insert_after_block(events: Sequence[StreamEvent], block: int, extra: StreamEvent) -> List[StreamEvent]:
    result = list(events)
    index = next(i for i, event in enumerate(result) if getattr(event, "block", None) == block)
    result.insert(index + 1, extra)
    return result


def signed(events: Sequence[StreamEvent], signature: str = "sig") -> List[StreamEvent]:
    return insert_after_block(events, 0, OpaqueDelta(block=0, key="signature", value=signature))


def with_raw(events: Sequence[StreamEvent], raw: Any) -> List[StreamEvent]:
    result = list(events)
    last = result[-1]
    result[-1] = Finish(reason=last.reason, raw_reason=last.raw_reason, raw=raw)
    return result


def step(seq: int, turn: str, msg: Message) -> StepRecord:
    return StepRecord(id=f"step-{seq}", session_id="session", turn_id=turn, seq=seq, message=msg, status=StepStatus.COMPLETED)


def user(seq: int, turn: str, text: str = "hello") -> StepRecord:
    return step(seq, turn, message(f"user-{seq}", "user", TextPart(text)))


def tool_result(seq: int, turn: str, call_id: str = "call_0") -> StepRecord:
    return step(seq, turn, message(f"tool-{seq}", "tool", ToolResultPart(call_id=call_id, name="lookup", content=(TextPart("ok"),))))


def assistant(seq: int, turn: str, events: Sequence[StreamEvent], origin: Origin = SCRIPTED_ORIGIN) -> StepRecord:
    return step(seq, turn, assemble(events, f"assistant-{seq}", origin))


def two_turn_history(origin: Origin = SCRIPTED_ORIGIN) -> List[StepRecord]:
    return [
        user(1, TURN_ONE),
        assistant(2, TURN_ONE, tool_step({"name": "lookup", "args": {"q": 1}}, reasoning="think one"), origin),
        tool_result(3, TURN_ONE),
        assistant(4, TURN_ONE, text_step("first answer", reasoning="think two"), origin),
        user(5, TURN_TWO, "next"),
        assistant(6, TURN_TWO, tool_step({"name": "lookup", "args": {"q": 2}, "id": "call_9"}, reasoning="think three"), origin),
        tool_result(7, TURN_TWO, "call_9"),
    ]


def run(steps, policy: ReplayPolicy, target: Origin = SCRIPTED_ORIGIN, turn_id: str = TURN_TWO):
    return policy_replay(steps, turn_id=turn_id, target=target, profile=scripted_profile(replay=policy.value))


def reasoning_texts(messages: Sequence[Message]) -> dict:
    return {m.id: [part.text for part in m.reasoning()] for m in messages if m.role is Role.ASSISTANT}


def test_send_all_keeps_the_reasoning_of_every_turn():
    messages, notices = run(two_turn_history(), ReplayPolicy.SEND_ALL)

    assert reasoning_texts(messages) == {
        "assistant-2": ["think one"], "assistant-4": ["think two"], "assistant-6": ["think three"],
    }
    assert notices == ()


def test_preserve_selects_the_same_messages_as_send_all():
    history = two_turn_history()

    assert run(history, ReplayPolicy.PRESERVE) == run(history, ReplayPolicy.SEND_ALL)


def test_drop_earlier_keeps_only_the_turn_in_progress():
    messages, notices = run(two_turn_history(), ReplayPolicy.DROP_EARLIER)

    assert reasoning_texts(messages) == {"assistant-2": [], "assistant-4": [], "assistant-6": ["think three"]}
    assert notices == ()


def test_drop_earlier_keeps_text_and_tool_calls_of_earlier_turns():
    messages, _ = run(two_turn_history(), ReplayPolicy.DROP_EARLIER)
    by_id = {m.id: m for m in messages}

    assert by_id["assistant-2"].tool_calls()[0].args == {"q": 1}
    assert by_id["assistant-4"].text() == "first answer"
    assert [m.id for m in messages] == ["user-1", "assistant-2", "tool-3", "assistant-4", "user-5", "assistant-6", "tool-7"]


def test_never_drops_reasoning_everywhere_including_the_turn_in_progress():
    messages, notices = run(two_turn_history(), ReplayPolicy.NEVER)

    assert reasoning_texts(messages) == {"assistant-2": [], "assistant-4": [], "assistant-6": []}
    assert notices == ()


def test_a_turn_id_that_matches_no_step_treats_every_step_as_earlier():
    messages, _ = run(two_turn_history(), ReplayPolicy.DROP_EARLIER, turn_id="turn-not-stored-yet")

    assert reasoning_texts(messages) == {"assistant-2": [], "assistant-4": [], "assistant-6": []}


def test_untouched_messages_are_passed_through_as_the_same_objects():
    history = two_turn_history()

    messages, _ = run(history, ReplayPolicy.DROP_EARLIER)

    assert messages[0] is history[0].message
    assert messages[2] is history[2].message
    assert messages[5] is history[5].message
    assert messages[1] is not history[1].message


def test_dropping_reasoning_discards_the_raw_payload_so_it_cannot_be_replayed():
    history = [
        user(1, TURN_ONE),
        assistant(2, TURN_ONE, with_raw(text_step("answer", reasoning="old"), {"provider": "payload"})),
        user(3, TURN_TWO),
        assistant(4, TURN_TWO, with_raw(text_step("now", reasoning="new"), {"provider": "payload-2"})),
    ]

    messages, _ = run(history, ReplayPolicy.DROP_EARLIER)

    assert messages[1].raw is None
    assert messages[3].raw == {"provider": "payload-2"}
    assert messages[3].replays_raw_to(SCRIPTED_ORIGIN)


def test_a_message_that_loses_everything_it_had_is_left_out():
    history = [
        user(1, TURN_ONE),
        step(2, TURN_ONE, message("only-thoughts", "assistant", ReasoningPart(text="hm"), origin=SCRIPTED_ORIGIN)),
        user(3, TURN_TWO),
    ]

    messages, _ = run(history, ReplayPolicy.DROP_EARLIER)

    assert [m.id for m in messages] == ["user-1", "user-3"]


def test_a_message_that_was_already_empty_is_kept():
    history = [step(1, TURN_ONE, message("empty", "assistant", origin=SCRIPTED_ORIGIN)), user(2, TURN_TWO)]

    messages, _ = run(history, ReplayPolicy.DROP_EARLIER)

    assert [m.id for m in messages] == ["empty", "user-2"]


def test_selection_is_deterministic():
    history = two_turn_history()

    assert run(history, ReplayPolicy.DROP_EARLIER) == run(history, ReplayPolicy.DROP_EARLIER)


class TestModelAndCredentialSwitch:
    def history(self, origin: Origin) -> List[StepRecord]:
        return [
            user(1, TURN_ONE),
            assistant(2, TURN_ONE, signed(tool_step({"name": "lookup", "args": {"q": 1}}, reasoning="private"), "sig-1"), origin),
            tool_result(3, TURN_ONE),
            assistant(4, TURN_ONE, signed(text_step("done", reasoning="more"), "sig-2"), origin),
        ]

    def test_signed_reasoning_is_kept_for_the_origin_that_made_it(self):
        messages, notices = run(self.history(ANTHROPIC), ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert reasoning_texts(messages) == {"assistant-2": ["private"], "assistant-4": ["more"]}
        assert messages[1].reasoning()[0].opaque == {"signature": "sig-1"}
        assert notices == ()

    def test_signed_reasoning_is_left_out_for_another_model(self):
        messages, notices = run(self.history(ANTHROPIC), ReplayPolicy.SEND_ALL, target=OTHER_MODEL)

        assert reasoning_texts(messages) == {"assistant-2": [], "assistant-4": []}
        assert [m.text() for m in messages if m.role is Role.ASSISTANT] == ["", "done"]
        assert len(notices) == 1
        assert notices[0].kind is NoticeKind.REASONING_DROPPED
        assert notices[0].message == "Reasoning made by claude-opus-5-5 was left out because the model changed."
        assert notices[0].step_seq == 4
        assert notices[0].detail == {"reason": "model", "model": "claude-opus-5-5", "steps": [2, 4]}

    def test_signed_reasoning_is_left_out_after_a_credential_change(self):
        messages, notices = run(self.history(ANTHROPIC), ReplayPolicy.SEND_ALL, target=OTHER_KEY)

        assert reasoning_texts(messages) == {"assistant-2": [], "assistant-4": []}
        assert len(notices) == 1
        assert notices[0].message == "Reasoning made with a different API key was left out."
        assert notices[0].detail == {"reason": "credential", "model": "claude-opus-5-5", "steps": [2, 4]}

    def test_a_change_of_adapter_counts_as_a_model_change(self):
        _, notices = run(self.history(ANTHROPIC), ReplayPolicy.SEND_ALL, target=GEMINI)

        assert notices[0].detail["reason"] == "model"

    def test_switching_back_restores_the_reasoning_because_the_stored_steps_are_untouched(self):
        history = self.history(ANTHROPIC)

        run(history, ReplayPolicy.SEND_ALL, target=OTHER_MODEL)
        messages, notices = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert reasoning_texts(messages) == {"assistant-2": ["private"], "assistant-4": ["more"]}
        assert notices == ()

    def test_redacted_signed_reasoning_is_left_out_too(self):
        history = [
            user(1, TURN_ONE),
            step(2, TURN_ONE, message("redacted", "assistant", ReasoningPart(redacted=True, opaque={"data": "enc"}), TextPart("ok"), origin=ANTHROPIC)),
        ]

        messages, notices = run(history, ReplayPolicy.SEND_ALL, target=OTHER_MODEL)

        assert messages[1].reasoning() == ()
        assert len(notices) == 1

    def test_unsigned_reasoning_text_travels_to_any_target(self):
        history = [
            user(1, TURN_ONE),
            assistant(2, TURN_ONE, text_step("hi", reasoning="plain thoughts"), OLLAMA),
        ]

        messages, notices = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert reasoning_texts(messages) == {"assistant-2": ["plain thoughts"]}
        assert notices == ()

    def test_legacy_history_keeps_its_reasoning_text_and_never_reports_a_drop(self):
        history = [
            user(1, TURN_ONE),
            assistant(2, TURN_ONE, text_step("old answer", reasoning="old thoughts"), LEGACY_ORIGIN),
        ]

        messages, notices = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert reasoning_texts(messages) == {"assistant-2": ["old thoughts"]}
        assert notices == ()

    def test_a_policy_drop_is_silent_even_for_foreign_signed_reasoning(self):
        _, notices = run(self.history(ANTHROPIC), ReplayPolicy.DROP_EARLIER, target=OTHER_MODEL, turn_id="later")

        assert notices == ()

    def test_never_stays_silent_for_foreign_signed_reasoning(self):
        _, notices = run(self.history(ANTHROPIC), ReplayPolicy.NEVER, target=OTHER_MODEL)

        assert notices == ()

    def test_a_switch_in_the_middle_of_a_turn_drops_foreign_reasoning_of_that_turn(self):
        history = [
            user(1, TURN_ONE),
            assistant(2, TURN_ONE, signed(tool_step({"name": "lookup", "args": {}}, reasoning="mine"), "sig"), ANTHROPIC),
            tool_result(3, TURN_ONE),
        ]

        messages, notices = run(history, ReplayPolicy.SEND_ALL, target=OTHER_MODEL, turn_id=TURN_ONE)

        assert reasoning_texts(messages) == {"assistant-2": []}
        assert notices[0].step_seq == 2

    def test_each_foreign_model_gets_its_own_notice(self):
        history = [
            user(1, TURN_ONE),
            assistant(2, TURN_ONE, signed(text_step("a", reasoning="x")), ANTHROPIC),
            user(3, TURN_ONE),
            assistant(4, TURN_ONE, signed(text_step("b", reasoning="y")), OTHER_MODEL),
            user(5, TURN_TWO),
        ]

        _, notices = run(history, ReplayPolicy.SEND_ALL, target=GEMINI)

        assert [(n.detail["model"], n.step_seq) for n in notices] == [("claude-opus-5-5", 2), ("claude-sonnet-5-5", 4)]


class TestForeignToolCalls:
    def history(self, origin: Origin) -> List[StepRecord]:
        events = insert_after_block(
            tool_step({"name": "lookup", "args": {"q": 1}}, {"name": "other", "args": {}, "id": "call_1"}),
            2,
            OpaqueDelta(block=2, key="thought_signature", value="real-signature"),
        )
        return [user(1, TURN_ONE), assistant(2, TURN_ONE, events, origin), tool_result(3, TURN_ONE)]

    def test_foreign_tool_calls_lose_provider_data_and_carry_the_placeholder_marker(self):
        messages, _ = run(self.history(GEMINI), ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        calls = messages[1].tool_calls()
        assert [call.opaque for call in calls] == [{PLACEHOLDER_SIGNATURE: True}, {PLACEHOLDER_SIGNATURE: True}]
        assert [(call.call_id, call.name, call.args) for call in calls] == [
            ("call_0", "lookup", {"q": 1}), ("call_1", "other", {}),
        ]

    def test_tool_calls_made_by_the_target_keep_their_own_signature(self):
        history = self.history(GEMINI)

        messages, _ = run(history, ReplayPolicy.SEND_ALL, target=GEMINI)

        assert messages[1] is history[1].message
        assert messages[1].tool_calls()[0].opaque == {"thought_signature": "real-signature"}
        assert messages[1].tool_calls()[1].opaque is None

    def test_legacy_tool_calls_are_foreign_to_every_target(self):
        messages, _ = run(self.history(LEGACY_ORIGIN), ReplayPolicy.SEND_ALL, target=GEMINI)

        assert all(call.opaque == {PLACEHOLDER_SIGNATURE: True} for call in messages[1].tool_calls())

    def test_tool_results_are_never_altered(self):
        history = self.history(GEMINI)

        messages, _ = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert messages[2] is history[2].message

    def test_a_credential_change_also_clears_tool_call_provider_data(self):
        messages, _ = run(self.history(ANTHROPIC), ReplayPolicy.SEND_ALL, target=OTHER_KEY)

        assert all(call.opaque == {PLACEHOLDER_SIGNATURE: True} for call in messages[1].tool_calls())

    def test_user_messages_are_never_altered_even_when_they_carry_a_compaction(self):
        history = [step(1, TURN_ONE, message("compact", "user", CompactionPart(summary="we agreed on X", opaque={"blob": "enc"})))]

        messages, _ = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert messages[0] is history[0].message

    def test_foreign_assistant_compaction_keeps_its_summary_and_loses_its_payload(self):
        history = [
            step(1, TURN_ONE, message("compact", "assistant", CompactionPart(summary="we agreed on X", opaque={"blob": "enc"}), origin=GEMINI)),
        ]

        messages, _ = run(history, ReplayPolicy.SEND_ALL, target=ANTHROPIC)

        assert messages[0].parts == (CompactionPart(summary="we agreed on X"),)


class TestMissingReasoning:
    def test_every_assistant_message_with_reasoning_passes(self):
        messages, _ = run(two_turn_history(), ReplayPolicy.SEND_ALL)

        assert missing_reasoning(messages) == ()

    def test_assistant_messages_without_reasoning_are_listed_in_order(self):
        messages, _ = run(two_turn_history(), ReplayPolicy.DROP_EARLIER)

        assert missing_reasoning(messages) == ("assistant-2", "assistant-4")

    def test_a_blank_reasoning_part_does_not_count(self):
        blank = message("blank", "assistant", ReasoningPart(text=""), TextPart("hi"), origin=SCRIPTED_ORIGIN)

        assert missing_reasoning([blank]) == ("blank",)

    def test_summary_only_and_signed_only_reasoning_counts(self):
        summary = message("summary", "assistant", ReasoningPart(summary="short"), TextPart("hi"), origin=SCRIPTED_ORIGIN)
        signed_only = message("signed", "assistant", ReasoningPart(opaque={"signature": "s"}), TextPart("hi"), origin=SCRIPTED_ORIGIN)

        assert missing_reasoning([summary, signed_only]) == ()

    def test_user_and_tool_messages_are_ignored(self):
        history = [user(1, TURN_ONE), tool_result(2, TURN_ONE)]

        assert missing_reasoning([item.message for item in history]) == ()


@pytest.mark.parametrize("policy", list(ReplayPolicy))
def test_an_empty_history_selects_nothing(policy):
    assert run([], policy) == ((), ())

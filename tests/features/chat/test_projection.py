from dataclasses import replace

import pytest

from src.features.chat import projection
from src.features.chat.dto import MessageResponse
from src.features.llm.harness.events import Usage
from src.features.llm.harness.messages import ReasoningPart, TextPart, ToolCallPart, message
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
from tests.fixtures.llm_turns import ask_step, final_step, result_step, state, user_step


def reasoning_step(seq, *parts, usage=None):
    return StepRecord(
        id=f"s1.a{seq}", session_id="s1", turn_id="t1", seq=seq,
        message=message(f"ma{seq}", "assistant", *parts, origin=SCRIPTED_ORIGIN),
        status=StepStatus.COMPLETED, usage=usage,
    )


def assistant_row(**fields):
    base = dict(
        id="row-a", session_id="s1", role="assistant", content="Done.", turn_id="t1",
        metadata={"step_range": {"first": 2, "last": 4}},
    )
    return MessageResponse(**{**base, **fields})


def test_reasoning_kinds_follow_what_the_provider_returned():
    assert projection.reasoning_block(ReasoningPart(text="chain")).kind == "raw"
    summary = projection.reasoning_block(ReasoningPart(summary="Checked both tabs."))
    assert (summary.kind, summary.text) == ("summary", "Checked both tabs.")
    assert projection.reasoning_block(ReasoningPart(text="chain", summary="short")).text == "chain"
    omitted = projection.reasoning_block(ReasoningPart())
    assert (omitted.kind, omitted.text, omitted.redacted) == ("omitted", "", False)
    assert projection.reasoning_block(ReasoningPart(redacted=True, opaque={"data": "x"})).redacted is True
    assert projection.reasoning_block(ReasoningPart(text="chain", duration_ms=1840)).duration_ms == 1840
    assert projection.reasoning_block(ReasoningPart(summary="short", duration_ms=7)).duration_ms == 7
    assert projection.reasoning_block(ReasoningPart()).duration_ms is None


def test_reasoning_blocks_keep_step_order_and_skip_other_roles():
    steps = [
        user_step(1),
        reasoning_step(2, ReasoningPart(text="first"), ReasoningPart(), TextPart(text="a")),
        result_step(3, [("c", "t", "r", False)]),
        reasoning_step(4, ReasoningPart(summary="last")),
    ]

    assert [(block.kind, block.text) for block in projection.reasoning_blocks(steps)] == [
        ("raw", "first"), ("omitted", ""), ("summary", "last"),
    ]


def test_the_row_text_is_every_assistant_text_in_order():
    steps = [
        user_step(1, text="never part of the answer"),
        ask_step(2, [("c1", "search", {})], text="Let me check. "),
        result_step(3, [("c1", "search", "found", False)]),
        final_step(4, "Found it."),
    ]

    assert projection.segment_text(steps) == "Let me check. Found it."


def test_usage_adds_up_across_steps_and_stays_unknown_when_nothing_reported():
    steps = [
        ask_step(2, [("c", "t", {})], usage=Usage(input_tokens=100, output_tokens=10, cached_input_tokens=80, cost=0.001)),
        final_step(4, "x", usage=Usage(input_tokens=150, output_tokens=20, reasoning_tokens=5, cost=0.002)),
    ]

    usage = projection.segment_usage(steps)

    assert (usage.input_tokens, usage.output_tokens, usage.reasoning_tokens) == (250, 30, 5)
    assert (usage.cached_input_tokens, usage.cache_write_tokens) == (80, None)
    assert usage.cost == pytest.approx(0.003)
    assert projection.segment_usage([user_step(1), final_step(2, "x")]) is None


def test_notices_belong_to_the_row_holding_their_step():
    current = state(notices=(
        Notice(kind=NoticeKind.REASONING_DROPPED, message="dropped", step_seq=3),
        Notice(kind=NoticeKind.RETRYING, message="retry", step_seq=9),
        Notice(kind=NoticeKind.TRUNCATED, message="cut"),
    ))

    inside = projection.segment_notices(current, 2, 4, closes_turn=False)
    closing = projection.segment_notices(current, 5, 9, closes_turn=True)

    assert [(item.kind, item.step_seq) for item in inside] == [("reasoning_dropped", 3)]
    assert [(item.kind, item.step_seq) for item in closing] == [("retrying", 9), ("truncated", None)]
    assert projection.segment_notices(None, 1, 9, closes_turn=True) == []


def test_executions_pair_each_call_with_its_result():
    steps = [
        user_step(1),
        ask_step(2, [("c1", "search", {"q": "cats"}), ("c2", "rename", {"id": 1})]),
        result_step(3, [("c1", "search", "2 hits", False), ("c2", "rename", "name taken", True)]),
    ]

    first, second = projection.tool_executions(steps, state())

    assert (first["tool_name"], first["arguments"], first["tool_call_id"]) == ("search", {"q": "cats"}, "c1")
    assert first["result"] == {"success": True, "data": "2 hits", "error": None}
    assert second["result"] == {"success": False, "data": "", "error": "name taken"}
    assert first["approval_status"] is None and first["skipped"] is False


def test_a_call_waiting_for_a_decision_is_pending_with_its_preview():
    steps = [user_step(1), ask_step(2, [("c1", "search", {}), ("c2", "delete_prompt", {"id": 7})])]
    waiting = state(
        status=TurnStatus.AWAITING_DECISION,
        pending=(PendingCall(call_id="c2", name="delete_prompt", decision=DecisionKind.APPROVAL, preview={"title": "old"}),),
        gathered=(result_step(3, [("c1", "search", "found", False)]).message.tool_results()[0],),
    )

    first, second = projection.tool_executions(steps, waiting)

    assert first["result"]["data"] == "found" and first["approval_status"] is None
    assert (second["approval_status"], second["preview"]) == ("pending", {"title": "old"})
    assert second["result"] == {"success": True, "data": "", "error": None}


def test_live_records_supply_what_steps_do_not_carry():
    steps = [ask_step(2, [("c1", "rename", {"id": 1})]), result_step(3, [("c1", "rename", "Renamed.", False)])]
    records = {"c1": {
        "duration_ms": 41, "approval_status": "approved", "approval_reason": None, "skipped": False,
        "preview": {"before": "a"}, "result": {"success": True, "data": "Renamed.", "error": None},
    }}

    (execution,) = projection.tool_executions(steps, state(), records)

    assert (execution["duration_ms"], execution["approval_status"], execution["preview"]) == (41, "approved", {"before": "a"})


def test_a_call_without_any_result_has_an_empty_outcome():
    (execution,) = projection.tool_executions([ask_step(2, [("c1", "search", {})])], state())

    assert execution["result"] == {"success": True, "data": "", "error": None}


def test_assistant_metadata_records_the_window_rounds_tokens_and_tool_cards():
    steps = [
        ask_step(2, [("c1", "search", {})], usage=Usage(input_tokens=100, output_tokens=10)),
        result_step(3, [("c1", "search", "found", False)]),
        final_step(4, "Done.", usage=Usage(input_tokens=150, output_tokens=20)),
    ]

    built = projection.assistant_metadata(
        steps, state(rounds_used=2), context_metadata={"form_state": {"a": 1}}, extra={"behavior_trace": {"v": 1}},
    )

    assert built["step_range"] == {"first": 2, "last": 4}
    assert (built["turn_rounds"], built["prompt_tokens"], built["completion_tokens"], built["tokens_used"]) == (2, 250, 30, 280)
    assert built["behavior_trace"] == {"v": 1}
    assert built["context_metadata"] == {"form_state": {"a": 1}}
    assert [item["tool_call_id"] for item in built["tool_executions"]] == ["c1"]
    assert projection.step_range(built) == (2, 4)


def test_a_text_only_row_has_no_tool_cards_or_context():
    built = projection.assistant_metadata([final_step(2, "hi")], state(), context_metadata={"a": 1})

    assert "tool_executions" not in built and "context_metadata" not in built and "prompt_tokens" not in built


@pytest.mark.parametrize("window", [None, "x", {"first": 1}, {"first": "1", "last": 2}, {"first": True, "last": 2}])
def test_a_malformed_step_range_is_ignored(window):
    assert projection.step_range({"step_range": window}) is None


def test_legacy_reasoning_never_rides_in_metadata():
    metadata = {
        "thinking": {"segments": [{"text": "secret"}]},
        "tool_transcript": [{"calls": [], "results": [], "reasoning": "secret"}, "odd"],
        "behavior_trace": {"thinking_visible": True, "steps": []},
        "tool_executions": [{"tool_name": "t"}],
    }

    cleaned = projection.metadata_without_reasoning(metadata)

    assert cleaned == {
        "tool_transcript": [{"calls": [], "results": []}, "odd"],
        "behavior_trace": {"steps": []},
        "tool_executions": [{"tool_name": "t"}],
    }
    assert "thinking" in metadata
    clean = {"tool_executions": []}
    assert projection.metadata_without_reasoning(clean) is clean
    assert projection.metadata_without_reasoning(None) is None


def conversation():
    steps = [
        user_step(1),
        reasoning_step(2, ReasoningPart(text="plan"), TextPart(text="Looking."), ToolCallPart(call_id="c1", name="search", args={}), usage=Usage(input_tokens=10)),
        result_step(3, [("c1", "search", "found", False)]),
        reasoning_step(4, ReasoningPart(summary="wrap up"), TextPart(text="Done."), usage=Usage(input_tokens=20, output_tokens=5)),
    ]
    return {"t1": steps}, {"t1": state(notices=(Notice(kind=NoticeKind.REASONING_DROPPED, message="dropped", step_seq=3),))}


def test_reasoning_is_attached_only_when_thinking_is_shown():
    steps, states = conversation()
    row = assistant_row()

    shown = projection.present([row], steps, states, show_thinking=True)[0]
    hidden = projection.present([row], steps, states, show_thinking=False)[0]

    assert [(block.kind, block.text) for block in shown.reasoning] == [("raw", "plan"), ("summary", "wrap up")]
    assert hidden.reasoning is None
    assert shown.usage.input_tokens == hidden.usage.input_tokens == 30
    assert [(item.kind, item.step_seq) for item in shown.notices] == [("reasoning_dropped", 3)]


def test_a_hidden_reasoning_key_is_absent_from_the_payload_not_empty():
    steps, states = conversation()
    row = assistant_row()

    hidden = projection.present([row], steps, states, show_thinking=False)[0].model_dump()
    shown = projection.present([row], steps, states, show_thinking=True)[0].model_dump()

    assert "reasoning" not in hidden
    assert shown["reasoning"][0] == {"kind": "raw", "text": "plan", "redacted": False, "duration_ms": None}
    assert shown["usage"]["cost"] is None and shown["notices"] == [{"kind": "reasoning_dropped", "message": "dropped", "step_seq": 3}]


def test_a_shown_row_without_reasoning_carries_an_empty_list():
    steps = {"t1": [user_step(1), final_step(2, "plain")]}
    row = assistant_row(metadata={"step_range": {"first": 2, "last": 2}})

    shown = projection.present([row], steps, {"t1": state()}, show_thinking=True)[0]

    assert shown.reasoning == []
    assert shown.notices is None and shown.usage is None


def test_rows_outside_a_window_are_left_alone():
    steps, states = conversation()
    user = assistant_row(id="row-u", role="user", metadata=None, turn_id="t1")
    unprojected = assistant_row(id="row-x", metadata={"tokens_used": 5}, turn_id=None)
    outside = assistant_row(id="row-y", metadata={"step_range": {"first": 8, "last": 9}})

    presented = projection.present([user, unprojected, outside], steps, states, show_thinking=True)

    assert presented == [user, unprojected, outside]


def test_legacy_thinking_is_stripped_whether_or_not_it_is_shown():
    steps, states = conversation()
    row = assistant_row(metadata={"step_range": {"first": 2, "last": 4}, "thinking": {"segments": [{"text": "old"}]}})

    for show in (True, False):
        (presented,) = projection.present([row], steps, states, show_thinking=show)
        assert "thinking" not in presented.metadata
        assert presented.metadata["step_range"] == {"first": 2, "last": 4}
    assert "thinking" in row.metadata


def test_the_second_row_of_a_turn_carries_the_turn_wide_notice():
    steps, states = conversation()
    states["t1"] = replace(states["t1"], notices=(Notice(kind=NoticeKind.TRUNCATED, message="cut"),))
    first = assistant_row(id="row-1", metadata={"step_range": {"first": 2, "last": 3}})
    second = assistant_row(id="row-2", metadata={"step_range": {"first": 4, "last": 4}})

    a, b = projection.present([first, second], steps, states, show_thinking=False)

    assert a.notices is None
    assert [item.kind for item in b.notices] == ["truncated"]

import json
from typing import Any, Dict

import pytest

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.build import user_message
from src.features.llm.harness.engine import (
    TOOL_BUDGET_EXHAUSTED_MESSAGE,
    BudgetExhausted,
    TurnInput,
    TurnOutcome,
)
from src.features.llm.harness.baseline import InMemoryTurnStore
from src.features.llm.harness.errors import InvalidRequest, Overloaded
from src.features.llm.harness.retry import BackoffRetry
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import ContextPart, ImagePart, Message, ReasoningPart, Role, TextPart, ToolCallPart
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import (
    DecisionKind,
    NoticeKind,
    NoticeRaised,
    StepCommitted,
    StepRecord,
    StepStatus,
    ToolFinished,
    ToolStarted,
    TurnSnapshot,
    TurnState,
    TurnStatus,
)
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from tests.fixtures.llm_engine import drain, scripted_config, scripted_engine, tool_context
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, FailMidStream, ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes a message back."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


class BrokenTool(BaseTool):
    name = "broken"
    description = "Always fails."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=False, data="", error="disk on fire")


class SnapshotTool(BaseTool):
    name = "snapshot"
    description = "Returns an image."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="here it is", image_data="aW1n")


class DeleteTool(BaseTool):
    name = "delete_prompt"
    description = "Deletes a prompt after approval."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}
    requires_approval = True

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Delete", target=kwargs.get("id")))


def turn(**overrides: Any) -> TurnInput:
    values: Dict[str, Any] = dict(
        session_id="s1",
        llm_id="cfg-scripted",
        system="SYS",
        history=(),
        new_messages=(user_message("find cats", message_id="u1"),),
        tool_context=tool_context(),
    )
    values.update(overrides)
    return TurnInput(**values)


def outcome_of(events) -> TurnOutcome:
    outcomes = [event for event in events if isinstance(event, TurnOutcome)]
    assert len(outcomes) == 1
    return outcomes[0]


def visible(events) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


async def test_a_plain_answer_without_tools_streams_and_commits_both_messages():
    adapter = ScriptedAdapter(text_step("hello there", reasoning="thinking"))
    rig = scripted_engine(adapter)

    events = await drain(rig.engine.run(turn(tool_context=None)))

    outcome = outcome_of(events)
    assert visible(events) == "hello there"
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == ["thinking"]
    assert outcome.content == "hello there" and outcome.reason == "answer"
    assert outcome.state.status is TurnStatus.COMPLETED
    steps = rig.store.steps("s1")
    assert [step.seq for step in steps] == [0, 1]
    assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT]
    assert steps[1].message.parts == (ReasoningPart(text="thinking", duration_ms=0), TextPart(text="hello there"))
    assert steps[1].usage == Usage(input_tokens=10, output_tokens=5, total_tokens=15)
    assert steps[1].finish is FinishReason.STOP
    assert adapter.requests[0].tools == ()
    assert adapter.requests[0].system == "SYS"


async def test_a_tool_round_runs_the_call_and_sends_reasoning_and_results_back():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "hi"}}, reasoning="need echo", text="Let me check."),
        text_step("Done."),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    outcome = outcome_of(events)
    assert outcome.content == "Done."
    assert outcome.rounds_used == 2
    assert [execution.tool_name for execution in outcome.executions] == ["echo"]
    assert [type(event).__name__ for event in events if isinstance(event, (ToolStarted, ToolFinished))] == ["ToolStarted", "ToolFinished"]
    second = adapter.requests[1].messages
    assert [message.role for message in second] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert second[1].reasoning() == (ReasoningPart(text="need echo", duration_ms=0),)
    assert second[1].text() == "Let me check."
    assert second[1].tool_calls()[0].args == {"message": "hi"}
    assert second[2].tool_results()[0].text() == "echo: hi"
    assert second[2].tool_results()[0].call_id == "c1"
    assert [tool.name for tool in adapter.requests[0].tools] == ["echo"]
    assert outcome.transcript == ({
        "calls": [{"id": "c1", "name": "echo", "arguments": {"message": "hi"}}],
        "results": [{"tool_call_id": "c1", "content": "echo: hi"}],
        "reasoning": "need echo",
    },)


async def test_an_xml_tool_call_stays_hidden_and_is_dispatched_after_the_step():
    xml = '<tool_call>{"name": "echo", "arguments": {"message": "x"}}</tool_call>'
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text="Sure. "), TextDelta(block=1, text=xml[:20]), TextDelta(block=1, text=xml[20:]), Finish(reason=FinishReason.STOP)],
        text_step("All set."),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert "<tool_call>" not in visible(events)
    assert visible(events).startswith("Sure.")
    dispatched = adapter.requests[1].messages[1]
    assert dispatched.text() == "Sure."
    assert dispatched.tool_calls()[0].name == "echo"
    assert dispatched.tool_calls()[0].args == {"message": "x"}
    assert outcome_of(events).content == "All set."


async def test_an_approval_gated_call_pauses_with_pending_and_gathered_results():
    adapter = ScriptedAdapter(
        tool_step(
            {"id": "c1", "name": "echo", "args": {"message": "a"}},
            {"id": "c2", "name": "delete_prompt", "args": {"id": "p1"}},
            {"id": "c3", "name": "echo", "args": {"message": "b"}},
        ),
    )
    rig = scripted_engine(adapter, [EchoTool(), DeleteTool()])

    events = await drain(rig.engine.run(turn()))

    outcome = outcome_of(events)
    assert outcome.pending and outcome.content == ""
    state = outcome.state
    assert state.status is TurnStatus.AWAITING_DECISION
    assert [(call.call_id, call.decision) for call in state.pending] == [("c2", DecisionKind.APPROVAL)]
    assert state.pending[0].preview["action"] == "Delete"
    assert [part.call_id for part in state.gathered] == ["c1", "c3"]
    assert state.gathered[1].is_error and "waiting for approval of delete_prompt" in state.gathered[1].text()
    assert [step.message.role for step in rig.store.steps("s1")] == [Role.USER, Role.ASSISTANT]
    assert rig.store.open_turn("s1") == state
    assert [(e.tool_name, e.approval_status, e.skipped) for e in outcome.executions] == [
        ("echo", None, False), ("delete_prompt", "pending", False), ("echo", None, True),
    ]
    assert [result["tool_call_id"] for result in outcome.transcript[0]["results"]] == ["c1", "c3"]


async def test_the_budget_wrap_up_appends_a_note_and_calls_without_tools():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}),
        text_step("Out of budget, here is what I have."),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn(max_rounds=1)))

    assert [event.max_rounds for event in events if isinstance(event, BudgetExhausted)] == [1]
    final = adapter.requests[1]
    assert final.tools == ()
    note = final.messages[-1]
    assert note.role is Role.USER
    assert note.parts == (ContextPart(kind="budget", text=f"<context>\n{TOOL_BUDGET_EXHAUSTED_MESSAGE}\n</context>"),)
    outcome = outcome_of(events)
    assert outcome.reason == "budget" and outcome.content == "Out of budget, here is what I have."
    assert outcome.completion == {"reason": "stop", "raw": "stop"}


async def test_the_iteration_nudge_is_appended_once_as_its_own_message():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}),
        tool_step({"id": "c2", "name": "echo", "args": {"message": "b"}}),
        text_step("done"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn(iteration_nudge="Keep going.")))

    first_user = adapter.requests[2].messages[0]
    assert first_user.parts == (TextPart(text="find cats"),)
    nudges = [
        message for message in adapter.requests[2].messages
        if any(isinstance(part, ContextPart) and part.kind == "nudge" for part in message.parts)
    ]
    assert len(nudges) == 1
    assert [message.role for message in adapter.requests[1].messages] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.USER]


async def test_a_truncated_xml_call_steers_with_a_trailing_note_then_answers():
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text='Working <tool_call>{"name": "echo", "arguments": {"mess'), Finish(reason=FinishReason.LENGTH)],
        text_step("Recovered."),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    retry = adapter.requests[1].messages
    assert retry[0].parts == (TextPart(text="find cats"),)
    assert retry[1].role is Role.ASSISTANT and "<tool_call>" not in retry[1].text()
    assert retry[2].role is Role.USER and retry[2].parts[0].kind == "rescue"
    assert outcome_of(events).content == "Recovered."


async def test_an_empty_response_is_retried_inside_the_tool_loop():
    adapter = ScriptedAdapter([Finish(reason=FinishReason.STOP)], text_step("second try"))
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome_of(events).content == "second try"
    assert len(adapter.requests) == 2
    assert [step.message.role for step in rig.store.steps("s1")] == [Role.USER, Role.ASSISTANT]


async def test_an_empty_answer_without_tools_is_retried_three_times_at_most():
    empty = [Finish(reason=FinishReason.STOP)]
    adapter = ScriptedAdapter(empty, empty, empty, text_step("never"))
    rig = scripted_engine(adapter)

    events = await drain(rig.engine.run(turn(tool_context=None)))

    assert len(adapter.requests) == 3
    assert outcome_of(events).content == ""


async def test_a_failed_call_is_not_rerun_with_the_same_arguments():
    broken = BrokenTool()
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "broken", "args": {}}),
        tool_step({"id": "c2", "name": "broken", "args": {}}),
        text_step("gave up"),
    )
    rig = scripted_engine(adapter, [broken])

    events = await drain(rig.engine.run(turn()))

    assert broken.calls == 1
    results = adapter.requests[2].messages[-1].tool_results()
    assert results[0].is_error and "already called broken" in results[0].text()
    assert outcome_of(events).tool_failures == {"broken": 1}
    assert outcome_of(events).state.failed_calls == {"broken:{}": "disk on fire"}


async def test_a_tool_image_travels_inside_the_tool_result():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "snapshot", "args": {}}), text_step("I see it"))
    rig = scripted_engine(adapter, [SnapshotTool()])

    await drain(rig.engine.run(turn()))

    result = adapter.requests[1].messages[-1].tool_results()[0]
    assert result.content == (TextPart(text="here it is"), ImagePart(media_type="image/jpeg", data="aW1n"))


async def test_closing_mid_stream_commits_the_partial_step_as_cancelled():
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text="partial "), TextDelta(block=1, text="answer"), Finish(reason=FinishReason.STOP)],
    )
    rig = scripted_engine(adapter)
    events = rig.engine.run(turn(tool_context=None))

    async for event in events:
        if isinstance(event, TextDelta):
            break
    await events.aclose()

    steps = rig.store.steps("s1")
    assert steps[-1].status is StepStatus.CANCELLED
    assert steps[-1].message.text() == "partial "
    state = rig.store.load_state(steps[-1].turn_id)
    assert state.status is TurnStatus.CANCELLED


async def test_a_provider_error_fails_the_turn_and_records_it():
    adapter = ScriptedAdapter(Overloaded("busy", status=529))
    rig = scripted_engine(adapter)

    with pytest.raises(Overloaded):
        await drain(rig.engine.run(turn(tool_context=None)))

    turn_id = rig.store.steps("s1")[0].turn_id
    state = rig.store.load_state(turn_id)
    assert state.status is TurnStatus.FAILED
    assert state.error["kind"] == "overloaded"


class RetryOnce:
    def delay(self, error, attempt):
        return 2.5 if attempt == 0 and error.retryable else None


async def test_a_retryable_error_before_any_event_is_retried_with_a_notice():
    adapter = ScriptedAdapter(Overloaded("busy"), text_step("back"))
    rig = scripted_engine(adapter, retry=RetryOnce())

    events = await drain(rig.engine.run(turn(tool_context=None)))

    notices = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in notices] == [NoticeKind.RETRYING]
    assert rig.sleeps == [2.5]
    assert outcome_of(events).content == "back"
    assert outcome_of(events).state.notices == tuple(notices)


async def test_steps_are_committed_with_consecutive_seqs_after_earlier_history():
    adapter = ScriptedAdapter(text_step("one"), text_step("two"))
    rig = scripted_engine(adapter)

    first = await drain(rig.engine.run(turn(tool_context=None)))
    second = await drain(rig.engine.run(turn(
        tool_context=None,
        history=tuple(step.message for step in rig.store.steps("s1")),
        new_messages=(user_message("again", message_id="u2"),),
    )))

    assert [step.seq for step in rig.store.steps("s1")] == [0, 1, 2, 3]
    assert outcome_of(first).state.first_seq == 0 and outcome_of(second).state.first_seq == 2
    assert [message.text() for message in adapter.requests[1].messages] == ["find cats", "one", "again"]


async def test_streamed_tool_arguments_reach_the_next_request_verbatim():
    adapter = ScriptedAdapter(
        [
            ToolCallStart(block=2, call_id="c1", name="echo"),
            ToolArgsDelta(block=2, text='{"message": "x"}'),
            ToolCallEnd(block=2),
            UsageReported(Usage(input_tokens=1)),
            Finish(reason=FinishReason.TOOL_CALLS),
        ],
        text_step("ok"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn()))

    call = adapter.requests[1].messages[1].tool_calls()[0]
    assert call == ToolCallPart(call_id="c1", name="echo", args={"message": "x"}, args_raw=json.dumps({"message": "x"}))


async def test_a_turn_records_the_seq_of_the_earliest_history_step_its_budget_trimmed():
    adapter = ScriptedAdapter(text_step("short answer"))
    rig = scripted_engine(adapter, config=scripted_config(max_tokens=200, profile_overrides={"context_window": 1024}))
    stored = [
        user_message("x" * 4000, message_id="h0"),
        Message(id="h1", role=Role.ASSISTANT, origin=SCRIPTED_ORIGIN, parts=(TextPart(text="y" * 4000),)),
        user_message("recent", message_id="h2"),
        Message(id="h3", role=Role.ASSISTANT, origin=SCRIPTED_ORIGIN, parts=(TextPart(text="ok"),)),
    ]
    rig.store.save_state(_prior_state())
    for seq, message in enumerate(stored):
        rig.store.append(StepRecord(
            id=f"st{seq}", session_id="s1", turn_id="t-old", seq=seq, message=message, status=StepStatus.COMPLETED,
        ))

    events = await drain(rig.engine.run(TurnInput(
        session_id="s1", llm_id="cfg-scripted", system="SYS", history=tuple(stored),
        new_messages=(user_message("now", message_id="u1"),),
    )))

    assert [message.id for message in adapter.requests[0].messages] == ["h2", "h3", "u1"]
    assert outcome_of(events).state.prefix_changed_at == 0


async def test_a_turn_that_trims_nothing_records_no_changed_prefix():
    adapter = ScriptedAdapter(text_step("answer"))
    rig = scripted_engine(adapter)

    events = await drain(rig.engine.run(TurnInput(
        session_id="s1", llm_id="cfg-scripted", system="SYS", history=(), new_messages=(user_message("hi"),),
    )))

    assert outcome_of(events).state.prefix_changed_at is None


def _prior_state():
    snapshot = TurnSnapshot(
        llm_config_id="cfg-scripted", system="SYS", tools=(), settings=Settings(),
        profile=CapabilityProfile(adapter="scripted", model="scripted-model"),
    )
    return TurnState(turn_id="t-old", session_id="s1", status=TurnStatus.COMPLETED, snapshot=snapshot, first_seq=0)


def _plain_turn(**overrides):
    values = dict(session_id="s1", llm_id="cfg-scripted", system="SYS", history=(), new_messages=(user_message("hi"),))
    values.update(overrides)
    return TurnInput(**values)


async def test_a_retryable_error_before_anything_visible_is_retried_and_leaves_no_failed_step():
    early = FailMidStream(
        [UsageReported(Usage(input_tokens=12, output_tokens=0, total_tokens=12)), ReasoningDelta(block=0, text="")],
        Overloaded("busy", status=529),
    )
    adapter = ScriptedAdapter(early, text_step("hello"))
    rig = scripted_engine(adapter, retry=BackoffRetry(max_retries=1, jitter=lambda: 0.0))

    events = await drain(rig.engine.run(_plain_turn()))

    steps = rig.store.steps("s1")
    assert [step.status for step in steps] == [StepStatus.COMPLETED, StepStatus.COMPLETED]
    assert steps[-1].message.text() == "hello" and steps[-1].message.reasoning() == ()
    assert outcome_of(events).content == "hello"
    assert not [event for event in events if isinstance(event, ReasoningDelta)]


@pytest.mark.parametrize("error", [InvalidRequest("bad request", status=400), RuntimeError("socket closed")])
async def test_a_failure_after_only_empty_reasoning_stores_no_empty_failed_step(error):
    early = FailMidStream([ReasoningDelta(block=0, text="")], error)
    rig = scripted_engine(ScriptedAdapter(early))

    with pytest.raises(type(error)):
        await drain(rig.engine.run(_plain_turn()))

    assert [step.message.role for step in rig.store.steps("s1")] == [Role.USER]


class _FailsTheFinalSave(InMemoryTurnStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def save_state(self, state):
        if state.status is TurnStatus.COMPLETED and not self.failed:
            self.failed = True
            raise OSError("disk full")
        super().save_state(state)


async def test_a_turn_whose_final_save_fails_is_recorded_failed_not_left_running():
    store = _FailsTheFinalSave()
    rig = scripted_engine(ScriptedAdapter(text_step("hello")), store=store)

    with pytest.raises(OSError):
        await drain(rig.engine.run(_plain_turn(turn_id="t-save")))

    assert store.load_state("t-save").status is TurnStatus.FAILED

import asyncio
from typing import Any, AsyncIterator, Dict, List

import pytest

from src.features.llm.harness.adapter import WireRequest
from src.features.llm.harness.build import user_message
from src.features.llm.harness.engine import TurnInput, TurnOutcome
from src.features.llm.harness.events import Finish, FinishReason, ReasoningDelta, TextDelta
from src.features.llm.harness.errors import OpaqueViolation
from src.features.llm.harness.messages import Message, ReasoningPart, Role, TextPart
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised, ToolStarted
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from tests.fixtures.llm_engine import drain, scripted_engine
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.calls += 1
        return ToolResult(success=True, data="echoed")


class GateTool(BaseTool):
    name = "gate"
    description = "Needs approval."
    parameters = {"type": "object", "properties": {}}
    requires_approval = True

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Open"))


def turn(**overrides: Any) -> TurnInput:
    values: Dict[str, Any] = dict(
        session_id="s1",
        llm_id="cfg-scripted",
        system="SYS",
        history=(),
        new_messages=(user_message("go", message_id="u1"),),
        tool_context=ToolContext(user_id="user-1"),
    )
    values.update(overrides)
    return TurnInput(**values)


def outcome(events: List[Any]) -> TurnOutcome:
    return next(event for event in events if isinstance(event, TurnOutcome))


def call_messages(request) -> List[Any]:
    return [message for message in request.messages if message.tool_calls()]


async def test_each_rounds_reasoning_rides_its_own_tool_call_into_later_requests():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {}}, reasoning="R1"),
        tool_step({"id": "c2", "name": "echo", "args": {}}),
        tool_step({"id": "c3", "name": "echo", "args": {}}, reasoning="R3"),
        text_step("done"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    sent = call_messages(adapter.requests[-1])
    assert [message.reasoning() for message in sent] == [(ReasoningPart(text="R1", duration_ms=0),), (), (ReasoningPart(text="R3", duration_ms=0),)]
    assert [round_.get("reasoning") for round_ in outcome(events).transcript] == ["R1", None, "R3"]


async def test_reasoning_of_an_abandoned_empty_attempt_is_not_carried():
    adapter = ScriptedAdapter(
        [ReasoningDelta(block=0, text="lost"), Finish(reason=FinishReason.STOP)],
        tool_step({"id": "c1", "name": "echo", "args": {}}, reasoning="kept"),
        text_step("done"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert [message.reasoning() for message in call_messages(adapter.requests[-1])] == [(ReasoningPart(text="kept", duration_ms=0),)]
    assert [event.text for event in events if isinstance(event, ReasoningDelta)] == ["lost", "kept"]


async def test_a_pausing_call_keeps_its_reasoning_in_the_transcript():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "gate", "args": {}}, reasoning="why"))
    rig = scripted_engine(adapter, [GateTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome(events).pending
    assert outcome(events).transcript[0]["reasoning"] == "why"


async def test_reasoning_streams_before_the_answer_and_never_into_it():
    adapter = ScriptedAdapter(text_step("the answer", reasoning="pondering"))
    rig = scripted_engine(adapter)

    events = await drain(rig.engine.run(turn(tool_context=None)))

    kinds = [type(event).__name__ for event in events if isinstance(event, (ReasoningDelta, TextDelta))]
    assert kinds == ["ReasoningDelta", "TextDelta"]
    assert outcome(events).content == "the answer"


async def test_the_wrap_up_call_reports_its_reasoning_too():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {}}), text_step("final", reasoning="wrap"))
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn(max_rounds=1)))

    assert "wrap" in [event.text for event in events if isinstance(event, ReasoningDelta)]
    assert outcome(events).reason == "budget"


async def test_only_a_model_written_answer_carries_its_completion():
    answer = await drain(scripted_engine(ScriptedAdapter(text_step("ok"))).engine.run(turn(tool_context=None)))
    truncated = [TextDelta(block=1, text='<tool_call>{"name": "echo", "arg'), Finish(reason=FinishReason.LENGTH, raw_reason="length")]
    rescue = await drain(scripted_engine(ScriptedAdapter(truncated, truncated, truncated), [EchoTool()]).engine.run(turn()))

    assert outcome(answer).completion == {"reason": "stop", "raw": "stop"}
    assert outcome(rescue).reason == "truncated" and outcome(rescue).completion is None


class ParkingAdapter(ScriptedAdapter):
    def __init__(self, *steps: Any, park_on: int) -> None:
        super().__init__(*steps)
        self.park_on = park_on
        self.markers: List[str] = []
        self.entered = asyncio.Event()

    async def send(self, wire: WireRequest, request: Any) -> AsyncIterator[bytes]:
        if len(self.requests) == self.park_on:
            self.entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.markers.append("provider_unwound")
        yield b""


async def test_cancelling_the_turn_unwinds_the_parked_provider_and_records_it():
    echo = EchoTool()
    adapter = ParkingAdapter(tool_step({"id": "c1", "name": "echo", "args": {}}), text_step("never"), park_on=2)
    rig = scripted_engine(adapter, [echo])

    task = asyncio.create_task(drain(rig.engine.run(turn())))
    await asyncio.wait_for(adapter.entered.wait(), timeout=5)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert adapter.markers == ["provider_unwound"]
    assert echo.calls == 1
    turn_id = rig.store.steps("s1")[0].turn_id
    assert rig.store.load_state(turn_id).status.value == "cancelled"


async def test_closing_the_turn_after_a_tool_start_runs_nothing_more():
    echo = EchoTool()
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {}}, {"id": "c2", "name": "echo", "args": {}}),
        text_step("never"),
    )
    rig = scripted_engine(adapter, [echo])
    events = rig.engine.run(turn())

    async for event in events:
        if isinstance(event, ToolStarted):
            break
    await events.aclose()

    assert echo.calls == 0
    assert len(adapter.requests) == 1


def replayed_history():
    return (
        user_message("earlier", message_id="h1"),
        Message(
            id="h2",
            role=Role.ASSISTANT,
            parts=(ReasoningPart(text="old thought", opaque={"signature": "sig"}), TextPart(text="old answer")),
            origin=SCRIPTED_ORIGIN,
            raw={"stored": "reply"},
        ),
    )


async def test_a_refused_replay_is_retried_once_without_raw_or_reasoning_and_says_so():
    adapter = ScriptedAdapter(OpaqueViolation("the stored reply no longer matches"), text_step("fresh"))
    rig = scripted_engine(adapter)

    events = await drain(rig.engine.run(turn(history=replayed_history(), tool_context=None)))

    assert adapter.requests[0].messages[1].raw == {"stored": "reply"}
    retried = adapter.requests[1].messages
    assert [message.id for message in retried] == ["h1", "h2", "u1"]
    assert all(message.raw is None and not message.reasoning() for message in retried)
    assert retried[1].text() == "old answer"
    raised = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in raised] == [NoticeKind.REASONING_DROPPED]
    assert raised[0].detail == {"error": "opaque_violation"}
    assert outcome(events).content == "fresh"
    assert outcome(events).state.notices == tuple(raised)


async def test_a_refusal_after_the_degraded_retry_fails_the_turn_without_looping():
    adapter = ScriptedAdapter(OpaqueViolation("first"), OpaqueViolation("again"), text_step("never sent"))
    rig = scripted_engine(adapter)

    with pytest.raises(OpaqueViolation, match="again"):
        await drain(rig.engine.run(turn(history=replayed_history(), tool_context=None)))

    assert len(adapter.requests) == 2


async def test_a_reasoning_block_is_timed_from_its_first_delta_until_the_next_block_starts():
    ticks = iter([10.0, 12.5])
    adapter = ScriptedAdapter(text_step("answer", reasoning="pondering"))
    rig = scripted_engine(adapter, clock=lambda: next(ticks))

    await drain(rig.engine.run(turn(tool_context=None)))

    (answer,) = [step.message for step in rig.store.steps("s1") if step.message.role is Role.ASSISTANT]
    assert answer.reasoning() == (ReasoningPart(text="pondering", duration_ms=2500),)


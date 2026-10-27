import asyncio
from dataclasses import replace
from typing import Any

import pytest

from src.features.llm.harness.events import Finish, FinishReason, TextDelta, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.messages import ContextPart, Role
from src.features.llm.harness.recovery import CANCELLED_REASON
from src.features.llm.harness.turn_state import Decision, StepCommitted, StepDiscarded, StepStatus, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from tests.features.llm.harness.test_engine_durable import (
    EchoTool,
    Rig,
    database,
    every_call_has_a_result,
    outcome_of,
    turn,
)
from tests.fixtures.llm_engine import drain
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


class ApprovalTool(BaseTool):
    name = "delete_prompt"
    description = "Deletes a prompt after approval."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}
    requires_approval = True

    def __init__(self) -> None:
        self.confirmed = []

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Delete", target=kwargs.get("id")))

    async def execute_confirmed(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.confirmed.append(kwargs.get("id"))
        return ToolResult(success=True, data=f"deleted {kwargs.get('id')}")


class HangingApproval(ApprovalTool):
    name = "hang_approval"

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def execute_confirmed(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.started.set()
        await asyncio.Event().wait()
        return ToolResult(success=True, data="")


def resumed(waiting, history, decisions, **overrides: Any):
    return turn(turn_id=waiting.turn_id, history=history, new_messages=(), resume=tuple(decisions), **overrides)


async def paused_rig(database, *calls, tools=None, more=(), nudge=None):
    approval = ApprovalTool()
    adapter = ScriptedAdapter(tool_step(*calls), *more)
    rig = Rig(adapter, tools or [EchoTool(), approval])
    outcome = outcome_of(await drain(rig.engine.run(turn(iteration_nudge=nudge))))
    assert outcome.pending
    return rig, adapter, approval, outcome.state


def stored_messages(rig):
    return tuple(step.message for step in rig.fresh().steps("s1"))


async def test_a_decided_pause_resumes_the_same_turn_with_one_tool_step_in_call_order(database):
    rig, adapter, approval, waiting = await paused_rig(
        database,
        {"id": "c1", "name": "echo", "args": {"message": "a"}},
        {"id": "c2", "name": "delete_prompt", "args": {"id": "p1"}},
        {"id": "c3", "name": "echo", "args": {"message": "b"}},
        more=[text_step("Deleted it.")],
    )

    outcome = outcome_of(await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c2", True)]))))

    stored = rig.fresh().steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert [step.seq for step in stored] == [0, 1, 2, 3]
    assert {step.turn_id for step in stored} == {waiting.turn_id}
    results = stored[2].message.tool_results()
    assert [(item.call_id, item.text()) for item in results] == [
        ("c1", "echo: a"), ("c2", "deleted p1"),
        ("c3", "Not run: waiting for approval of delete_prompt. Call it again if still needed."),
    ]
    assert approval.confirmed == ["p1"]
    assert outcome.reason == "answer" and outcome.rounds_used == 2
    final = rig.fresh().load_state(waiting.turn_id)
    assert final.status is TurnStatus.COMPLETED and final.pending == () and final.gathered == ()
    assert [message.role for message in adapter.requests[1].messages] == [Role.USER, Role.ASSISTANT, Role.TOOL]


async def test_a_rejection_becomes_the_calls_result(database):
    rig, adapter, approval, waiting = await paused_rig(
        database, {"id": "c2", "name": "delete_prompt", "args": {"id": "p1"}}, more=[text_step("Fine.")],
    )

    await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c2", False, "keep it")])))

    (tool_step_,) = [step for step in rig.fresh().steps("s1") if step.message.role is Role.TOOL]
    assert tool_step_.message.tool_results()[0].text() == "User rejected this. Reason: keep it"
    assert approval.confirmed == []


async def test_deciding_only_some_calls_pauses_again_without_a_tool_step(database):
    rig, _adapter, approval, waiting = await paused_rig(
        database,
        {"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}},
        {"id": "c2", "name": "delete_prompt", "args": {"id": "p2"}},
        more=[text_step("Both gone.")],
    )

    again = outcome_of(await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c1", True)]))))

    assert again.pending
    assert [step.message.role for step in rig.fresh().steps("s1")] == [Role.USER, Role.ASSISTANT]
    paused = rig.fresh().open_turn("s1")
    assert [call.call_id for call in paused.pending] == ["c2"]
    assert [(part.call_id, part.text()) for part in paused.gathered] == [("c1", "deleted p1")]

    done = outcome_of(await drain(rig.engine.run(resumed(paused, stored_messages(rig), [Decision("c2", True)]))))

    (tool_step_,) = [step for step in rig.fresh().steps("s1") if step.message.role is Role.TOOL]
    assert [item.text() for item in tool_step_.message.tool_results()] == ["deleted p1", "deleted p2"]
    assert done.reason == "answer"
    assert approval.confirmed == ["p1", "p2"]


async def test_a_turn_that_is_not_waiting_cannot_be_resumed_and_is_left_alone(database):
    rig = Rig(ScriptedAdapter(text_step("Done.")), [])
    finished = outcome_of(await drain(rig.engine.run(turn(tool_context=None)))).state
    before = rig.fresh().steps("s1")

    with pytest.raises(ValueError):
        await drain(rig.engine.run(resumed(finished, stored_messages(rig), [], tool_context=None)))

    assert rig.fresh().load_state(finished.turn_id).status is TurnStatus.COMPLETED
    assert rig.fresh().steps("s1") == before


async def test_a_decision_for_a_call_that_was_never_pending_fails_the_resume(database):
    rig, _adapter, _approval, waiting = await paused_rig(
        database, {"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}},
    )

    with pytest.raises(ValueError):
        await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("nope", True)])))

    stored = rig.fresh().steps("s1")
    assert every_call_has_a_result(stored)
    assert rig.fresh().load_state(waiting.turn_id).status is TurnStatus.FAILED


async def test_cancelling_a_resumed_decision_closes_every_open_call_and_keeps_gathered_results(database):
    hanging = HangingApproval()
    rig, _adapter, _approval, waiting = await paused_rig(
        database,
        {"id": "c1", "name": "echo", "args": {"message": "kept"}},
        {"id": "c2", "name": "hang_approval", "args": {}},
        tools=[EchoTool(), hanging],
    )
    history = stored_messages(rig)

    task = asyncio.create_task(drain(rig.engine.run(resumed(waiting, history, [Decision("c2", True)]))))
    await asyncio.wait_for(hanging.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stored = rig.fresh().steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert stored[2].status is StepStatus.CANCELLED
    results = stored[2].message.tool_results()
    assert [(item.call_id, item.is_error) for item in results] == [("c1", False), ("c2", True)]
    assert results[0].text() == "echo: kept" and results[1].text() == CANCELLED_REASON
    assert rig.fresh().load_state(waiting.turn_id).status is TurnStatus.CANCELLED
    assert every_call_has_a_result(stored)


async def test_a_single_call_can_be_settled_without_running_the_turn(database):
    rig, _adapter, approval, waiting = await paused_rig(
        database, {"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}},
    )
    call = rig.fresh().steps("s1")[1].message.tool_calls()[0]

    finished = await rig.engine.settle(waiting, call, Decision("c1", True), ToolContext(user_id="user-1"))

    assert finished.result.text() == "deleted p1" and not finished.result.is_error
    assert finished.record.approval_status == "approved"
    assert approval.confirmed == ["p1"]
    assert [step.message.role for step in rig.fresh().steps("s1")] == [Role.USER, Role.ASSISTANT]


async def test_a_resumed_turn_keeps_the_prefix_change_it_recorded_before_the_pause(database):
    rig, _adapter, _approval, waiting = await paused_rig(
        database, {"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}}, more=[text_step("Deleted it.")],
    )
    rig.store.save_state(replace(waiting, prefix_changed_at=0))

    await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c1", True)])))

    assert rig.fresh().load_state(waiting.turn_id).prefix_changed_at == 0


class FailingTool(BaseTool):
    name = "lookup"
    description = "Always fails."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=False, data="", error="index offline")


async def test_a_resumed_turn_still_counts_and_remembers_the_failures_from_before_the_pause(database):
    rig, _adapter, _approval, waiting = await paused_rig(
        database,
        {"id": "c1", "name": "lookup", "args": {}},
        {"id": "c2", "name": "delete_prompt", "args": {"id": "p1"}},
        tools=[FailingTool(), ApprovalTool()],
        more=[text_step("Done.")],
    )
    assert waiting.tool_failures == {"lookup": 1}

    outcome = outcome_of(await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c2", True)]))))

    assert outcome.tool_failures == {"lookup": 1}
    assert outcome.state.failed_calls == {"lookup:{}": "index offline"}


class WriteNote(BaseTool):
    name = "write_note"
    description = "Saves a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="saved")


async def test_a_step_retried_after_a_resume_is_discarded_at_the_seq_its_retry_takes(database):
    cut_off = [
        TextDelta(block=1, text="Saving it. "),
        ToolCallStart(block=2, call_id="t1", name="write_note"),
        ToolArgsDelta(block=2, text='{"text": "a lo'),
        Finish(reason=FinishReason.LENGTH, raw_reason="length"),
    ]
    rig, _adapter, _approval, waiting = await paused_rig(
        database,
        {"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}},
        tools=[WriteNote(), ApprovalTool()],
        more=[cut_off, tool_step({"id": "t1", "name": "write_note", "args": {"text": "a long note"}}), text_step("Saved.")],
    )
    before = {step.seq for step in rig.fresh().steps("s1")}

    events = await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c1", True)])))

    (discarded,) = [event for event in events if isinstance(event, StepDiscarded)]
    retried = next(
        event.step for event in events[events.index(discarded):]
        if isinstance(event, StepCommitted) and event.step.message.role is Role.ASSISTANT
    )
    assert discarded.seq == retried.seq
    assert discarded.seq not in before


async def test_a_resumed_turn_that_was_already_nudged_is_not_nudged_again(database):
    rig, _adapter, _approval, waiting = await paused_rig(
        database,
        {"id": "c0", "name": "echo", "args": {"message": "a"}},
        tools=[EchoTool(), ApprovalTool()],
        more=[
            tool_step({"id": "c1", "name": "delete_prompt", "args": {"id": "p1"}}),
            tool_step({"id": "c2", "name": "echo", "args": {"message": "b"}}),
            text_step("Done."),
        ],
        nudge="keep going",
    )

    await drain(rig.engine.run(resumed(waiting, stored_messages(rig), [Decision("c1", True)], iteration_nudge="keep going")))

    nudges = [
        part for step in rig.fresh().steps("s1") for part in step.message.parts
        if isinstance(part, ContextPart) and part.kind == "nudge"
    ]
    assert len(nudges) == 1

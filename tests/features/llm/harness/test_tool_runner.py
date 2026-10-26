import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from src.features.llm.harness.engine import ToolRound
from src.features.llm.harness.events import Finish, FinishReason, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.messages import ImagePart, Role, ToolCallPart, ToolResultPart, message
from src.features.llm.harness.tool_runner import IncompleteBatch, ParallelToolRunner, batch_results
from src.features.llm.harness.turn_state import (
    Decision,
    DecisionKind,
    PendingCall,
    ToolFinished,
    ToolStarted,
    TurnStatus,
)
from src.features.llm.tools.base import ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from src.features.llm.tools.workflow import ToolCallGuard
from tests.features.llm.harness.probes import Probe, Shared
from tests.features.llm.harness.stepping import play_step, request_for, turn_state
from tests.fixtures.llm_harness import ScriptedAdapter, scripted_profile, text_step, tool_step

WAIT = 3


class Rig:
    def __init__(self, *names: str, shared: Optional[Shared] = None, **runner_options: Any) -> None:
        self.log: List[str] = []
        self.shared = shared or Shared()
        registry = ToolRegistry()
        for name in names:
            registry.register(Probe(
                name, self.log, read_only=name.startswith("read_"), gated=name.startswith("gate_"), shared=self.shared,
            ))
        self.executor = ToolExecutor(tool_registry=registry, llm_service=None)
        self.runner = ParallelToolRunner(self.executor, **runner_options)
        self.round = ToolRound(
            tool_context=ToolContext(user_id="user-1"), allowed_tools=None, deferred_tools={}, guard=ToolCallGuard(),
        )

    async def collect(self, calls, state=None, decisions=()) -> list:
        async def drain() -> list:
            return [event async for event in self.runner.run(calls, state or turn_state(), self.round, decisions)]

        return await asyncio.wait_for(drain(), WAIT)


def call(call_id: str, name: str, args: Optional[Dict[str, Any]] = None) -> ToolCallPart:
    return ToolCallPart(call_id=call_id, name=name, args={} if args is None else args)


def shape(events) -> List[str]:
    labels = []
    for event in events:
        if isinstance(event, ToolStarted):
            labels.append(f"started:{event.call.call_id}")
        elif isinstance(event, ToolFinished):
            labels.append(f"finished:{event.result.call_id}")
        else:
            labels.append(f"pending:{event.call_id}")
    return labels


def finished(events) -> List[ToolResultPart]:
    return [event.result for event in events if isinstance(event, ToolFinished)]


def paused(events, base=None):
    base = base or turn_state()
    pending = tuple(event for event in events if isinstance(event, PendingCall))
    waiting = {item.call_id for item in pending}
    return replace(
        base,
        status=TurnStatus.AWAITING_DECISION,
        pending=pending,
        gathered=base.gathered + tuple(result for result in finished(events) if result.call_id not in waiting),
    )


async def test_read_only_calls_run_concurrently():
    shared = Shared()
    shared.barrier = asyncio.Barrier(2)
    rig = Rig("read_a", "read_b", shared=shared)
    events = await rig.collect([call("a", "read_a"), call("b", "read_b")])
    assert shape(events) == ["started:a", "started:b", "finished:a", "finished:b"]
    assert shared.peak == 2


async def test_results_follow_emission_order_when_the_first_call_finishes_last():
    shared = Shared()
    shared.hold = {"read_a": asyncio.Event()}
    shared.entered = {"read_b": asyncio.Event()}
    rig = Rig("read_a", "read_b", shared=shared)

    async def release() -> None:
        await shared.entered["read_b"].wait()
        await asyncio.sleep(0)
        shared.hold["read_a"].set()

    releaser = asyncio.ensure_future(release())
    events = await rig.collect([call("a", "read_a"), call("b", "read_b")])
    await releaser
    assert [result.call_id for result in finished(events)] == ["a", "b"]
    assert rig.log.index("end:read_b") < rig.log.index("end:read_a")
    assert [execution.tool_call_id for execution in rig.round.executions] == ["a", "b"]


async def test_a_tool_not_marked_read_only_runs_alone_whatever_its_name():
    rig = Rig("write_a", "write_b", "write_c")
    events = await rig.collect([call("a", "write_a"), call("b", "write_b"), call("c", "write_c")])
    assert rig.log == [
        "start:write_a", "end:write_a", "start:write_b", "end:write_b", "start:write_c", "end:write_c",
    ]
    assert shape(events) == ["started:a", "finished:a", "started:b", "finished:b", "started:c", "finished:c"]
    assert rig.shared.peak == 1


async def test_a_mutating_call_separates_the_read_only_calls_around_it():
    rig = Rig("read_a", "write_w", "read_c", "read_d")
    await rig.collect([call("a", "read_a"), call("w", "write_w"), call("c", "read_c"), call("d", "read_d")])
    assert rig.log.index("end:read_a") < rig.log.index("start:write_w")
    assert rig.log.index("end:write_w") < rig.log.index("start:read_c")
    assert rig.log.index("end:write_w") < rig.log.index("start:read_d")
    assert rig.shared.peak == 2


async def test_concurrency_is_bounded():
    rig = Rig("read_x", max_concurrency=2)
    await rig.collect([call(f"r{index}", "read_x") for index in range(5)])
    assert rig.shared.peak == 2
    assert len(rig.log) == 10


async def test_an_unknown_tool_gets_an_error_result_and_the_batch_goes_on():
    rig = Rig("read_a")
    events = await rig.collect([call("x", "ghost"), call("a", "read_a")])
    first, second = finished(events)
    assert first.call_id == "x" and first.is_error and "not found" in first.text()
    assert second.text() == "read_a done"


async def test_a_call_with_unparseable_arguments_is_never_executed_and_still_recorded():
    rig = Rig("read_a", "write_x")
    broken = ToolCallPart(call_id="x", name="write_x", args=None, args_raw='{"query": "cat')
    events = await rig.collect([broken, call("a", "read_a")])
    assert rig.log == ["start:read_a", "end:read_a"]
    outcome = finished(events)[0]
    assert outcome.call_id == "x" and outcome.is_error
    assert "cut off" in outcome.text() and "Send the call again" in outcome.text()
    assert [execution.tool_call_id for execution in rig.round.executions] == ["x", "a"]
    assert rig.round.executions[0].result.success is False
    assert "cut off" in rig.round.executions[0].result.error


async def test_a_call_cut_off_by_the_output_limit_never_reaches_a_tool():
    adapter = ScriptedAdapter(
        [
            ToolCallStart(block=0, call_id="t1", name="write_note"),
            ToolArgsDelta(block=0, text='{"text": "a long no'),
            Finish(reason=FinishReason.LENGTH, raw_reason="length"),
        ],
    )
    reply, _ = await play_step(adapter, scripted_profile(), request_for())
    (cut,) = reply.tool_calls()
    assert cut.args is None
    rig = Rig("write_note")
    events = await rig.collect(reply.tool_calls())
    assert rig.log == []
    assert finished(events)[0].is_error


async def test_the_approval_batch_returns_every_result_together_in_emission_order():
    adapter = ScriptedAdapter(
        tool_step(
            {"id": "a", "name": "read_a"},
            {"id": "b", "name": "gate_b"},
            {"id": "c", "name": "read_c"},
            {"id": "d", "name": "write_d"},
        ),
        text_step("all done"),
    )
    profile = scripted_profile()
    reply, _ = await play_step(adapter, profile, request_for())
    calls = reply.tool_calls()
    rig = Rig("read_a", "gate_b", "read_c", "write_d")

    first = await rig.collect(calls)
    assert shape(first) == [
        "started:a", "started:b", "started:c", "finished:a", "finished:b", "pending:b", "finished:c", "finished:d",
    ]
    assert "start:write_d" not in rig.log
    waiting = paused(first)
    assert [item.call_id for item in waiting.pending] == ["b"]
    assert waiting.pending[0].preview["action"] == "Do"
    assert [result.call_id for result in waiting.gathered] == ["a", "c", "d"]
    with pytest.raises(IncompleteBatch):
        batch_results(calls, waiting.gathered)
    assert len(adapter.requests) == 1

    second = await rig.collect(calls, waiting, [Decision("b", True)])
    assert shape(second) == ["started:b", "finished:b"]
    assert rig.log.count("start:gate_b") == 1 and rig.log.count("confirm:gate_b") == 1

    ordered = batch_results(calls, waiting.gathered + tuple(finished(second)))
    assert [result.call_id for result in ordered] == ["a", "b", "c", "d"]
    assert [result.text() for result in ordered] == [
        "read_a done",
        "gate_b applied",
        "read_c done",
        "Not run: waiting for approval of gate_b. Call it again if still needed.",
    ]
    assert [result.is_error for result in ordered] == [False, False, False, True]

    await play_step(adapter, profile, request_for(messages=[reply, message("tool_1", Role.TOOL, *ordered)]), message_id="msg_2")
    sent = adapter.requests[1].messages[-1]
    assert [result.call_id for result in sent.tool_results()] == ["a", "b", "c", "d"]
    assert adapter.exhausted


async def test_the_records_say_what_happened_to_every_call_in_emission_order():
    rig = Rig("read_a", "gate_b", "write_d")
    first = await rig.collect([call("a", "read_a"), call("b", "gate_b"), call("d", "write_d")])
    records = [event.record for event in first if isinstance(event, ToolFinished)]
    assert [(item.tool_call_id, item.approval_status, item.skipped) for item in rig.round.executions] == [
        ("a", None, False), ("b", "pending", False), ("d", None, True),
    ]
    assert records == rig.round.executions
    assert rig.round.executions[1].result.preview.target == "gate_b"
    assert first[[type(item).__name__ for item in first].index("PendingCall")].decision is DecisionKind.APPROVAL

    waiting = paused(first)
    second = await rig.collect([call("a", "read_a"), call("b", "gate_b"), call("d", "write_d")], waiting, [Decision("b", True)])
    assert second[-1].record.approval_status == "approved"
    assert second[-1].record.tool_call_id == "b"


async def test_a_denied_call_becomes_an_error_result_with_the_reason():
    rig = Rig("gate_b")
    calls = [call("b", "gate_b")]
    waiting = paused(await rig.collect(calls))
    events = await rig.collect(calls, waiting, [Decision("b", False, "not this one")])
    (denied,) = finished(events)
    assert denied.is_error
    assert denied.text() == "User rejected this. Reason: not this one"
    assert "confirm:gate_b" not in rig.log
    assert shape(events) == ["finished:b"]
    assert events[0].record.approval_status == "rejected"


async def test_a_denial_without_a_reason_still_reads_as_a_rejection():
    rig = Rig("gate_b")
    calls = [call("b", "gate_b")]
    waiting = paused(await rig.collect(calls))
    events = await rig.collect(calls, waiting, [Decision("b", False)])
    assert finished(events)[0].text() == "User rejected this."


async def test_a_confirmed_call_that_fails_is_recorded_as_failed():
    rig = Rig("gate_b")
    calls = [call("b", "gate_b")]
    waiting = paused(await rig.collect(calls))
    rig.executor.execute_tool_confirmed = AsyncMock(return_value=ToolResult(success=False, data="", error="nope"))
    events = await rig.collect(calls, waiting, [Decision("b", True)])
    assert finished(events)[0].is_error and finished(events)[0].text() == "Error: nope"
    assert events[-1].record.approval_status == "failed"


async def test_undecided_approvals_stay_pending_while_a_decided_one_runs():
    rig = Rig("gate_1", "gate_2", "write_w")
    calls = [call("g1", "gate_1"), call("g2", "gate_2"), call("w", "write_w")]
    first = await rig.collect(calls)
    waiting = paused(first)
    assert [item.call_id for item in waiting.pending] == ["g1", "g2"]
    assert [result.call_id for result in waiting.gathered] == ["w"]

    partial = await rig.collect(calls, waiting, [Decision("g1", True)])
    assert shape(partial) == ["started:g1", "finished:g1", "pending:g2"]
    assert partial[-1] is waiting.pending[1]

    rest = paused(partial, replace(waiting, gathered=waiting.gathered))
    final = await rig.collect(calls, rest, [Decision("g2", True)])
    assert shape(final) == ["started:g2", "finished:g2"]


async def test_a_write_before_an_approval_runs_immediately():
    rig = Rig("write_w", "gate_b")
    events = await rig.collect([call("w", "write_w"), call("b", "gate_b")])
    assert shape(events) == ["started:w", "finished:w", "started:b", "finished:b", "pending:b"]
    assert "end:write_w" in rig.log


async def test_a_decision_for_a_call_that_is_not_pending_is_refused():
    rig = Rig("read_a")
    with pytest.raises(ValueError, match="ghost"):
        await rig.collect([call("a", "read_a")], decisions=[Decision("ghost", True)])


async def test_calls_waiting_on_an_outside_decision_are_not_executed():
    rig = Rig("read_x", "read_a")
    outside = PendingCall("x", "read_x", DecisionKind.EXTERNAL)
    state = turn_state(pending=(outside,))
    events = await rig.collect([call("x", "read_x"), call("a", "read_a")], state)
    assert events[0] is outside
    assert rig.log == ["start:read_a", "end:read_a"]
    with pytest.raises(ValueError):
        await rig.collect([call("x", "read_x")], state, [Decision("x", True)])


async def test_a_failure_in_the_executor_becomes_an_error_result_and_the_batch_goes_on():
    rig = Rig("read_a", "read_b")
    real = rig.executor.execute_tool_guarded

    async def flaky(name, *args, **kwargs):
        if name == "read_a":
            raise RuntimeError("boom")
        return await real(name, *args, **kwargs)

    rig.executor.execute_tool_guarded = flaky
    events = await rig.collect([call("a", "read_a"), call("b", "read_b")])
    first, second = finished(events)
    assert first.is_error and "failed unexpectedly" in first.text()
    assert not second.is_error


async def test_a_tool_image_travels_with_its_result():
    registry = ToolRegistry()
    registry.register(Probe("read_snap", [], read_only=True, image="aW1n"))
    executor = ToolExecutor(tool_registry=registry, llm_service=None)
    round_ = ToolRound(tool_context=ToolContext(user_id="u"), allowed_tools=None, deferred_tools={}, guard=ToolCallGuard())
    events = [event async for event in ParallelToolRunner(executor).run([call("s", "read_snap")], turn_state(), round_)]
    (result,) = finished(events)
    assert [type(item).__name__ for item in result.content] == ["TextPart", "ImagePart"]
    assert isinstance(result.content[1], ImagePart) and result.content[1].data == "aW1n"


async def test_closing_the_run_cancels_every_read_still_in_flight():
    shared = Shared()
    shared.hold = {"read_a": asyncio.Event(), "read_b": asyncio.Event()}
    shared.entered = {"read_a": asyncio.Event(), "read_b": asyncio.Event()}
    rig = Rig("read_a", "read_b", shared=shared)

    async def drain() -> None:
        async for _ in rig.runner.run([call("a", "read_a"), call("b", "read_b")], turn_state(), rig.round):
            pass

    task = asyncio.ensure_future(drain())
    await asyncio.wait_for(shared.entered["read_a"].wait(), WAIT)
    await asyncio.wait_for(shared.entered["read_b"].wait(), WAIT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, WAIT)
    assert sorted(shared.cancelled) == ["read_a", "read_b"]


def test_batch_results_orders_by_the_calls_and_names_what_is_missing():
    calls = [call("a", "read_a"), call("b", "read_b")]
    results = [ToolResultPart(call_id="b", name="read_b", content=()), ToolResultPart(call_id="a", name="read_a", content=())]
    assert [result.call_id for result in batch_results(calls, results)] == ["a", "b"]
    with pytest.raises(IncompleteBatch, match="b"):
        batch_results(calls, results[1:])


def test_the_runner_rejects_a_concurrency_limit_below_one():
    with pytest.raises(ValueError):
        ParallelToolRunner(ToolExecutor(tool_registry=ToolRegistry(), llm_service=None), max_concurrency=0)

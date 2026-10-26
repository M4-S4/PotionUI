import asyncio
from dataclasses import replace
from typing import Any, Dict, List, Optional

import pytest

from src.features.llm.harness.events import Finish, FinishReason, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.messages import Role, TextPart, ToolCallPart, ToolResultPart, message
from src.features.llm.harness.tool_runner import (
    IncompleteBatch,
    ParallelToolRunner,
    ToolOutcome,
    batch_results,
    skipped_results,
)
from src.features.llm.harness.turn_state import (
    Decision,
    DecisionKind,
    PendingCall,
    ToolFinished,
    ToolStarted,
    TurnStatus,
)
from tests.features.llm.harness.stepping import play_step, request_for, turn_state
from tests.fixtures.llm_harness import ScriptedAdapter, scripted_profile, text_step, tool_step

WAIT = 3


def result_for(call: ToolCallPart, text: str, is_error: bool = False) -> ToolResultPart:
    return ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart(text),), is_error=is_error)


def call(call_id: str, name: str, args: Optional[Dict[str, Any]] = None) -> ToolCallPart:
    return ToolCallPart(call_id=call_id, name=name, args={} if args is None else args)


class Backend:
    def __init__(self) -> None:
        self.log: List[str] = []
        self.hold: Dict[str, asyncio.Event] = {}
        self.entered: Dict[str, asyncio.Event] = {}
        self.fail: set = set()
        self.barrier: Optional[asyncio.Barrier] = None
        self.running = 0
        self.peak = 0
        self.cancelled: List[str] = []

    def parallel_safe(self, name: str) -> bool:
        return name.startswith(("read_", "gate_"))

    async def execute(self, tool_call: ToolCallPart, context: Any, state: Any) -> ToolOutcome:
        self.log.append(f"start:{tool_call.call_id}")
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            if tool_call.call_id in self.entered:
                self.entered[tool_call.call_id].set()
            if tool_call.call_id in self.fail:
                raise RuntimeError("boom")
            if self.barrier is not None and tool_call.name.startswith("read_"):
                await self.barrier.wait()
            if tool_call.call_id in self.hold:
                await self.hold[tool_call.call_id].wait()
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(tool_call.call_id)
            raise
        finally:
            self.running -= 1
        self.log.append(f"end:{tool_call.call_id}")
        if tool_call.name.startswith("gate_"):
            return ToolOutcome(result_for(tool_call, "preview"), pending=True, preview={"for": tool_call.call_id})
        return ToolOutcome(result_for(tool_call, f"{tool_call.name} done"), record=f"record:{tool_call.call_id}")

    async def confirm(self, tool_call: ToolCallPart, context: Any, state: Any) -> ToolOutcome:
        self.log.append(f"confirm:{tool_call.call_id}")
        await asyncio.sleep(0)
        return ToolOutcome(result_for(tool_call, f"{tool_call.name} applied"), record=f"applied:{tool_call.call_id}")


async def collect(runner, calls, state, decisions=()) -> list:
    async def drain() -> list:
        return [event async for event in runner.run(calls, state, None, decisions)]

    return await asyncio.wait_for(drain(), WAIT)


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


def pending_state(events, base=None):
    base = base or turn_state()
    return replace(
        base,
        status=TurnStatus.AWAITING_DECISION,
        pending=tuple(event for event in events if isinstance(event, PendingCall)),
        gathered=base.gathered + tuple(finished(events)),
    )


async def test_read_only_calls_run_concurrently():
    backend = Backend()
    backend.barrier = asyncio.Barrier(2)
    events = await collect(
        ParallelToolRunner(backend), [call("a", "read_a"), call("b", "read_b")], turn_state(),
    )
    assert shape(events) == ["started:a", "started:b", "finished:a", "finished:b"]
    assert backend.peak == 2


async def test_results_follow_emission_order_when_the_first_call_finishes_last():
    backend = Backend()
    backend.hold = {"a": asyncio.Event()}
    backend.entered = {"b": asyncio.Event()}

    async def release() -> None:
        await backend.entered["b"].wait()
        await asyncio.sleep(0)
        backend.hold["a"].set()

    releaser = asyncio.ensure_future(release())
    events = await collect(ParallelToolRunner(backend), [call("a", "read_a"), call("b", "read_b")], turn_state())
    await releaser
    assert [result.call_id for result in finished(events)] == ["a", "b"]
    assert backend.log.index("end:b") < backend.log.index("end:a")


async def test_mutating_calls_run_one_at_a_time_in_emission_order():
    backend = Backend()
    events = await collect(
        ParallelToolRunner(backend),
        [call("a", "write_a"), call("b", "write_b"), call("c", "write_c")],
        turn_state(),
    )
    assert backend.log == ["start:a", "end:a", "start:b", "end:b", "start:c", "end:c"]
    assert shape(events) == [
        "started:a", "finished:a", "started:b", "finished:b", "started:c", "finished:c",
    ]
    assert backend.peak == 1


async def test_a_mutating_call_separates_the_read_only_calls_around_it():
    backend = Backend()
    await collect(
        ParallelToolRunner(backend),
        [call("a", "read_a"), call("w", "write_w"), call("c", "read_c"), call("d", "read_d")],
        turn_state(),
    )
    assert backend.log.index("end:a") < backend.log.index("start:w")
    assert backend.log.index("end:w") < backend.log.index("start:c")
    assert backend.log.index("end:w") < backend.log.index("start:d")
    assert backend.peak == 2


async def test_concurrency_is_bounded():
    backend = Backend()
    calls = [call(f"r{i}", "read_x") for i in range(5)]
    await collect(ParallelToolRunner(backend, max_concurrency=2), calls, turn_state())
    assert backend.peak == 2
    assert len(backend.log) == 10


async def test_a_call_with_unparseable_arguments_is_never_executed():
    backend = Backend()
    broken = ToolCallPart(call_id="x", name="write_x", args=None, args_raw='{"query": "cat')
    events = await collect(ParallelToolRunner(backend), [broken, call("a", "read_a")], turn_state())
    assert backend.log == ["start:a", "end:a"]
    outcome = finished(events)[0]
    assert outcome.call_id == "x" and outcome.is_error
    assert "cut off" in outcome.text() and "Send the call again" in outcome.text()
    assert shape(events)[:2] == ["started:x", "started:a"]


async def test_a_call_cut_off_by_the_output_limit_never_reaches_the_backend():
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
    backend = Backend()
    events = await collect(ParallelToolRunner(backend), reply.tool_calls(), turn_state())
    assert backend.log == []
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
    backend = Backend()
    runner = ParallelToolRunner(backend)

    first = await collect(runner, calls, turn_state())
    assert shape(first) == ["started:a", "started:b", "started:c", "finished:a", "pending:b", "finished:c"]
    assert "start:d" not in backend.log
    paused = pending_state(first)
    assert [item.call_id for item in paused.pending] == ["b"]
    assert paused.pending[0].preview == {"for": "b"}
    with pytest.raises(IncompleteBatch):
        batch_results(calls, paused.gathered)
    assert len(adapter.requests) == 1

    second = await collect(runner, calls, paused, [Decision("b", True)])
    assert shape(second) == ["started:b", "finished:b", "started:d", "finished:d"]
    assert backend.log.index("confirm:b") < backend.log.index("start:d")
    assert backend.log.count("start:b") == 1

    ordered = batch_results(calls, paused.gathered + tuple(finished(second)))
    assert [result.call_id for result in ordered] == ["a", "b", "c", "d"]
    assert [result.text() for result in ordered] == ["read_a done", "gate_b applied", "read_c done", "write_d done"]

    tool_message = message("tool_1", Role.TOOL, *ordered)
    await play_step(adapter, profile, request_for(messages=[reply, tool_message]), message_id="msg_2")
    sent = adapter.requests[1].messages[-1]
    assert [result.call_id for result in sent.tool_results()] == ["a", "b", "c", "d"]
    assert adapter.exhausted


async def test_a_denied_call_becomes_an_error_result_with_the_reason():
    backend = Backend()
    runner = ParallelToolRunner(backend)
    calls = [call("b", "gate_b")]
    paused = pending_state(await collect(runner, calls, turn_state()))
    events = await collect(runner, calls, paused, [Decision("b", False, "not this one")])
    (denied,) = finished(events)
    assert denied.is_error
    assert denied.text() == "User rejected this. Reason: not this one"
    assert "confirm:b" not in backend.log
    assert shape(events) == ["started:b", "finished:b"]


async def test_a_denial_without_a_reason_still_reads_as_a_rejection():
    runner = ParallelToolRunner(Backend())
    calls = [call("b", "gate_b")]
    paused = pending_state(await collect(runner, calls, turn_state()))
    events = await collect(runner, calls, paused, [Decision("b", False)])
    assert finished(events)[0].text() == "User rejected this."


async def test_undecided_approvals_stay_pending_and_hold_back_later_writes():
    backend = Backend()
    runner = ParallelToolRunner(backend)
    calls = [call("b1", "gate_1"), call("b2", "gate_2"), call("w", "write_w")]
    first = await collect(runner, calls, turn_state())
    assert shape(first) == ["started:b1", "started:b2", "pending:b1", "pending:b2"]
    paused = pending_state(first)

    partial = await collect(runner, calls, paused, [Decision("b1", True)])
    assert shape(partial) == ["started:b1", "finished:b1", "pending:b2"]
    assert partial[-1] is paused.pending[1]
    assert "start:w" not in backend.log

    rest = pending_state(partial, paused)
    final = await collect(runner, calls, rest, [Decision("b2", True)])
    assert shape(final) == ["started:b2", "finished:b2", "started:w", "finished:w"]


async def test_a_write_before_an_approval_runs_immediately():
    backend = Backend()
    events = await collect(ParallelToolRunner(backend), [call("w", "write_w"), call("b", "gate_b")], turn_state())
    assert shape(events) == ["started:w", "finished:w", "started:b", "pending:b"]


async def test_a_decision_for_a_call_that_is_not_pending_is_refused():
    runner = ParallelToolRunner(Backend())
    with pytest.raises(ValueError, match="ghost"):
        await collect(runner, [call("a", "read_a")], turn_state(), [Decision("ghost", True)])


async def test_calls_waiting_on_an_outside_decision_are_not_executed():
    backend = Backend()
    outside = PendingCall("x", "read_x", DecisionKind.EXTERNAL)
    events = await collect(
        ParallelToolRunner(backend), [call("x", "read_x"), call("a", "read_a")], turn_state(pending=(outside,)),
    )
    assert events[0] is outside
    assert backend.log == ["start:a", "end:a"]
    with pytest.raises(ValueError):
        await collect(ParallelToolRunner(backend), [call("x", "read_x")], turn_state(pending=(outside,)), [Decision("x", True)])


async def test_a_backend_failure_becomes_an_error_result_and_the_batch_goes_on():
    backend = Backend()
    backend.fail = {"a"}
    events = await collect(ParallelToolRunner(backend), [call("a", "read_a"), call("b", "read_b")], turn_state())
    first, second = finished(events)
    assert first.is_error and "failed unexpectedly" in first.text()
    assert not second.is_error


async def test_closing_the_run_cancels_the_reads_still_in_flight():
    backend = Backend()
    backend.hold = {"a": asyncio.Event()}
    backend.entered = {"a": asyncio.Event()}
    runner = ParallelToolRunner(backend)

    async def drain() -> None:
        async for _ in runner.run([call("a", "read_a")], turn_state(), None):
            pass

    task = asyncio.ensure_future(drain())
    await asyncio.wait_for(backend.entered["a"].wait(), WAIT)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.cancelled == ["a"]


async def test_skipped_calls_get_error_results_so_the_batch_stays_valid():
    calls = [call("a", "read_a"), call("b", "write_b")]
    done = (result_for(calls[0], "ok"),)
    skipped = skipped_results(calls, [result.call_id for result in done], "the turn was cancelled")
    assert [result.call_id for result in skipped] == ["b"]
    assert skipped[0].is_error and skipped[0].text() == "Not run: the turn was cancelled"
    assert [result.call_id for result in batch_results(calls, done + skipped)] == ["a", "b"]


async def test_the_runner_rejects_a_concurrency_limit_below_one():
    with pytest.raises(ValueError):
        ParallelToolRunner(Backend(), max_concurrency=0)

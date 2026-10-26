import asyncio
from typing import Any, Dict, List

import pytest

from src.features.chat import decisions, projection
from src.features.llm.harness import codec
from src.features.llm.harness.build import user_message
from src.features.llm.harness.engine import TurnEngine, TurnInput, TurnOutcome
from src.features.llm.harness.errors import HarnessError, Overloaded
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.messages import Role, TextPart
from src.features.llm.harness.recovery import CANCELLED_REASON, FAILED_REASON
from src.features.llm.harness.retry import BackoffRetry
from src.features.llm.harness.store import SqliteTurnStore
from src.features.llm.harness.tool_runner import ParallelToolRunner
from src.features.llm.harness.turn_state import StepCommitted, StepStatus, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult
from src.features.llm.tools.executor import ToolExecutor
from src.features.llm.tools.registry import ToolRegistry
from tests.fixtures.llm_engine import drain, scripted_config, scripted_gateway, tool_context
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step
from tests.fixtures.persistence_base import PersistenceTestBase


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes a message back."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


class DeleteTool(BaseTool):
    name = "delete_prompt"
    description = "Deletes a prompt after approval."
    parameters = {"type": "object", "properties": {"id": {"type": "string"}}}
    requires_approval = True

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Delete", target=kwargs.get("id")))


class HangingTool(BaseTool):
    name = "hang"
    description = "Never finishes."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.started.set()
        await asyncio.Event().wait()
        return ToolResult(success=True, data="")


class _Database(PersistenceTestBase):
    in_memory = True

    def runTest(self) -> None:
        pass


@pytest.fixture
def database():
    case = _Database()
    case.setUp()
    with case.db.get_cursor() as cursor:
        cursor.execute(
            "INSERT INTO chat_sessions (id, user_id, mode, name, status) VALUES ('s1', 'u1', 'assistant', 'chat', 'active')"
        )
    yield case.db
    case.tearDown()


class Rig:
    def __init__(self, adapter: Any, tools: List[BaseTool], runner: Any = None) -> None:
        gateway = scripted_gateway(adapter, scripted_config())
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        executor = ToolExecutor(tool_registry=registry, llm_service=gateway)
        self.store = SqliteTurnStore()
        self.engine = TurnEngine(
            gateway=gateway,
            tools=executor,
            runner=runner or ParallelToolRunner(executor),
            store=self.store,
            retry=BackoffRetry(max_retries=0),
        )

    def fresh(self) -> SqliteTurnStore:
        return SqliteTurnStore()


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
    (outcome,) = [event for event in events if isinstance(event, TurnOutcome)]
    return outcome


def every_call_has_a_result(steps) -> bool:
    asked = [call.call_id for step in steps if step.message.role is Role.ASSISTANT for call in step.message.tool_calls()]
    answered = [result.call_id for step in steps if step.message.role is Role.TOOL for result in step.message.tool_results()]
    return sorted(asked) == sorted(answered)


async def test_a_finished_tool_turn_is_stored_and_its_history_feeds_the_next_request_unchanged(database):
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "hi"}}, reasoning="plan"),
        text_step("Done.", reasoning="wrap up"),
        text_step("Anything else?"),
    )
    rig = Rig(adapter, [EchoTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    fresh = rig.fresh()
    stored = fresh.steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert [step.seq for step in stored] == [0, 1, 2, 3]
    assert [step.status for step in stored] == [StepStatus.COMPLETED] * 4
    state = fresh.load_state(outcome.state.turn_id)
    assert state.status is TurnStatus.COMPLETED and state.first_seq == 0 and state.rounds_used == 2
    assert state.gathered == () and state.pending == ()
    assert fresh.open_turn("s1") is None
    second_request = adapter.requests[1].messages
    assert codec.dumps(second_request) == codec.dumps([step.message for step in stored[:3]])

    history = tuple(step.message for step in fresh.steps("s1"))
    await drain(rig.engine.run(turn(history=history, new_messages=(user_message("more", message_id="u2"),))))

    third_request = adapter.requests[2].messages
    assert codec.dumps(third_request[:-1]) == codec.dumps([step.message for step in stored])
    assert third_request[-1].text() == "more"
    assert [step.seq for step in rig.fresh().steps("s1")] == [0, 1, 2, 3, 4, 5]


async def test_the_turn_row_exists_as_soon_as_the_first_step_is_stored(database):
    adapter = ScriptedAdapter(text_step("never read"))
    rig = Rig(adapter, [])
    events = rig.engine.run(turn(tool_context=None))

    first = await events.__anext__()

    assert isinstance(first, StepCommitted)
    running = rig.fresh().open_turn("s1")
    assert running.status is TurnStatus.RUNNING
    assert [step.message.role for step in rig.fresh().steps("s1")] == [Role.USER]
    await events.aclose()
    assert rig.fresh().load_state(running.turn_id).status is TurnStatus.CANCELLED


async def test_a_store_failure_while_aborting_does_not_hide_the_cancellation(database):
    rig = Rig(ScriptedAdapter(text_step("never read")), [])
    events = rig.engine.run(turn(tool_context=None))
    await events.__anext__()

    def broken(state):
        raise RuntimeError("database is gone")

    rig.store.save_state = broken

    await events.aclose()


async def test_an_approval_pause_is_stored_and_survives_a_restart(database):
    adapter = ScriptedAdapter(
        tool_step(
            {"id": "c1", "name": "echo", "args": {"message": "a"}},
            {"id": "c2", "name": "delete_prompt", "args": {"id": "p1"}},
            {"id": "c3", "name": "echo", "args": {"message": "b"}},
        ),
    )
    rig = Rig(adapter, [EchoTool(), DeleteTool()])

    outcome = outcome_of(await drain(rig.engine.run(turn())))

    restarted = rig.fresh()
    assert restarted.fail_interrupted() == 0
    waiting = restarted.open_turn("s1")
    assert waiting == outcome.state
    assert waiting.status is TurnStatus.AWAITING_DECISION
    assert [call.call_id for call in waiting.pending] == ["c2"]
    assert [part.call_id for part in waiting.gathered] == ["c1", "c3"]
    steps = restarted.steps("s1")
    assert [step.message.role for step in steps] == [Role.USER, Role.ASSISTANT]
    cards = projection.tool_executions(steps[1:], waiting)
    assert [(card["tool_call_id"], card["approval_status"]) for card in cards] == [("c1", None), ("c2", "pending"), ("c3", None)]
    assert cards[2]["result"]["success"] is False
    call_ids = decisions.execution_call_ids({"tool_executions": cards})
    assert decisions.decision_for(waiting, call_ids, 1, True).call_id == "c2"
    assert decisions.rejections_for_reply(waiting, "no")[0].call_id == "c2"


async def test_cancelling_mid_stream_keeps_the_partial_text_and_ends_the_turn(database):
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text="partial "), TextDelta(block=1, text="answer"), Finish(reason=FinishReason.STOP)],
    )
    rig = Rig(adapter, [])
    events = rig.engine.run(turn(tool_context=None))
    async for event in events:
        if isinstance(event, TextDelta):
            break

    await events.aclose()

    fresh = rig.fresh()
    stored = fresh.steps("s1")
    assert stored[-1].status is StepStatus.CANCELLED and stored[-1].message.text() == "partial "
    assert fresh.load_state(stored[-1].turn_id).status is TurnStatus.CANCELLED
    assert fresh.open_turn("s1") is None


async def test_cancelling_while_the_model_writes_a_tool_call_closes_the_call(database):
    adapter = ScriptedAdapter([
        ToolCallStart(block=2, call_id="c1", name="echo"),
        ToolArgsDelta(block=2, text='{"message": "h'),
        TextDelta(block=3, text="Let me look."),
        Finish(reason=FinishReason.TOOL_CALLS),
    ])
    rig = Rig(adapter, [EchoTool()])
    events = rig.engine.run(turn())
    async for event in events:
        if isinstance(event, TextDelta):
            break

    await events.aclose()

    stored = rig.fresh().steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert stored[1].status is StepStatus.CANCELLED and stored[1].message.tool_calls()[0].args is None
    (closing,) = stored[2].message.tool_results()
    assert (closing.call_id, closing.is_error, closing.text()) == ("c1", True, CANCELLED_REASON)
    assert stored[2].status is StepStatus.CANCELLED
    assert every_call_has_a_result(stored)


async def test_cancelling_while_a_tool_runs_keeps_the_results_already_gathered(database):
    hanging = HangingTool()
    adapter = ScriptedAdapter(tool_step(
        {"id": "c1", "name": "echo", "args": {"message": "done"}},
        {"id": "c2", "name": "hang", "args": {}},
    ))
    rig = Rig(adapter, [EchoTool(), hanging])
    task = asyncio.create_task(drain(rig.engine.run(turn())))
    waiting = asyncio.create_task(hanging.started.wait())
    await asyncio.wait({task, waiting}, return_when=asyncio.FIRST_COMPLETED)
    waiting.cancel()
    assert hanging.started.is_set(), "the turn ended before the tool started"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    fresh = rig.fresh()
    stored = fresh.steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    results = stored[2].message.tool_results()
    assert [(item.call_id, item.is_error) for item in results] == [("c1", False), ("c2", True)]
    assert results[0].text() == "echo: done"
    assert results[1].text() == CANCELLED_REASON
    assert fresh.load_state(stored[0].turn_id).status is TurnStatus.CANCELLED
    assert every_call_has_a_result(stored)


async def test_a_provider_error_after_a_tool_round_keeps_everything_before_it(database):
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}),
        Overloaded("busy", status=529),
    )
    rig = Rig(adapter, [EchoTool()])

    with pytest.raises(Overloaded):
        await drain(rig.engine.run(turn()))

    fresh = rig.fresh()
    stored = fresh.steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert stored[2].message.tool_results()[0].text() == "echo: a"
    state = fresh.load_state(stored[0].turn_id)
    assert state.status is TurnStatus.FAILED
    assert HarnessError.from_json(state.error) == Overloaded("busy", status=529)
    assert fresh.open_turn("s1") is None
    assert every_call_has_a_result(stored)


async def test_a_provider_error_on_the_first_call_keeps_the_user_message(database):
    rig = Rig(ScriptedAdapter(Overloaded("busy", status=529)), [])

    with pytest.raises(Overloaded):
        await drain(rig.engine.run(turn(tool_context=None)))

    fresh = rig.fresh()
    (only,) = fresh.steps("s1")
    assert only.message.role is Role.USER and only.message.parts == (TextPart(text="find cats"),)
    assert fresh.load_state(only.turn_id).status is TurnStatus.FAILED


async def test_a_failure_while_a_tool_round_is_open_closes_the_call_as_failed(database):
    class Exploding:
        async def run(self, calls, state, context):
            raise RuntimeError("runner broke")
            yield

    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {}}))
    rig = Rig(adapter, [EchoTool()], runner=Exploding())

    with pytest.raises(RuntimeError):
        await drain(rig.engine.run(turn()))

    fresh = rig.fresh()
    stored = fresh.steps("s1")
    assert [step.message.role for step in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL]
    assert stored[2].status is StepStatus.FAILED
    assert [(item.is_error, item.text()) for item in stored[2].message.tool_results()] == [(True, FAILED_REASON)]
    state = fresh.load_state(stored[0].turn_id)
    assert state.status is TurnStatus.FAILED and state.error["message"] == "runner broke"
    assert every_call_has_a_result(stored)

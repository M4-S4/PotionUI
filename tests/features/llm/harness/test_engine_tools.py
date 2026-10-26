import asyncio

from src.features.llm.harness.messages import Role
from src.features.llm.harness.turn_state import PendingCall, StepCommitted, ToolFinished, TurnStatus
from tests.features.llm.harness.probes import Probe, Shared
from tests.features.llm.harness.stepping import engine_turn, outcome_of
from tests.fixtures.llm_engine import drain, scripted_engine
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


def probes(log, shared, *names):
    return [
        Probe(name, log, read_only=name.startswith("read_"), gated=name.startswith("gate_"), shared=shared)
        for name in names
    ]


async def test_read_only_tools_in_one_round_run_together_and_reach_the_model_in_emission_order():
    log = []
    shared = Shared()
    shared.barrier = asyncio.Barrier(2)
    adapter = ScriptedAdapter(
        tool_step({"id": "a", "name": "read_a"}, {"id": "b", "name": "read_b"}),
        text_step("both read"),
    )
    rig = scripted_engine(adapter, probes(log, shared, "read_a", "read_b"))

    events = await asyncio.wait_for(drain(rig.engine.run(engine_turn())), 3)

    assert shared.peak == 2
    assert outcome_of(events).content == "both read"
    sent = adapter.requests[1].messages[-1]
    assert sent.role is Role.TOOL
    assert [(result.call_id, result.text()) for result in sent.tool_results()] == [("a", "read_a done"), ("b", "read_b done")]
    assert [execution.tool_call_id for execution in outcome_of(events).executions] == ["a", "b"]


async def test_a_round_with_an_approval_pauses_with_every_other_call_resolved():
    log = []
    adapter = ScriptedAdapter(
        tool_step({"id": "a", "name": "read_a"}, {"id": "b", "name": "gate_b"}, {"id": "c", "name": "write_c"}),
        text_step("never reached"),
    )
    rig = scripted_engine(adapter, probes(log, Shared(), "read_a", "gate_b", "write_c"))

    events = await drain(rig.engine.run(engine_turn()))

    outcome = outcome_of(events)
    assert outcome.pending and outcome.state.status is TurnStatus.AWAITING_DECISION
    assert [item.call_id for item in outcome.state.pending] == ["b"]
    assert [item.call_id for item in events if isinstance(item, PendingCall)] == []
    assert [result.call_id for result in outcome.state.gathered] == ["a", "c"]
    assert outcome.state.gathered[1].is_error and "waiting for approval of gate_b" in outcome.state.gathered[1].text()
    assert "start:write_c" not in log
    assert [(item.tool_call_id, item.approval_status, item.skipped) for item in outcome.executions] == [
        ("a", None, False), ("b", "pending", False), ("c", None, True),
    ]
    assert len(adapter.requests) == 1
    committed = [event.step.message.role for event in events if isinstance(event, StepCommitted)]
    assert committed == [Role.USER, Role.ASSISTANT]
    assert [event.result.call_id for event in events if isinstance(event, ToolFinished)] == ["a", "b", "c"]

from dataclasses import replace
from typing import Any, List

import pytest

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.errors import OutputTruncated
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised, StepCommitted, StepDiscarded, StepStatus, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from tests.features.llm.harness.stepping import engine_turn, outcome_of
from tests.fixtures.llm_engine import drain, scripted_config, scripted_engine
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


class WriteNote(BaseTool):
    name = "write_note"
    description = "Saves a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def __init__(self) -> None:
        self.saved: List[Any] = []

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.saved.append(kwargs)
        return ToolResult(success=True, data="saved")


def cut_off_call():
    return [
        ToolCallStart(block=2, call_id="t1", name="write_note"),
        ToolArgsDelta(block=2, text='{"text": "a long no'),
        Finish(reason=FinishReason.LENGTH, raw_reason="length"),
    ]


def complete_call():
    return tool_step({"id": "t1", "name": "write_note", "args": {"text": "a long note"}})


def truncation_notices(events):
    return [event.notice for event in events if isinstance(event, NoticeRaised) and event.notice.kind is NoticeKind.TRUNCATED]


async def test_a_cut_off_tool_call_is_retried_with_a_larger_limit_and_only_the_complete_call_runs():
    adapter = ScriptedAdapter(cut_off_call(), complete_call(), text_step("saved it"))
    note = WriteNote()
    rig = scripted_engine(adapter, [note])

    events = await drain(rig.engine.run(engine_turn(settings=Settings(max_tokens=2000))))

    assert note.saved == [{"text": "a long note"}]
    assert [request.settings.max_tokens for request in adapter.requests] == [2000, 8000, 2000]
    (notice,) = truncation_notices(events)
    assert notice.detail == {"max_tokens": 8000, "previous": 2000}
    outcome = outcome_of(events)
    assert outcome.content == "saved it"
    assert outcome.rounds_used == 2
    assert outcome.state.notices == (notice,)
    steps = rig.store.steps("s1")
    assert [step.message.role.value for step in steps] == ["user", "assistant", "tool", "assistant"]
    assert all(step.status is StepStatus.COMPLETED for step in steps)


async def test_a_call_that_is_cut_off_again_after_the_retry_fails_visibly_and_never_runs():
    adapter = ScriptedAdapter(cut_off_call(), cut_off_call(), text_step("never reached"))
    note = WriteNote()
    rig = scripted_engine(adapter, [note])
    seen: list = []

    with pytest.raises(OutputTruncated):
        async for event in rig.engine.run(engine_turn(settings=Settings(max_tokens=2000))):
            seen.append(event)

    assert note.saved == []
    assert len(adapter.requests) == 2
    notices = truncation_notices(seen)
    assert len(notices) == 2 and "again after the output limit was raised" in notices[1].message
    state = rig.store.load_state(rig.store.steps("s1")[0].turn_id)
    assert state.status is TurnStatus.FAILED
    assert state.error["kind"] == "output_truncated"
    stored = rig.store.steps("s1")
    failed, closing = stored[-2], stored[-1]
    assert failed.status is StepStatus.FAILED and failed.message.tool_calls()[0].args is None
    assert closing.message.role.value == "tool" and closing.status is StepStatus.FAILED
    assert closing.message.tool_results()[0].call_id == "t1" and closing.message.tool_results()[0].is_error


class CappedProfiles:
    def __init__(self, inner, cap):
        self._inner = inner
        self._cap = cap

    def resolve(self, config):
        return replace(self._inner.resolve(config), max_output_tokens=self._cap)

    async def refresh(self, config):
        return self.resolve(config)

    async def ensure(self, config):
        return self.resolve(config)


async def test_the_retry_grows_the_limit_the_configuration_sends_when_the_turn_sets_none():
    adapter = ScriptedAdapter(cut_off_call(), complete_call(), text_step("saved it"))
    note = WriteNote()
    rig = scripted_engine(adapter, [note], config=scripted_config(max_tokens=2000))

    events = await drain(rig.engine.run(engine_turn()))

    assert note.saved == [{"text": "a long note"}]
    assert [request.settings.max_tokens for request in adapter.requests] == [2000, 8000, 2000]
    assert [request.config.max_tokens for request in adapter.requests] == [2000, 2000, 2000]
    (notice,) = truncation_notices(events)
    assert notice.detail == {"max_tokens": 8000, "previous": 2000}


async def test_the_retry_stops_at_the_profile_maximum():
    adapter = ScriptedAdapter(cut_off_call(), complete_call(), text_step("saved it"))
    rig = scripted_engine(adapter, [WriteNote()], config=scripted_config(max_tokens=2000))
    rig.gateway.profiles = CappedProfiles(rig.gateway.profiles, 5000)

    await drain(rig.engine.run(engine_turn()))

    assert [request.settings.max_tokens for request in adapter.requests] == [2000, 5000, 2000]


async def test_a_limit_that_cannot_be_raised_fails_at_once():
    adapter = ScriptedAdapter(cut_off_call(), text_step("never reached"))
    note = WriteNote()
    rig = scripted_engine(adapter, [note], config=scripted_config(max_tokens=2000))
    rig.gateway.profiles = CappedProfiles(rig.gateway.profiles, 2000)

    with pytest.raises(OutputTruncated) as raised:
        await drain(rig.engine.run(engine_turn()))

    assert note.saved == [] and len(adapter.requests) == 1
    assert raised.value.detail["max_tokens"] == 2000


async def test_a_cut_off_answer_is_kept_and_marked_incomplete_with_a_notice():
    cut_text = [TextDelta(block=1, text="The answer is"), Finish(reason=FinishReason.LENGTH, raw_reason="length")]
    adapter = ScriptedAdapter(cut_text)
    rig = scripted_engine(adapter, [WriteNote()])

    events = await drain(rig.engine.run(engine_turn()))

    outcome = outcome_of(events)
    assert outcome.content == "The answer is" and outcome.reason == "answer"
    assert rig.store.steps("s1")[-1].status is StepStatus.INCOMPLETE
    assert len(truncation_notices(events)) == 1
    assert len(adapter.requests) == 1


async def test_a_cut_off_answer_without_tools_is_marked_incomplete_too():
    cut_text = [TextDelta(block=1, text="The answer is"), Finish(reason=FinishReason.LENGTH, raw_reason="length")]
    rig = scripted_engine(ScriptedAdapter(cut_text))

    events = await drain(rig.engine.run(engine_turn(tool_context=None)))

    assert rig.store.steps("s1")[-1].status is StepStatus.INCOMPLETE
    assert len(truncation_notices(events)) == 1


async def test_a_reply_that_finished_normally_raises_no_truncation_notice():
    rig = scripted_engine(ScriptedAdapter(text_step("fine")), [WriteNote()])

    events = await drain(rig.engine.run(engine_turn()))

    assert truncation_notices(events) == []
    assert rig.store.steps("s1")[-1].status is StepStatus.COMPLETED


async def test_a_retried_cut_off_step_that_showed_text_is_discarded_at_the_seq_its_retry_takes():
    adapter = ScriptedAdapter([TextDelta(block=1, text="Saving it. "), *cut_off_call()], complete_call(), text_step("saved it"))
    rig = scripted_engine(adapter, [WriteNote()])

    events = await drain(rig.engine.run(engine_turn(settings=Settings(max_tokens=2000))))

    (discarded,) = [event for event in events if isinstance(event, StepDiscarded)]
    committed = [event for event in events if isinstance(event, StepCommitted)]
    retried = committed[1]
    assert retried.step.message.tool_calls()
    assert discarded.seq == retried.step.seq
    assert events.index(discarded) < events.index(retried)


async def test_a_retried_cut_off_step_that_showed_nothing_is_not_announced():
    adapter = ScriptedAdapter(cut_off_call(), complete_call(), text_step("saved it"))
    rig = scripted_engine(adapter, [WriteNote()])

    events = await drain(rig.engine.run(engine_turn(settings=Settings(max_tokens=2000))))

    assert not [event for event in events if isinstance(event, StepDiscarded)]

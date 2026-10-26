from typing import Any, List

import pytest

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.errors import OutputTruncated
from src.features.llm.harness.events import Finish, FinishReason, TextDelta, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised, StepStatus, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from tests.features.llm.harness.stepping import engine_turn, outcome_of
from tests.fixtures.llm_engine import drain, scripted_engine
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
    assert [request.settings.max_tokens for request in adapter.requests] == [2000, 8000, 8000]
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
    failed = rig.store.steps("s1")[-1]
    assert failed.status is StepStatus.FAILED and failed.message.tool_calls()[0].args is None


async def test_a_limit_that_cannot_be_raised_fails_at_once():
    adapter = ScriptedAdapter(cut_off_call(), text_step("never reached"))
    note = WriteNote()
    rig = scripted_engine(adapter, [note])

    with pytest.raises(OutputTruncated):
        await drain(rig.engine.run(engine_turn()))

    assert note.saved == [] and len(adapter.requests) == 1


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

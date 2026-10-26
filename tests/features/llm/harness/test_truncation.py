from dataclasses import replace

from src.features.llm.harness.adapter import Settings
from src.features.llm.harness.events import Finish, FinishReason, ToolArgsDelta, ToolCallStart
from src.features.llm.harness.messages import Origin, TextPart, ToolCallPart, ToolResultPart, message
from src.features.llm.harness.tool_runner import ParallelToolRunner, ToolOutcome
from src.features.llm.harness.truncation import (
    MAX_TRUNCATION_RETRIES,
    TRUNCATION_GROWTH,
    OutputTruncated,
    TruncationAction,
    decide_truncation,
    raised_limit,
)
from src.features.llm.harness.turn_state import NoticeKind, ToolFinished
from tests.features.llm.harness.stepping import play_step, request_for, turn_state
from tests.fixtures.llm_harness import ScriptedAdapter, scripted_profile, tool_step

ORIGIN = Origin(adapter="scripted", provider="scripted", model="m")


def profile(**changes):
    return replace(scripted_profile(), **changes)


def reply(*parts):
    return message("msg", "assistant", *parts, origin=ORIGIN)


def call_part(args=None):
    return ToolCallPart(call_id="c1", name="write_note", args=args)


def test_a_reply_that_was_not_cut_off_needs_nothing():
    for reason in (FinishReason.STOP, FinishReason.TOOL_CALLS, None):
        decision = decide_truncation(reply(call_part({})), reason, Settings(), profile())
        assert decision.action is TruncationAction.NONE
        assert decision.notice is None


def test_a_cut_off_tool_call_is_retried_once_with_a_larger_limit():
    decision = decide_truncation(
        reply(call_part(None)), FinishReason.LENGTH, Settings(max_tokens=1000), profile(max_output_tokens=64000),
    )
    assert decision.action is TruncationAction.RETRY
    assert decision.max_tokens == 1000 * TRUNCATION_GROWTH
    assert decision.notice.kind is NoticeKind.TRUNCATED
    assert decision.notice.detail == {"max_tokens": 4000, "previous": 1000}


def test_the_larger_limit_never_passes_the_profile_maximum():
    assert raised_limit(Settings(max_tokens=30000), profile(max_output_tokens=32768)) == 32768
    decision = decide_truncation(
        reply(call_part(None)), FinishReason.LENGTH, Settings(max_tokens=30000), profile(max_output_tokens=32768),
    )
    assert decision.max_tokens == 32768


def test_an_unset_limit_goes_to_the_profile_maximum():
    assert raised_limit(Settings(), profile(max_output_tokens=8192)) == 8192


def test_a_limit_that_cannot_grow_fails_visibly_without_a_retry():
    for settings, max_output in ((Settings(max_tokens=8192), 8192), (Settings(), None), (Settings(max_tokens=100), 100)):
        decision = decide_truncation(
            reply(call_part(None)), FinishReason.LENGTH, settings, profile(max_output_tokens=max_output),
        )
        assert decision.action is TruncationAction.FAIL
        assert isinstance(decision.error, OutputTruncated)
        assert "tool was not run" in decision.error.message
        assert decision.notice.kind is NoticeKind.TRUNCATED


def test_a_second_cut_off_after_the_retry_fails_visibly():
    decision = decide_truncation(
        reply(call_part(None)),
        FinishReason.LENGTH,
        Settings(max_tokens=4000),
        profile(max_output_tokens=64000),
        retries_used=MAX_TRUNCATION_RETRIES,
    )
    assert decision.action is TruncationAction.FAIL
    assert "again after the output limit was raised" in decision.error.message
    assert decision.max_tokens is None


def test_cut_off_text_is_kept_and_marked_incomplete():
    decision = decide_truncation(
        reply(TextPart("The answer is")), FinishReason.LENGTH, Settings(max_tokens=100), profile(max_output_tokens=1000),
    )
    assert decision.action is TruncationAction.INCOMPLETE
    assert decision.error is None
    assert decision.notice.kind is NoticeKind.TRUNCATED


async def test_a_truncated_call_is_retried_with_a_larger_limit_and_only_the_complete_call_runs():
    adapter = ScriptedAdapter(
        [
            ToolCallStart(block=0, call_id="t1", name="write_note"),
            ToolArgsDelta(block=0, text='{"text": "a long no'),
            Finish(reason=FinishReason.LENGTH, raw_reason="length"),
        ],
        tool_step({"id": "t1", "name": "write_note", "args": {"text": "a long note"}}),
    )
    scripted = profile(max_output_tokens=16000)
    settings = Settings(max_tokens=2000)
    executed = []

    class Backend:
        def parallel_safe(self, name):
            return False

        async def execute(self, call, context, state):
            executed.append(call.args)
            return ToolOutcome(result=ToolResultPart(call_id=call.call_id, name=call.name, content=(TextPart("saved"),)))

        async def confirm(self, call, context, state):
            raise AssertionError("not gated")

    runner = ParallelToolRunner(Backend())
    retries = 0
    while True:
        step, assembler = await play_step(adapter, scripted, request_for(settings=settings), message_id=f"msg_{retries}")
        decision = decide_truncation(step, assembler.finish.reason, settings, scripted, retries)
        if decision.action is not TruncationAction.RETRY:
            break
        assert executed == []
        settings = replace(settings, max_tokens=decision.max_tokens)
        retries += 1
    assert decision.action is TruncationAction.NONE
    assert [request.settings.max_tokens for request in adapter.requests] == [2000, 8000]
    events = [event async for event in runner.run(step.tool_calls(), turn_state(), None)]
    assert executed == [{"text": "a long note"}]
    assert [event.result.is_error for event in events if isinstance(event, ToolFinished)] == [False]

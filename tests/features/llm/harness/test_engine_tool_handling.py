from typing import Any, Dict, List

import pytest

from src.features.llm.harness.build import JPEG, user_message
from src.features.llm.harness.engine import MAX_RESCUE_RETRIES, TurnInput, TurnOutcome
from src.features.llm.harness.errors import RateLimited
from src.features.llm.harness.events import Finish, FinishReason, TextDelta
from src.features.llm.harness.messages import ImagePart, Role, TextPart
from src.features.llm.harness.retry import BackoffRetry
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised, ToolFinished, TurnStatus
from src.features.llm.tools.base import BaseTool, ToolApprovalPreview, ToolContext, ToolResult, ToolSource
from src.features.llm.tools.executor import ToolExecutor
from tests.fixtures.llm_engine import drain, scripted_config, scripted_engine
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes a message back."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


class OtherTool(BaseTool):
    name = "other"
    description = "Another tool."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="other")


class RaisingTool(BaseTool):
    name = "fail"
    description = "Always raises."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        raise RuntimeError("intentional failure")


class ImageTool(BaseTool):
    name = "image_tool"
    description = "Returns an image."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="rendered", image_data="dG9vbA==")


class BigResultTool(BaseTool):
    name = "big_result"
    description = "Returns a lot."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="x" * 20000)


class SourcedTool(BaseTool):
    name = "sourced"
    description = "Returns sources."
    parameters = {"type": "object", "properties": {}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="found", sources=[ToolSource(source_type="prompt", title="Cats", url="https://x.invalid/p1")])


class PreviewTool(BaseTool):
    name = "preview"
    description = "Needs approval."
    parameters = {"type": "object", "properties": {}}
    requires_approval = True

    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        if not self.succeed:
            return ToolResult(success=False, data="", error="nothing to change")
        return ToolResult(success=True, data="", preview=ToolApprovalPreview(action="Change"))


def turn(**overrides: Any) -> TurnInput:
    values: Dict[str, Any] = dict(
        session_id="s1",
        llm_id="cfg-scripted",
        system="SYS",
        history=(),
        new_messages=(user_message("go", message_id="u1"),),
        tool_context=ToolContext(user_id="user-1", llm_id="cfg-scripted"),
    )
    values.update(overrides)
    return TurnInput(**values)


def outcome(events: List[Any]) -> TurnOutcome:
    return next(event for event in events if isinstance(event, TurnOutcome))


def visible(events: List[Any]) -> str:
    return "".join(event.text for event in events if isinstance(event, TextDelta))


def results(request) -> List[Any]:
    return [part for message in request.messages for part in message.tool_results()]


class _VisionRepository:
    def __init__(self, supports_vision):
        self.supports_vision = supports_vision

    def get_configuration(self, config_id):
        return scripted_config(supports_vision=self.supports_vision) if self.supports_vision is not None else None


async def test_only_allowed_tools_are_offered_and_a_disallowed_call_gets_a_teaching_error():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "other", "args": {}}), text_step("ok"))
    rig = scripted_engine(adapter, [EchoTool(), OtherTool()])

    events = await drain(rig.engine.run(turn(allowed_tools=("echo",))))

    assert [tool.name for tool in adapter.requests[0].tools] == ["echo"]
    [result] = results(adapter.requests[1])
    assert result.is_error and "other" in result.text() and "echo" in result.text()
    assert outcome(events).executions[0].result.success is False


async def test_an_unknown_tool_and_a_raising_tool_both_return_error_results():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "missing", "args": {}}, {"id": "c2", "name": "fail", "args": {}}),
        text_step("ok"),
    )
    rig = scripted_engine(adapter, [RaisingTool()])

    await drain(rig.engine.run(turn()))

    missing, failing = results(adapter.requests[1])
    assert missing.is_error and "not found" in missing.text()
    assert failing.is_error and failing.text().startswith("Error:")


async def test_the_users_image_stays_on_the_user_message_for_every_step():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("seen"))
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn(new_messages=(user_message("look", images=("aW1n",)),))))

    for request in adapter.requests:
        assert request.messages[0].parts[1] == ImagePart(media_type=JPEG, data="aW1n")


@pytest.mark.parametrize(
    "supports_vision, attached",
    [(True, True), (None, True), (False, False)],
)
async def test_a_tool_image_is_kept_only_when_the_model_can_see_it(supports_vision, attached):
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "image_tool", "args": {}}), text_step("ok"))
    rig = scripted_engine(adapter, [ImageTool()])
    context = ToolContext(user_id="user-1", llm_id="cfg-scripted", llm_repository=_VisionRepository(supports_vision))

    await drain(rig.engine.run(turn(tool_context=context)))

    [result] = results(adapter.requests[1])
    images = [item for item in result.content if isinstance(item, ImagePart)]
    assert bool(images) is attached
    assert ("cannot see images" in result.text()) is not attached


async def test_a_failed_preview_does_not_pause_the_turn():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "preview", "args": {}}), text_step("nothing to do"))
    rig = scripted_engine(adapter, [PreviewTool(succeed=False)])

    events = await drain(rig.engine.run(turn()))

    assert outcome(events).state.status is TurnStatus.COMPLETED
    assert outcome(events).content == "nothing to do"


async def test_tool_end_payloads_carry_sources():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "sourced", "args": {}}), text_step("ok"))
    rig = scripted_engine(adapter, [SourcedTool()])

    events = await drain(rig.engine.run(turn()))

    finished = next(event for event in events if isinstance(event, ToolFinished))
    payload = ToolExecutor.tool_end_payload(finished.record)
    assert payload["sources"][0]["title"] == "Cats"
    assert payload["success"] is True and "arguments" not in payload


async def test_an_oversized_result_is_truncated_with_a_teaching_marker():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "big_result", "args": {}}), text_step("ok"))
    rig = scripted_engine(adapter, [BigResultTool()])

    await drain(rig.engine.run(turn()))

    [result] = results(adapter.requests[1])
    assert len(result.text()) < 9000
    assert result.text().startswith("x" * 100)
    assert "[Result truncated at 8000 characters." in result.text()


async def test_a_tool_action_tag_is_repaired_into_a_dispatched_call():
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text='<tool_action type="echo" message="hi">'), Finish(reason=FinishReason.STOP)],
        text_step("done echoing"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    result = outcome(events)
    assert result.content == "done echoing"
    assert [(e.tool_name, e.arguments) for e in result.executions] == [("echo", {"message": "hi"})]
    assert result.rescues == [{"tool_name": "echo", "repaired": True, "original_format": "tool_action_tag"}]
    assert "<tool_action" not in visible(events)


async def test_an_ambiguous_near_miss_is_steered_then_bounded_with_an_honest_fallback():
    ambiguous = [TextDelta(block=1, text='<tool_action type="echo">'), Finish(reason=FinishReason.STOP)]
    adapter = ScriptedAdapter(*([ambiguous] * (MAX_RESCUE_RETRIES + 1)))
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    result = outcome(events)
    assert len(adapter.requests) == MAX_RESCUE_RETRIES + 1
    assert result.reason == "ambiguous" and "couldn't format" in result.content
    assert result.executions == () and result.completion is None
    second = adapter.requests[1].messages
    assert any(message.role is Role.USER and "<tool_call>" in message.parts[0].text for message in second[1:])


async def test_prose_mentioning_a_tool_is_not_rescued():
    adapter = ScriptedAdapter(text_step("I already used echo to check that."))
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome(events).content == "I already used echo to check that."
    assert outcome(events).rescues is None and len(adapter.requests) == 1


async def test_an_unclosed_tool_call_never_shows_partial_json_and_falls_back_after_bounded_retries():
    truncated = [TextDelta(block=1, text='<tool_call>{"name": "echo", "arguments": {"mess'), Finish(reason=FinishReason.LENGTH)]
    adapter = ScriptedAdapter(*([truncated] * (MAX_RESCUE_RETRIES + 1)))
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert visible(events) == ""
    assert outcome(events).reason == "truncated"
    assert '{"name"' not in outcome(events).content


async def test_a_closed_tool_call_with_broken_json_gets_a_corrective_retry():
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text='<tool_call>{"name": "echo", "arguments": {message: hi}}</tool_call>'), Finish(reason=FinishReason.STOP)],
        text_step("recovered"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome(events).content == "recovered"
    assert len(adapter.requests) == 2


async def test_a_registered_tool_action_never_leaks_while_an_unregistered_one_streams():
    adapter = ScriptedAdapter(
        [
            TextDelta(block=1, text='Here <tool_action type="update_segment" id="1">new text</tool_action> '),
            TextDelta(block=1, text='<tool_act'),
            TextDelta(block=1, text='ion type="echo" message="x"></tool_action>'),
            Finish(reason=FinishReason.STOP),
        ],
        text_step("done"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    shown = visible(events)
    assert '<tool_action type="update_segment" id="1">new text</tool_action>' in shown
    assert 'type="echo"' not in shown


async def test_different_arguments_and_successful_repeats_are_not_blocked():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}),
        tool_step({"id": "c2", "name": "echo", "args": {"message": "a"}}, {"id": "c3", "name": "echo", "args": {"message": "b"}}),
        text_step("ok"),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert [part.text() for part in results(adapter.requests[2])] == ["echo: a", "echo: a", "echo: b"]
    assert outcome(events).tool_failures is None


async def test_failures_are_counted_per_tool():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "fail", "args": {}}, {"id": "c2", "name": "missing", "args": {}}),
        text_step("ok"),
    )
    rig = scripted_engine(adapter, [RaisingTool()])

    events = await drain(rig.engine.run(turn()))

    assert outcome(events).tool_failures == {"fail": 1, "missing": 1}


async def test_the_callers_history_is_never_mutated():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("ok"))
    rig = scripted_engine(adapter, [EchoTool()])
    history = (user_message("earlier", message_id="u0"),)
    request = turn(history=history)

    await drain(rig.engine.run(request))

    assert request.history == history
    assert [message.id for message in adapter.requests[0].messages] == ["u0", "u1"]


async def test_streamed_text_before_a_tool_call_reaches_the_caller_and_the_stored_step():
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}, text="Checking. "),
        text_step("Done."),
    )
    rig = scripted_engine(adapter, [EchoTool()])

    events = await drain(rig.engine.run(turn()))

    assert visible(events) == "Checking. Done."
    assert adapter.requests[1].messages[1].parts[0] == TextPart(text="Checking. ")


async def test_a_text_tool_call_turned_into_a_call_drops_the_raw_reply_while_an_untouched_step_keeps_it():
    xml = '<tool_call>{"name": "echo", "arguments": {"message": "hi"}}</tool_call>'
    adapter = ScriptedAdapter(
        [TextDelta(block=1, text=xml), Finish(reason=FinishReason.STOP, raw_reason="stop", raw={"content": xml})],
        [TextDelta(block=1, text="done"), Finish(reason=FinishReason.STOP, raw_reason="stop", raw={"content": "done"})],
    )
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn()))

    call_step, answer_step = [step for step in rig.store.steps("s1") if step.message.role is Role.ASSISTANT]
    assert [call.name for call in call_step.message.tool_calls()] == ["echo"]
    assert call_step.message.text() == ""
    assert call_step.message.raw is None
    assert answer_step.message.raw == {"content": "done"}
    assert adapter.requests[1].messages[1].raw is None


async def test_a_call_that_failed_before_the_turn_resumed_is_still_refused_with_its_first_error():
    class CountingEcho(EchoTool):
        runs = 0

        async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
            CountingEcho.runs += 1
            return await super().execute(context, **kwargs)

    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("gave up"))
    rig = scripted_engine(adapter, [CountingEcho()])

    events = await drain(rig.engine.run(turn(failed_calls={'echo:{"message": "a"}': "disk on fire"})))

    assert CountingEcho.runs == 0
    (result,) = results(adapter.requests[1])
    assert result.is_error and "it failed: disk on fire" in result.text()
    assert outcome(events).state.failed_calls == {'echo:{"message": "a"}': "disk on fire"}


async def test_offering_a_tool_that_needs_approval_asks_for_the_long_prompt_cache():
    adapter = ScriptedAdapter(text_step("hi"))
    rig = scripted_engine(adapter, [EchoTool(), PreviewTool()])

    await drain(rig.engine.run(turn()))

    assert adapter.requests[0].settings.cache_ttl == "1h"


async def test_without_approval_tools_the_prompt_cache_ttl_is_left_alone():
    adapter = ScriptedAdapter(text_step("hi"))
    rig = scripted_engine(adapter, [EchoTool()])

    await drain(rig.engine.run(turn()))

    assert adapter.requests[0].settings.cache_ttl is None


async def test_the_same_notice_is_raised_once_per_turn():
    slow = RateLimited("slow down", status=429)
    adapter = ScriptedAdapter(
        slow, tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), slow, text_step("done"),
    )
    rig = scripted_engine(adapter, [EchoTool()], retry=BackoffRetry(max_retries=1, jitter=lambda: 0.0))

    events = await drain(rig.engine.run(turn()))

    raised = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in raised] == [NoticeKind.RETRYING]
    assert outcome(events).state.notices == tuple(raised)
    assert outcome(events).content == "done"
    assert len(rig.sleeps) == 2


class FlakyTool(BaseTool):
    name = "flaky"
    description = "Times out once."
    parameters = {"type": "object", "properties": {}}

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("the service took too long")
        return ToolResult(success=True, data="ok")


async def test_a_call_that_raised_may_be_tried_again_with_the_same_arguments():
    flaky = FlakyTool()
    adapter = ScriptedAdapter(
        tool_step({"id": "c1", "name": "flaky", "args": {}}), tool_step({"id": "c2", "name": "flaky", "args": {}}), text_step("done"),
    )
    rig = scripted_engine(adapter, [flaky])

    events = await drain(rig.engine.run(turn()))

    assert flaky.calls == 2
    assert outcome(events).tool_failures == {"flaky": 1}
    assert outcome(events).state.failed_calls == {}

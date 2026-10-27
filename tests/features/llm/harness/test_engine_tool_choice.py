from dataclasses import replace
from typing import Any

from src.features.llm.harness.adapter import Settings, ToolChoice, ToolSpec
from src.features.llm.harness.build import user_message
from src.features.llm.harness.messages import Message, Role, TextPart, ToolCallPart, ToolResultPart
from src.features.llm.harness.turn_state import NoticeKind, NoticeRaised
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from tests.features.llm.harness.stepping import engine_turn, outcome_of
from tests.fixtures.llm_engine import drain, scripted_engine
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="echoed")


class ChoiceProfiles:
    def __init__(self, inner: Any, *modes: str) -> None:
        self._inner = inner
        self._modes = frozenset(modes)

    def resolve(self, config: Any) -> Any:
        return replace(self._inner.resolve(config), tool_choice=self._modes)

    async def refresh(self, config: Any) -> Any:
        return self.resolve(config)

    async def ensure(self, config: Any) -> Any:
        return self.resolve(config)


def rig_with(adapter, *modes):
    rig = scripted_engine(adapter, [EchoTool()])
    rig.gateway.profiles = ChoiceProfiles(rig.gateway.profiles, *modes)
    return rig


def tool_history():
    return (
        user_message("echo it", message_id="h1"),
        Message(id="h2", role=Role.ASSISTANT, parts=(ToolCallPart(call_id="c0", name="echo", args={"message": "a"}),), origin=SCRIPTED_ORIGIN),
        Message(id="h3", role=Role.TOOL, parts=(ToolResultPart(call_id="c0", name="echo", content=(TextPart(text="echoed"),)),)),
        Message(id="h4", role=Role.ASSISTANT, parts=(TextPart(text="Done."),), origin=SCRIPTED_ORIGIN),
    )


async def test_the_budget_wrap_up_keeps_the_offered_tools_and_forbids_calling_them():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("out of rounds"))
    rig = rig_with(adapter, "auto", "none")

    events = await drain(rig.engine.run(engine_turn(max_rounds=1)))

    wrap_up = adapter.requests[-1]
    assert [tool.name for tool in wrap_up.tools] == [tool.name for tool in adapter.requests[0].tools] == ["echo"]
    assert wrap_up.settings.tool_choice == ToolChoice("none")
    assert outcome_of(events).content == "out of rounds"


async def test_the_wrap_up_drops_the_tools_when_the_model_cannot_be_told_not_to_call_them():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("out of rounds"))
    rig = rig_with(adapter, "auto")

    await drain(rig.engine.run(engine_turn(max_rounds=1)))

    assert adapter.requests[-1].tools == ()
    assert adapter.requests[-1].settings.tool_choice is None


async def test_a_tools_off_answer_over_tool_history_declares_the_tools_in_use_without_letting_them_run():
    spec = ToolSpec(name="echo", description="Echoes.", parameters=EchoTool.parameters)
    adapter = ScriptedAdapter(text_step("plain answer"))
    rig = rig_with(adapter, "auto", "none")

    events = await drain(rig.engine.run(engine_turn(tool_context=None, history=tool_history(), tools_in_use=(spec,))))

    assert adapter.requests[0].tools == (spec,)
    assert adapter.requests[0].settings.tool_choice == ToolChoice("none")
    assert outcome_of(events).state.snapshot.tools == (spec,)


async def test_a_tools_off_answer_without_tools_in_use_declares_the_registered_tools_its_history_called():
    adapter = ScriptedAdapter(text_step("plain answer"))
    rig = rig_with(adapter, "auto", "none")

    await drain(rig.engine.run(engine_turn(tool_context=None, history=tool_history())))

    assert [tool.name for tool in adapter.requests[0].tools] == ["echo"]
    assert adapter.requests[0].settings.tool_choice == ToolChoice("none")


async def test_a_tools_off_answer_without_tool_history_sends_no_tools():
    adapter = ScriptedAdapter(text_step("plain answer"))
    rig = rig_with(adapter, "auto", "none")

    await drain(rig.engine.run(engine_turn(tool_context=None, history=(user_message("hi", message_id="h1"),))))

    assert adapter.requests[0].tools == ()
    assert adapter.requests[0].settings.tool_choice is None


async def test_calls_returned_despite_none_are_not_kept_in_a_tools_off_answer():
    adapter = ScriptedAdapter(tool_step({"id": "c9", "name": "echo", "args": {}}, text="Answer."))
    rig = rig_with(adapter, "auto", "none")

    events = await drain(rig.engine.run(engine_turn(tool_context=None, history=tool_history())))

    stored = rig.store.steps("s1")[-1].message
    assert stored.tool_calls() == ()
    assert stored.text() == "Answer."
    assert outcome_of(events).content == "Answer."


async def test_a_downgraded_tool_choice_is_announced_once_per_turn():
    adapter = ScriptedAdapter(tool_step({"id": "c1", "name": "echo", "args": {"message": "a"}}), text_step("done"))
    rig = rig_with(adapter, "auto")

    events = await drain(rig.engine.run(engine_turn(settings=Settings(tool_choice=ToolChoice("required")))))

    raised = [event.notice for event in events if isinstance(event, NoticeRaised)]
    assert [notice.kind for notice in raised] == [NoticeKind.TOOL_CHOICE_DOWNGRADED]
    assert all(request.settings.tool_choice == ToolChoice("auto") for request in adapter.requests)
    assert outcome_of(events).state.notices == tuple(raised)

from typing import Any, List, Optional, Sequence

from src.features.llm.harness.adapter import RenderRequest, ToolSpec
from src.features.llm.harness.build import user_message
from src.features.llm.harness.cache import Divergence, Segment, first_divergence, request_segments
from src.features.llm.harness.engine import TurnInput, TurnOutcome
from src.features.llm.harness.messages import ContextPart, Message, Role
from src.features.llm.tools.base import BaseTool, ToolContext, ToolResult
from tests.fixtures.llm_engine import EngineRig, drain, scripted_engine, tool_context
from tests.fixtures.llm_harness import ScriptedAdapter, text_step, tool_step


class EchoTool(BaseTool):
    name = "echo"
    description = "Echoes a message back."
    parameters = {"type": "object", "properties": {"message": {"type": "string"}}}

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data=f"echo: {kwargs.get('message', '')}")


class LateTool(BaseTool):
    name = "late"
    description = "Only loaded on demand."
    parameters = {"type": "object", "properties": {}}
    context_group = "extras"

    async def execute(self, context: ToolContext, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, data="late result")


def turn(**overrides: Any) -> TurnInput:
    values = dict(
        session_id="s1",
        llm_id="cfg-scripted",
        system="SYS",
        history=(),
        tool_context=tool_context(),
        allowed_tools=("echo",),
        deferred_tools={"extras": ("late",)},
    )
    values.update(overrides)
    return TurnInput(**values)


def tool_names(request: RenderRequest) -> List[str]:
    return [spec.name for spec in request.tools]


def outcome_of(events: Sequence[Any]) -> TurnOutcome:
    return next(event for event in events if isinstance(event, TurnOutcome))


def stored(rig: EngineRig) -> tuple:
    return tuple(step.message for step in rig.store.steps("s1"))


def without_context(message: Message) -> Message:
    if message.role is not Role.USER:
        return message
    return Message(id=message.id, role=message.role, parts=tuple(p for p in message.parts if not isinstance(p, ContextPart)))


def question(number: int) -> Message:
    return user_message(
        f"question {number}",
        contexts=[ContextPart(kind="workspace", text=f"workspace state {number}")],
        message_id=f"u{number}",
    )


def session_script() -> ScriptedAdapter:
    return ScriptedAdapter(
        tool_step({"id": "c1", "name": "echo", "args": {"message": "one"}}),
        text_step("first answer"),
        tool_step({"id": "c2", "name": "use_tools", "args": {"group": "extras"}}),
        tool_step({"id": "c3", "name": "late", "args": {}}),
        text_step("second answer"),
        text_step("third answer"),
    )


async def run_session(rig: EngineRig, *, keep_context: bool = True, seed_tools: bool = True) -> None:
    history: tuple = ()
    tools_in_use: tuple = ()
    for number in (1, 2, 3):
        events = await drain(rig.engine.run(turn(
            history=history, new_messages=(question(number),), tools_in_use=tools_in_use,
        )))
        messages = stored(rig)
        history = messages if keep_context else tuple(without_context(message) for message in messages)
        if seed_tools:
            tools_in_use = outcome_of(events).state.snapshot.tools


def consecutive(requests: Sequence[RenderRequest]) -> List[Optional[Divergence]]:
    segments = [request_segments(request) for request in requests]
    return [first_divergence(before, after) for before, after in zip(segments, segments[1:])]


def tools_end(request: RenderRequest) -> int:
    return sum(
        len(segment.data) + 1
        for segment in request_segments(request)
        if segment.section == "params" or segment.section.startswith("tools")
    )


def sections(request: RenderRequest, prefix: str) -> List[Segment]:
    return [segment for segment in request_segments(request) if segment.section.startswith(prefix)]


async def test_a_three_turn_session_only_ever_extends_the_previous_request():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    requests = adapter.requests
    assert len(requests) == 6
    found = consecutive(requests)
    loaded = requests[2]
    assert found == [None, None, Divergence(offset=tools_end(loaded), section="system"), None, None]


async def test_the_system_prompt_is_byte_identical_in_every_request():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    systems = {segment.data for request in adapter.requests for segment in sections(request, "system")}
    assert len(systems) == 1
    assert {request.system for request in adapter.requests} == {"SYS"}


async def test_loading_a_group_with_use_tools_appends_to_the_tool_list_without_reordering():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    names = [tool_names(request) for request in adapter.requests]
    assert names == [
        ["echo", "use_tools"],
        ["echo", "use_tools"],
        ["echo", "use_tools"],
        ["echo", "use_tools", "late"],
        ["echo", "use_tools", "late"],
        ["echo", "use_tools", "late"],
    ]
    before, after = adapter.requests[2], adapter.requests[3]
    assert sections(after, "tools")[:2] == sections(before, "tools")


async def test_the_use_tools_schema_that_went_out_is_not_rewritten_when_the_group_list_shrinks():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    first = next(spec for spec in adapter.requests[0].tools if spec.name == "use_tools")
    last = next(spec for spec in adapter.requests[5].tools if spec.name == "use_tools")
    assert first == last and "extras" in first.description


async def test_without_the_carried_tool_list_a_new_turn_drops_the_loaded_group_and_breaks_the_prefix():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig, seed_tools=False)

    assert tool_names(adapter.requests[5]) == ["echo", "use_tools"]
    before_third_tool = sum(len(segment.data) + 1 for segment in request_segments(adapter.requests[4])[:3])
    assert consecutive(adapter.requests)[-1] == Divergence(offset=before_third_tool, section="tools[2]")


async def test_per_turn_context_stays_on_the_stored_user_message_for_every_later_request():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    last = adapter.requests[5].messages
    users = [message for message in last if message.role is Role.USER]
    assert [[part.text for part in message.parts if isinstance(part, ContextPart)] for message in users] == [
        ["workspace state 1"], ["workspace state 2"], ["workspace state 3"],
    ]


async def test_context_attached_only_when_the_request_is_built_breaks_the_prefix_at_the_earlier_message():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig, keep_context=False)

    found = consecutive(adapter.requests)
    assert found[1] is not None and found[1].section == "messages[0]"


async def test_every_request_of_a_session_carries_the_session_id_as_its_cache_key():
    adapter = session_script()
    rig = scripted_engine(adapter, [LateTool(), EchoTool()])

    await run_session(rig)

    assert {request.settings.cache_key for request in adapter.requests} == {"s1"}


async def test_tools_carried_over_from_an_earlier_turn_are_dropped_when_no_longer_available():
    adapter = ScriptedAdapter(text_step("ok"))
    rig = scripted_engine(adapter, [EchoTool()])
    gone = ToolSpec(name="gone", description="removed since", parameters={"type": "object"})
    echo = ToolSpec(name="echo", description="Echoes a message back.", parameters={"type": "object"})

    await drain(rig.engine.run(turn(
        deferred_tools={}, new_messages=(question(1),), tools_in_use=(echo, gone),
    )))

    assert tool_names(adapter.requests[0]) == ["echo"]

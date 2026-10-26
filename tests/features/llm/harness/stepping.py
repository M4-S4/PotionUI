from dataclasses import replace
from typing import AsyncIterator, Optional, Sequence, Tuple

from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.events import StreamEvent
from src.features.llm.harness.messages import Message
from src.features.llm.harness.profile import CapabilityProfile
from src.features.llm.harness.turn_state import TurnSnapshot, TurnState, TurnStatus
from tests.fixtures.llm_harness import SCRIPTED_ORIGIN, ScriptedAdapter, scripted_profile


def request_for(
    system: str = "",
    messages: Sequence[Message] = (),
    settings: Optional[Settings] = None,
    tools: Sequence[ToolSpec] = (),
) -> RenderRequest:
    return RenderRequest(
        config=None, system=system, messages=tuple(messages), tools=tuple(tools), settings=settings or Settings(),
    )


async def stream_events(
    adapter: ScriptedAdapter, profile: CapabilityProfile, request: RenderRequest,
) -> AsyncIterator[StreamEvent]:
    wire = adapter.render(request, profile)
    decoder = adapter.decoder(request, profile)
    async for chunk in adapter.send(wire, request):
        for event in decoder.feed(chunk):
            yield event
    for event in decoder.close():
        yield event


async def play_step(
    adapter: ScriptedAdapter,
    profile: CapabilityProfile,
    request: RenderRequest,
    message_id: str = "msg_1",
) -> Tuple[Message, StepAssembler]:
    assembler = StepAssembler(message_id, SCRIPTED_ORIGIN)
    async for event in stream_events(adapter, profile, request):
        assembler.feed(event)
    return assembler.message(), assembler


def scripted_complete(adapter: ScriptedAdapter, profile: CapabilityProfile):
    counter = iter(range(1, 100))

    async def complete(system, messages, settings, tools) -> Message:
        message, _ = await play_step(
            adapter, profile, request_for(system, messages, settings, tools), message_id=f"msg_{next(counter)}",
        )
        return message

    return complete


def turn_state(profile: Optional[CapabilityProfile] = None, **changes) -> TurnState:
    snapshot = TurnSnapshot(
        llm_config_id="cfg", system="", tools=(), settings=Settings(), profile=profile or scripted_profile(),
    )
    state = TurnState(turn_id="turn_1", session_id="session_1", status=TurnStatus.RUNNING, snapshot=snapshot, first_seq=1)
    return replace(state, **changes)

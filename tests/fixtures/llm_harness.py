import itertools
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.assembler import StepAssembler
from src.features.llm.harness.errors import HarnessError, InvalidRequest, WireFailure
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import Message, Origin
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource

SCRIPTED_ORIGIN = Origin(adapter="scripted", provider="scripted", model="scripted-model")

@dataclass(frozen=True)
class FailMidStream:
    events: Sequence[StreamEvent]
    error: HarnessError


Step = Union[Sequence[StreamEvent], HarnessError, FailMidStream]


def text_step(text: str, *, reasoning: str = "", usage: Optional[Usage] = None) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    if reasoning:
        events.append(ReasoningDelta(block=0, text=reasoning))
    events.append(TextDelta(block=1, text=text))
    events.append(UsageReported(usage or Usage(input_tokens=10, output_tokens=5, total_tokens=15)))
    events.append(Finish(reason=FinishReason.STOP, raw_reason="stop"))
    return events


def tool_step(*calls: Dict[str, Any], reasoning: str = "", text: str = "") -> List[StreamEvent]:
    events: List[StreamEvent] = []
    if reasoning:
        events.append(ReasoningDelta(block=0, text=reasoning))
    if text:
        events.append(TextDelta(block=1, text=text))
    for offset, call in enumerate(calls):
        block = 2 + offset
        events.append(ToolCallStart(block=block, call_id=call.get("id", f"call_{offset}"), name=call["name"]))
        events.append(ToolArgsDelta(block=block, text=json.dumps(call.get("args", {}))))
        events.append(ToolCallEnd(block=block))
    events.append(UsageReported(Usage(input_tokens=10, output_tokens=5, total_tokens=15)))
    events.append(Finish(reason=FinishReason.TOOL_CALLS, raw_reason="tool_calls"))
    return events


def scripted_profile(**fields: Any) -> CapabilityProfile:
    return CapabilityProfile(adapter="scripted", model="scripted-model").layered(
        [ProfileLayer(ProfileSource.OVERRIDE, fields)] if fields else []
    )


class _Replay:
    def __init__(self, events: Sequence[StreamEvent], eager: bool = False) -> None:
        self._events = list(events)
        self._eager = eager

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        if not self._eager:
            return []
        events, self._events = self._events, []
        return events

    def close(self) -> List[StreamEvent]:
        return self._events


class ScriptedAdapter:
    name = "scripted"

    def __init__(self, *steps: Step) -> None:
        self._steps = list(steps)
        self._cursor = itertools.count()
        self.requests: List[RenderRequest] = []
        self._current: Optional[Step] = None

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        self.requests.append(request)
        index = next(self._cursor)
        if index >= len(self._steps):
            raise AssertionError(f"the script has {len(self._steps)} step(s); step {index + 1} was requested")
        self._current = self._steps[index]
        return WireRequest(method="POST", url="scripted://step", headers={}, body={"step": index})

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> _Replay:
        if isinstance(self._current, HarnessError):
            return _Replay([])
        if isinstance(self._current, FailMidStream):
            return _Replay(self._current.events, eager=True)
        return _Replay(self._current or [])

    async def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]:
        if isinstance(self._current, HarnessError):
            raise self._current
        yield b""
        if isinstance(self._current, FailMidStream):
            raise self._current.error

    def classify_error(self, failure: WireFailure) -> HarnessError:
        return InvalidRequest(f"scripted failure {failure.status}", status=failure.status)

    async def discover(self, config: Any) -> ProfileLayer:
        return ProfileLayer(ProfileSource.DISCOVERED, {})

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> list:
        return []

    @property
    def exhausted(self) -> bool:
        return len(self.requests) == len(self._steps)


def decode_chunks(decoder: Any, chunks: Sequence[bytes]) -> List[StreamEvent]:
    events: List[StreamEvent] = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.close())
    return events


def assemble(events: Sequence[StreamEvent], message_id: str = "msg_1", origin: Origin = SCRIPTED_ORIGIN) -> Message:
    assembler = StepAssembler(message_id, origin)
    assembler.feed_all(events)
    return assembler.message()

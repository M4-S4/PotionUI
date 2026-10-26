import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.features.llm.harness.errors import MalformedStream
from src.features.llm.harness.events import (
    Finish,
    OpaqueDelta,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import Message, Origin, Part, ReasoningPart, Role, TextPart, ToolCallPart


@dataclass
class _Block:
    kind: str
    text: List[str] = field(default_factory=list)
    summary: List[str] = field(default_factory=list)
    phase: Optional[str] = None
    redacted: bool = False
    opaque: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    name: str = ""
    wire_id: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    args_given: bool = False
    args_deltas: bool = False
    ended: bool = False


def parse_args(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw is None or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except ValueError:
        return None
    return decoded if isinstance(decoded, dict) else None


class StepAssembler:
    def __init__(self, message_id: str, origin: Origin) -> None:
        self._message_id = message_id
        self._origin = origin
        self._blocks: Dict[int, _Block] = {}
        self.usage: Optional[Usage] = None
        self.finish: Optional[Finish] = None

    @property
    def complete(self) -> bool:
        return self.finish is not None

    def feed_all(self, events: Iterable[StreamEvent]) -> None:
        for event in events:
            self.feed(event)

    def feed(self, event: StreamEvent) -> None:
        if self.finish is not None:
            raise MalformedStream(f"{type(event).__name__} after finish")
        if isinstance(event, TextDelta):
            block = self._block(event.block, "text")
            block.text.append(event.text)
            block.phase = block.phase or event.phase
        elif isinstance(event, ReasoningDelta):
            block = self._block(event.block, "reasoning")
            (block.summary if event.summary else block.text).append(event.text)
            block.redacted = block.redacted or event.redacted
        elif isinstance(event, OpaqueDelta):
            if event.block not in self._blocks:
                raise MalformedStream(f"opaque data for unknown block {event.block}")
            self._blocks[event.block].opaque[event.key] = event.value
        elif isinstance(event, ToolCallStart):
            if event.block in self._blocks:
                raise MalformedStream(f"block {event.block} reopened as a tool call")
            self._blocks[event.block] = _Block(
                kind="tool_call", call_id=event.call_id, name=event.name, wire_id=event.wire_id,
            )
        elif isinstance(event, ToolArgsDelta):
            block = self._tool_block(event.block)
            block.text.append(event.text)
            block.args_deltas = True
        elif isinstance(event, ToolCallEnd):
            block = self._tool_block(event.block)
            block.ended = True
            if event.args is not None:
                block.args = dict(event.args)
                block.args_given = True
        elif isinstance(event, UsageReported):
            self.usage = event.usage if self.usage is None else self.usage.merged(event.usage)
        elif isinstance(event, Finish):
            self.finish = event
        else:
            raise MalformedStream(f"unknown event {type(event).__name__}")

    def _block(self, index: int, kind: str) -> _Block:
        block = self._blocks.setdefault(index, _Block(kind=kind))
        if block.kind != kind:
            raise MalformedStream(f"block {index} is {block.kind}, got {kind}")
        return block

    def _tool_block(self, index: int) -> _Block:
        block = self._blocks.get(index)
        if block is None or block.kind != "tool_call":
            raise MalformedStream(f"tool arguments for unknown block {index}")
        return block

    def message(self) -> Message:
        parts: List[Part] = []
        for index in sorted(self._blocks):
            part = self._part(self._blocks[index])
            if part is not None:
                parts.append(part)
        raw = self.finish.raw if self.finish is not None else None
        return Message(id=self._message_id, role=Role.ASSISTANT, parts=tuple(parts), origin=self._origin, raw=raw)

    @staticmethod
    def _part(block: _Block) -> Optional[Part]:
        opaque = dict(block.opaque) or None
        if block.kind == "text":
            text = "".join(block.text)
            return TextPart(text=text, phase=block.phase) if text else None
        if block.kind == "reasoning":
            return ReasoningPart(
                text="".join(block.text), summary="".join(block.summary), redacted=block.redacted, opaque=opaque,
            )
        args_raw = "".join(block.text) if block.args_deltas else None
        if block.args_given:
            args = block.args
        elif block.ended:
            args = parse_args(args_raw)
        else:
            args = None
        return ToolCallPart(
            call_id=block.call_id, name=block.name, args=args, args_raw=args_raw, wire_id=block.wire_id, opaque=opaque,
        )

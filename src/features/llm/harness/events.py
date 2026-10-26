from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Any, ClassVar, Dict, Optional, Union


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    REFUSAL = "refusal"
    PAUSE = "pause"
    CONTEXT_WINDOW = "context_window"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Usage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    cost: Optional[float] = None
    raw: Any = None

    def merged(self, later: "Usage") -> "Usage":
        return Usage(**{
            item.name: getattr(later, item.name) if getattr(later, item.name) is not None else getattr(self, item.name)
            for item in fields(self)
        })


@dataclass(frozen=True)
class TextDelta:
    TYPE: ClassVar[str] = "text_delta"
    block: int
    text: str
    phase: Optional[str] = None


@dataclass(frozen=True)
class ReasoningDelta:
    TYPE: ClassVar[str] = "reasoning_delta"
    block: int
    text: str
    summary: bool = False
    redacted: bool = False


@dataclass(frozen=True)
class OpaqueDelta:
    TYPE: ClassVar[str] = "opaque_delta"
    block: int
    key: str
    value: Any


@dataclass(frozen=True)
class ToolCallStart:
    TYPE: ClassVar[str] = "tool_call_start"
    block: int
    call_id: str
    name: str
    wire_id: Optional[str] = None


@dataclass(frozen=True)
class ToolArgsDelta:
    TYPE: ClassVar[str] = "tool_args_delta"
    block: int
    text: str


@dataclass(frozen=True)
class ToolCallEnd:
    TYPE: ClassVar[str] = "tool_call_end"
    block: int
    args: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class UsageReported:
    TYPE: ClassVar[str] = "usage"
    usage: Usage


@dataclass(frozen=True)
class Finish:
    TYPE: ClassVar[str] = "finish"
    reason: FinishReason
    raw_reason: Optional[str] = None
    raw: Any = None


StreamEvent = Union[
    TextDelta,
    ReasoningDelta,
    OpaqueDelta,
    ToolCallStart,
    ToolArgsDelta,
    ToolCallEnd,
    UsageReported,
    Finish,
]

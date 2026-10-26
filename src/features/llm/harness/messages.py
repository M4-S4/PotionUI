import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Dict, Optional, Tuple, Union


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class TextPart:
    TYPE: ClassVar[str] = "text"
    text: str
    phase: Optional[str] = None


@dataclass(frozen=True)
class ReasoningPart:
    TYPE: ClassVar[str] = "reasoning"
    text: str = ""
    summary: str = ""
    redacted: bool = False
    opaque: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


@dataclass(frozen=True)
class ToolCallPart:
    TYPE: ClassVar[str] = "tool_call"
    call_id: str
    name: str
    args: Optional[Dict[str, Any]]
    args_raw: Optional[str] = None
    wire_id: Optional[str] = None
    opaque: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class ImagePart:
    TYPE: ClassVar[str] = "image"
    media_type: str
    data: Optional[str] = None
    ref: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.ref is None):
            raise ValueError("ImagePart needs exactly one of data or ref")


@dataclass(frozen=True)
class FilePart:
    TYPE: ClassVar[str] = "file"
    media_type: str
    name: Optional[str] = None
    data: Optional[str] = None
    ref: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.ref is None):
            raise ValueError("FilePart needs exactly one of data or ref")


@dataclass(frozen=True)
class JsonPart:
    TYPE: ClassVar[str] = "json"
    value: Any


ResultContent = Union[TextPart, ImagePart, JsonPart]


@dataclass(frozen=True)
class ToolResultPart:
    TYPE: ClassVar[str] = "tool_result"
    call_id: str
    name: str
    content: Tuple[ResultContent, ...]
    is_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", tuple(self.content))
        for item in self.content:
            if not isinstance(item, (TextPart, ImagePart, JsonPart)):
                raise ValueError(f"tool result content cannot hold {type(item).__name__}")

    def text(self) -> str:
        return "".join(item.text for item in self.content if isinstance(item, TextPart))


@dataclass(frozen=True)
class ContextPart:
    TYPE: ClassVar[str] = "context"
    kind: str
    text: str


@dataclass(frozen=True)
class CompactionPart:
    TYPE: ClassVar[str] = "compaction"
    summary: str
    opaque: Optional[Dict[str, Any]] = None


Part = Union[
    TextPart,
    ReasoningPart,
    ToolCallPart,
    ToolResultPart,
    ImagePart,
    FilePart,
    ContextPart,
    CompactionPart,
]

_ALLOWED_PARTS = {
    Role.SYSTEM: (TextPart,),
    Role.USER: (TextPart, ImagePart, FilePart, ContextPart, CompactionPart),
    Role.ASSISTANT: (TextPart, ReasoningPart, ToolCallPart, ImagePart, CompactionPart),
    Role.TOOL: (ToolResultPart,),
}


@dataclass(frozen=True)
class Origin:
    adapter: str
    provider: str
    model: str
    credential: Optional[str] = None


LEGACY_ORIGIN = Origin(adapter="legacy", provider="legacy", model="legacy")


def credential_fingerprint(base_url: Optional[str], api_key: Optional[str]) -> str:
    material = f"{(base_url or '').strip().rstrip('/')}\n{api_key or ''}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Message:
    id: str
    role: Role
    parts: Tuple[Part, ...]
    origin: Optional[Origin] = None
    raw: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", Role(self.role))
        object.__setattr__(self, "parts", tuple(self.parts))
        allowed = _ALLOWED_PARTS[self.role]
        for part in self.parts:
            if not isinstance(part, allowed):
                raise ValueError(f"a {self.role.value} message cannot hold {type(part).__name__}")
        if self.role is Role.ASSISTANT and self.origin is None:
            raise ValueError("an assistant message needs an origin")

    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))

    def tool_calls(self) -> Tuple[ToolCallPart, ...]:
        return tuple(part for part in self.parts if isinstance(part, ToolCallPart))

    def tool_results(self) -> Tuple[ToolResultPart, ...]:
        return tuple(part for part in self.parts if isinstance(part, ToolResultPart))

    def reasoning(self) -> Tuple[ReasoningPart, ...]:
        return tuple(part for part in self.parts if isinstance(part, ReasoningPart))

    def replays_raw_to(self, target: Origin) -> bool:
        return self.raw is not None and self.origin == target


def message(id: str, role: Union[Role, str], *parts: Part, origin: Optional[Origin] = None, raw: Any = None) -> Message:
    return Message(id=id, role=Role(role), parts=tuple(parts), origin=origin, raw=raw)


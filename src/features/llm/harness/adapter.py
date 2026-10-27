from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, List, Mapping, Optional, Protocol, Tuple, Union, runtime_checkable

from src.features.llm.harness.errors import HarnessError, WireFailure
from src.features.llm.harness.events import StreamEvent
from src.features.llm.harness.messages import Message
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer

if TYPE_CHECKING:
    from src.features.llm.model_listing import DiscoveredModel
    from src.features.llm.repository import LLMConfig


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Any
    strict: bool = False

    @classmethod
    def from_schema(cls, schema: Mapping[str, Any]) -> ToolSpec:
        function = schema["function"]
        return cls(
            name=function["name"],
            description=function.get("description", ""),
            parameters=function.get("parameters", {}),
            strict=bool(function.get("strict", False)),
        )

    def function_schema(self) -> dict:
        function = {"name": self.name, "description": self.description, "parameters": self.parameters}
        if self.strict:
            function["strict"] = True
        return {"type": "function", "function": function}


@dataclass(frozen=True)
class ToolChoice:
    mode: str
    name: Optional[str] = None


@dataclass(frozen=True)
class Settings:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning: Union[bool, str, None] = None
    tool_choice: Optional[ToolChoice] = None
    response_schema: Any = None
    cache_key: Optional[str] = None
    cache_ttl: Optional[str] = None
    keep_model_loaded: bool = False

    @classmethod
    def from_options(cls, options: Optional[Mapping[str, Any]]) -> Settings:
        options = options or {}
        return cls(
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            top_k=options.get("top_k"),
            min_p=options.get("min_p"),
            max_tokens=options.get("max_tokens"),
            reasoning=_reasoning_from_options(options),
        )


EFFORT_OFF = "none"


def _reasoning_from_options(options: Mapping[str, Any]) -> Union[bool, str, None]:
    effort = options.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        return False if effort == EFFORT_OFF else effort
    think = options.get("think")
    if isinstance(think, bool) or (isinstance(think, str) and think.strip()):
        return think
    thinking = options.get("thinking")
    return thinking if isinstance(thinking, bool) else None


@dataclass(frozen=True)
class RenderRequest:
    config: LLMConfig
    system: str
    messages: Tuple[Message, ...]
    tools: Tuple[ToolSpec, ...] = ()
    settings: Settings = field(default_factory=Settings)
    stream: bool = True


@dataclass(frozen=True)
class WireRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    body: Any
    stream: bool = True

    def content(self) -> bytes:
        return json.dumps(self.body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class StreamDecoder(Protocol):
    def feed(self, chunk: bytes) -> List[StreamEvent]: ...

    def close(self) -> List[StreamEvent]: ...


@runtime_checkable
class Adapter(Protocol):
    name: str

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest: ...

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> StreamDecoder: ...

    def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]: ...

    def classify_error(self, failure: WireFailure) -> HarnessError: ...

    async def discover(self, config: LLMConfig) -> ProfileLayer: ...

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]: ...

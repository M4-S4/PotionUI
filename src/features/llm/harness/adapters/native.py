import codecs
import weakref
from contextlib import aclosing
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from src.features.llm.harness import prompt_tools
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolSpec, WireRequest
from src.features.llm.harness.blocks import REASONING_BLOCK, TEXT_BLOCK
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import HarnessError, InvalidRequest, WireFailure
from src.features.llm.harness.events import (
    Finish,
    FinishReason,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import Message, Role
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.harness.rendering import image_data, last_user_index, selected_image
from src.features.llm.harness.think_tags import TEXT, THINKING, ThinkTagStage
from src.features.llm.model_listing import DiscoveredModel, DiscoveredWindow, sorted_models
from src.features.llm.native.runtime import NativeOutOfMemory, NativeRuntime, RunStats
from src.features.llm.repository import LLMConfig
from src.platform.filesystem.model_roots import ModelRootResolver
from src.platform.runtime.model_lifecycle.lifecycle import ModelLifecycle

_OFF_LEVELS = frozenset({"none", "off", "false", "disabled"})


def render_chat(
    system: Optional[str],
    messages: Sequence[Message],
    tools: Sequence[ToolSpec],
    *,
    with_image: bool = True,
) -> List[Dict[str, Any]]:
    head = (system or "") + prompt_tools.render_inline_tools_text(tools)
    rendered: List[Dict[str, Any]] = []
    if head:
        rendered.append({"role": "system", "content": head})
    for message in messages:
        if message.role is Role.SYSTEM:
            rendered.append({"role": "system", "content": message.text()})
        elif message.role is Role.USER:
            rendered.append({"role": "user", "content": user_text(message)})
        elif message.role is Role.ASSISTANT:
            rendered.append({
                "role": "assistant",
                "content": prompt_tools.assistant_text(message.text(), message.tool_calls()),
            })
        else:
            for result in message.tool_results():
                rendered.append({"role": "user", "content": prompt_tools.result_text(result.name, result.text())})
    if with_image:
        image = selected_image(messages)
        index = last_user_index(rendered)
        if image is not None and index is not None:
            rendered[index]["images"] = [image_data(image)]
    return rendered


def requested_thinking(settings: Settings, config: LLMConfig) -> Optional[bool]:
    value = settings.reasoning
    if value is None:
        value = (config.provider_options or {}).get("thinking")
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() not in _OFF_LEVELS
    return bool(value)


def render_sampling(request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
    config = request.config
    settings = request.settings
    options = config.provider_options or {}

    def pick(value: Any, *fallbacks: Any) -> Any:
        for candidate in (value, *fallbacks):
            if candidate is not None:
                return candidate
        return None

    sampling = {
        "temperature": pick(settings.temperature, getattr(config, "temperature", None)),
        "top_p": pick(settings.top_p, options.get("top_p")),
        "top_k": pick(settings.top_k, options.get("top_k")),
        "min_p": pick(settings.min_p, options.get("min_p")),
        "repetition_penalty": options.get("repetition_penalty"),
        "max_new_tokens": pick(settings.max_tokens, getattr(config, "max_tokens", None), profile.max_output_tokens),
    }
    return {key: value for key, value in sampling.items() if value is not None}


class NativeDecoder:
    def __init__(self, stats: RunStats) -> None:
        self._stats = stats
        self._bytes = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._think = ThinkTagStage()

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        return self._events(self._think.feed(self._bytes.decode(chunk)))

    def close(self) -> List[StreamEvent]:
        events = self._events(self._think.feed(self._bytes.decode(b"", final=True)))
        events.extend(self._events(self._think.finish()))
        stats = self._stats
        if stats.prompt_tokens is not None or stats.completion_tokens is not None:
            prompt, completion = stats.prompt_tokens, stats.completion_tokens
            events.append(UsageReported(Usage(
                input_tokens=prompt,
                output_tokens=completion,
                total_tokens=(prompt or 0) + (completion or 0),
                raw={"thinking_mode": stats.thinking_mode} if stats.thinking_mode else None,
            )))
        reason = {"stop": FinishReason.STOP, "length": FinishReason.LENGTH}.get(stats.finish, FinishReason.UNKNOWN)
        events.append(Finish(reason=reason, raw_reason=stats.finish))
        return events

    @staticmethod
    def _events(pieces) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for piece in pieces:
            if piece.kind == THINKING:
                events.append(ReasoningDelta(block=REASONING_BLOCK, text=piece.text))
            elif piece.kind == TEXT:
                events.append(TextDelta(block=TEXT_BLOCK, text=piece.text))
        return events


class NativeAdapter:
    name = "native"

    def __init__(
        self,
        model_lifecycle: Optional[ModelLifecycle] = None,
        model_roots: Optional[ModelRootResolver] = None,
    ) -> None:
        self.runtime = NativeRuntime(model_lifecycle, model_roots)
        self._runs: "weakref.WeakValueDictionary[int, RunStats]" = weakref.WeakValueDictionary()

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        return WireRequest(
            method="POST",
            url=f"native://{request.config.model}",
            headers={},
            body={
                "model": request.config.model,
                "messages": render_chat(request.system, request.messages, request.tools),
                "sampling": render_sampling(request, profile),
                "enable_thinking": requested_thinking(request.settings, request.config),
            },
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> NativeDecoder:
        stats = RunStats()
        self._runs[id(request)] = stats
        return NativeDecoder(stats)

    async def send(self, wire: WireRequest, request: RenderRequest) -> AsyncIterator[bytes]:
        stats = self._runs.pop(id(request), None) or RunStats()
        body = wire.body
        try:
            async with aclosing(self.runtime.stream(
                request.config, body["messages"], body["sampling"], body["enable_thinking"], stats,
            )) as texts:
                async for text in texts:
                    yield text.encode("utf-8")
        except (NativeOutOfMemory, ValueError) as exc:
            raise self.classify_error(WireFailure(status=None, in_stream=True, cause=exc)) from exc

    def classify_error(self, failure: WireFailure) -> HarnessError:
        cause = failure.cause
        message = str(cause) if cause is not None else "native generation failed"
        if isinstance(cause, NativeOutOfMemory):
            return HarnessError(message, code="out_of_memory")
        if isinstance(cause, ValueError):
            return InvalidRequest(message)
        return HarnessError(message, code=type(cause).__name__ if cause is not None else None)

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        fields: Dict[str, Any] = {"prompt_tools": True, **self.runtime.probe(config.model)}
        return ProfileLayer(ProfileSource.DISCOVERED, fields)

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        return sorted_models([
            DiscoveredModel(id=entry["name"], details={"model_type": entry["model_type"], "vision": entry["vision"]})
            for entry in self.runtime.list_checkpoints()
        ])

    async def discover_context_window(
        self, base_url: str, api_key: Optional[str], model: str,
    ) -> Optional[DiscoveredWindow]:
        return await self.runtime.discover_context_window(base_url, api_key, model)

    async def release(self, config: LLMConfig) -> bool:
        return await self.runtime.release(config)

    def token_counter(self, config: LLMConfig):
        return self.runtime.token_counter(config)

    def messages_token_counter(self, config: LLMConfig):
        count = self.runtime.prompt_token_counter(config, requested_thinking(Settings(), config))
        if count is None:
            return None

        def _count(system_message: Optional[str], messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]] = None) -> int:
            specs = [ToolSpec.from_schema(schema) for schema in tools or ()]
            return count(render_chat(system_message, messages, specs, with_image=False))

        return _count

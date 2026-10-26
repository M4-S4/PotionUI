import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from src.features.llm.harness import finish as finish_reasons
from src.features.llm.harness import prompt_tools, tool_ids
from src.features.llm.harness.adapter import RenderRequest, WireRequest
from src.features.llm.harness.context import user_text
from src.features.llm.harness.errors import HarnessError, WireFailure
from src.features.llm.harness.events import (
    Finish,
    ReasoningDelta,
    StreamEvent,
    TextDelta,
    ToolArgsDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    UsageReported,
)
from src.features.llm.harness.messages import ImagePart, Message, Role
from src.features.llm.harness.ndjson import NDJSONFramer
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource
from src.features.llm.harness.rendering import bearer_headers, image_data, reasoning_text, result_text
from src.features.llm.harness.think_tags import CLOSED, THINKING, ThinkTagStage
from src.features.llm.harness.tool_args import arguments_object, object_from_wire
from src.features.llm.harness.transport import HttpAdapter
from src.features.llm.model_listing import (
    CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
    DiscoveredModel,
    DiscoveredWindow,
    ModelListingError,
    fetch_json,
    positive_int,
    sorted_models,
)
from src.features.llm.repository import LLMConfig

OPTION_KEYS = (
    "num_gpu",
    "num_thread",
    "num_batch",
    "num_keep",
    "seed",
    "top_k",
    "top_p",
    "min_p",
    "tfs_z",
    "typical_p",
    "repeat_penalty",
    "repeat_last_n",
    "presence_penalty",
    "frequency_penalty",
    "mirostat",
    "mirostat_tau",
    "mirostat_eta",
    "stop",
)

KEEP_ALIVE_LOADED = "5m"
TRAINED_WINDOW_CAP_TOKENS = 32768
_MODELFILE_NUM_CTX = re.compile(r"^\s*num_ctx\s+(\d+)\s*$", re.MULTILINE)


def parse_show_window(payload: Any) -> Optional[DiscoveredWindow]:
    if not isinstance(payload, dict):
        return None
    parameters = payload.get("parameters")
    if isinstance(parameters, str):
        match = _MODELFILE_NUM_CTX.search(parameters)
        runtime = positive_int(int(match.group(1))) if match else None
        if runtime:
            return DiscoveredWindow(tokens=runtime, kind="runtime")
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        for key, value in model_info.items():
            if isinstance(key, str) and key.endswith(".context_length"):
                trained = positive_int(value)
                if trained:
                    return DiscoveredWindow(tokens=trained, kind="trained")
    return None


def render_options(request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
    config = request.config
    settings = request.settings
    configured = config.provider_options or {}
    options: Dict[str, Any] = {
        "temperature": config.temperature,
        "num_predict": config.max_tokens,
    }
    if profile.context_window:
        options["num_ctx"] = profile.context_window
    for key in OPTION_KEYS:
        if key in configured:
            options[key] = configured[key]
    if settings.temperature is not None:
        options["temperature"] = settings.temperature
    if settings.top_p is not None:
        options["top_p"] = settings.top_p
    if settings.top_k is not None:
        options["top_k"] = settings.top_k
    if settings.min_p is not None:
        options["min_p"] = settings.min_p
    if settings.max_tokens is not None:
        options["num_predict"] = settings.max_tokens
    return options


def render_think(request: RenderRequest) -> Any:
    if request.settings.reasoning is not None:
        return request.settings.reasoning
    configured = (request.config.provider_options or {}).get("think")
    return configured if configured is not None else False


def render_keep_alive(request: RenderRequest) -> Any:
    configured = (request.config.provider_options or {}).get("keep_alive")
    if request.settings.keep_model_loaded:
        return configured if configured not in (None, 0, "0", "0s") else KEEP_ALIVE_LOADED
    return configured if configured is not None else 0


def render_messages(request: RenderRequest, profile: CapabilityProfile, system: str, inline_tools: bool) -> List[Dict[str, Any]]:
    ids = tool_ids.wire_ids(request.messages, profile.tool_id_rule)
    rendered: List[Dict[str, Any]] = []
    if system:
        rendered.append({"role": "system", "content": system})
    for message in request.messages:
        if message.role is Role.SYSTEM:
            rendered.append({"role": "system", "content": message.text()})
        elif message.role is Role.USER:
            entry: Dict[str, Any] = {"role": "user", "content": user_text(message)}
            images = [image_data(part) for part in message.parts if isinstance(part, ImagePart)]
            if images:
                entry["images"] = images
            rendered.append(entry)
        elif message.role is Role.ASSISTANT:
            rendered.append(_assistant(message, ids, inline_tools))
        else:
            for result in message.tool_results():
                if inline_tools:
                    rendered.append({
                        "role": "user",
                        "content": prompt_tools.result_text(result.name, result_text(result)),
                    })
                else:
                    rendered.append({
                        "role": "tool",
                        "content": result_text(result),
                        "tool_call_id": ids.get(result.call_id, result.call_id),
                    })
    return rendered


def _assistant(message: Message, ids: Dict[str, str], inline_tools: bool) -> Dict[str, Any]:
    calls = message.tool_calls()
    if calls and inline_tools:
        entry: Dict[str, Any] = {"role": "assistant", "content": prompt_tools.assistant_text(message.text(), calls)}
    else:
        entry = {"role": "assistant", "content": message.text()}
        if calls:
            entry["tool_calls"] = [
                {
                    "id": ids[call.call_id],
                    "type": "function",
                    "function": {"name": call.name, "arguments": arguments_object(call)},
                }
                for call in calls
            ]
    thinking = reasoning_text(message)
    if thinking:
        entry["thinking"] = thinking
    return entry


def usage_from(data: Dict[str, Any]) -> Usage:
    def count(key: str) -> Optional[int]:
        value = data.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    prompt, completion = count("prompt_eval_count"), count("eval_count")
    total = None if prompt is None and completion is None else (prompt or 0) + (completion or 0)
    raw = {key: value for key, value in data.items() if key not in ("message", "done")}
    return Usage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=total,
        cached_input_tokens=count("prompt_eval_cached_count"),
        raw=raw,
    )


class OllamaChatDecoder:
    def __init__(
        self,
        seed: str,
        inline_tools: bool,
        classify: Callable[[WireFailure], HarnessError],
        label: str = "ollama-chat",
    ) -> None:
        self._seed = seed
        self._classify = classify
        self._framer = NDJSONFramer(label)
        self._think = ThinkTagStage()
        self._inline = prompt_tools.PromptToolStage() if inline_tools else None
        self._next_block = 0
        self._kind: Optional[str] = None
        self._block = 0
        self._calls = 0
        self._finished = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        return self._objects(self._framer.feed(chunk))

    def close(self) -> List[StreamEvent]:
        events = self._objects(self._framer.close())
        if not self._finished:
            events.extend(self._finish(None))
        return events

    def _objects(self, values: List[Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for data in values:
            if self._finished:
                break
            if isinstance(data, dict):
                events.extend(self._object(data))
        return events

    def _object(self, data: Dict[str, Any]) -> List[StreamEvent]:
        error = data.get("error")
        if error and not data.get("message"):
            text = error if isinstance(error, str) else json.dumps(error)
            status = 404 if "not found" in text.lower() else 500
            raise self._classify(WireFailure(status=status, body=json.dumps(data).encode("utf-8"), in_stream=True))
        events: List[StreamEvent] = []
        message = data.get("message")
        if isinstance(message, dict):
            thinking = message.get("thinking")
            if isinstance(thinking, str) and thinking:
                events.append(ReasoningDelta(block=self._use("reasoning"), text=thinking))
            content = message.get("content")
            if isinstance(content, str) and content:
                events.extend(self._text(self._think.feed(content)))
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    events.extend(self._native_call(call))
        if data.get("done"):
            events.extend(self._finish(data.get("done_reason"), usage_from(data)))
        return events

    def _use(self, kind: str) -> int:
        if kind != self._kind:
            self._kind = kind
            self._block = self._next_block
            self._next_block += 1
        return self._block

    def _text(self, pieces) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for piece in pieces:
            if piece.kind == THINKING:
                events.append(ReasoningDelta(block=self._use("reasoning"), text=piece.text))
            elif piece.kind == CLOSED:
                self._kind = None
            elif self._inline is None:
                events.append(TextDelta(block=self._use("text"), text=piece.text))
            else:
                events.extend(self._inline_items(self._inline.feed(piece.text)))
        return events

    def _inline_items(self, items) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for item in items:
            if isinstance(item, prompt_tools.TextItem):
                events.append(TextDelta(block=self._use("text"), text=item.text))
            else:
                events.extend(self._emit_call(None, item.name, item.args, item.raw if item.args is None else None))
        return events

    def _native_call(self, call: Any) -> List[StreamEvent]:
        if not isinstance(call, dict):
            return []
        function = call.get("function")
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments")
        wire_id = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else None
        args = object_from_wire(arguments)
        raw = arguments if isinstance(arguments, str) and args is None else None
        return self._emit_call(wire_id, str(function.get("name") or ""), args, raw)

    def _emit_call(
        self, wire_id: Optional[str], name: str, args: Optional[Dict[str, Any]], raw: Optional[str],
    ) -> List[StreamEvent]:
        block = self._next_block
        self._next_block += 1
        self._kind = "tool"
        call_id = wire_id or tool_ids.mint_call_id(self._seed, self._calls)
        self._calls += 1
        events: List[StreamEvent] = [ToolCallStart(block=block, call_id=call_id, name=name, wire_id=wire_id)]
        if raw is not None:
            events.append(ToolArgsDelta(block=block, text=raw))
            events.append(ToolCallEnd(block=block))
        else:
            events.append(ToolCallEnd(block=block, args=args))
        return events

    def _finish(self, raw_reason: Optional[str], usage: Optional[Usage] = None) -> List[StreamEvent]:
        self._finished = True
        events = self._text(self._think.finish())
        if self._inline is not None:
            events.extend(self._inline_items(self._inline.finish()))
        if usage is not None:
            events.append(UsageReported(usage))
        reason = finish_reasons.with_tool_calls(finish_reasons.from_ollama(raw_reason), self._calls > 0)
        events.append(Finish(reason=reason, raw_reason=raw_reason))
        return events


class OllamaChatAdapter(HttpAdapter):
    name = "ollama-chat"
    display_name = "Ollama"

    def build_body(self, request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
        inline_tools = bool(request.tools) and profile.prompt_tools
        system = request.system
        if inline_tools:
            system = (system or "") + prompt_tools.render_tools_text(request.tools)
        body: Dict[str, Any] = {
            "model": request.config.model,
            "messages": render_messages(request, profile, system, inline_tools),
            "stream": True,
            "keep_alive": render_keep_alive(request),
            "think": render_think(request),
            "options": render_options(request, profile),
        }
        if request.tools and profile.tools and not inline_tools:
            body["tools"] = [tool.function_schema() for tool in request.tools]
        return body

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        return WireRequest(
            method="POST",
            url=f"{request.config.base_url.rstrip('/')}/api/chat",
            headers={"Content-Type": "application/json", **bearer_headers(request.config.api_key)},
            body=self.build_body(request, profile),
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> OllamaChatDecoder:
        return OllamaChatDecoder(
            seed=tool_ids.seed_for(request.messages),
            inline_tools=bool(request.tools) and profile.prompt_tools,
            classify=self.classify_error,
            label=self.name,
        )

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        payload = await fetch_json(self.display_name, base_url, "/api/tags", bearer_headers(api_key))
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelListingError(f"{self.display_name} returned no model list.")
        models = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model")
            if not name:
                continue
            raw_details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            details = {
                key: raw_details[source]
                for key, source in (
                    ("parameter_size", "parameter_size"),
                    ("quantization", "quantization_level"),
                    ("family", "family"),
                )
                if raw_details.get(source)
            }
            size = entry.get("size")
            models.append(DiscoveredModel(
                id=str(name),
                size=size if isinstance(size, int) else None,
                modified_at=entry.get("modified_at") if isinstance(entry.get("modified_at"), str) else None,
                details=details or None,
            ))
        return sorted_models(models)

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        fields: Dict[str, Any] = {}
        try:
            payload = await fetch_json(
                self.display_name,
                config.base_url,
                "/api/show",
                bearer_headers(config.api_key),
                body={"model": config.model},
                timeout=CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
            )
        except Exception:
            logging.debug("%s discovery failed for %s", self.display_name, config.model, exc_info=True)
            return ProfileLayer(ProfileSource.DISCOVERED, fields)
        window = parse_show_window(payload)
        if window:
            fields["context_window"] = (
                min(window.tokens, TRAINED_WINDOW_CAP_TOKENS) if window.kind == "trained" else window.tokens
            )
        capabilities = payload.get("capabilities") if isinstance(payload, dict) else None
        if isinstance(capabilities, list):
            names = {item for item in capabilities if isinstance(item, str)}
            fields["tools"] = "tools" in names
            fields["vision"] = "vision" in names
            fields["reasoning"] = "toggle" if "thinking" in names else "none"
        return ProfileLayer(ProfileSource.DISCOVERED, fields)

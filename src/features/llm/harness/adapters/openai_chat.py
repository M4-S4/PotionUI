import json
import logging
from typing import Any, Callable, Dict, List, Optional

from src.features.llm.harness import finish as finish_reasons
from src.features.llm.harness import tool_ids
from src.features.llm.harness.adapter import RenderRequest, Settings, ToolChoice, WireRequest
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
from src.features.llm.harness.rendering import bearer_headers, image_data_uri, reasoning_text, result_text
from src.features.llm.harness.profile import CapabilityProfile, ProfileLayer, ProfileSource, ReasoningKind, ToolIdRule
from src.features.llm.harness.sse import DONE_SENTINEL, SSEFramer, SSERecord
from src.features.llm.harness.think_tags import CLOSED, TEXT, THINKING, ThinkTagStage
from src.features.llm.harness.tool_args import arguments_text
from src.features.llm.harness.transport import HttpAdapter
from src.features.llm.model_listing import (
    CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
    DiscoveredModel,
    ModelListingError,
    fetch_json,
    positive_int,
    sorted_models,
)
from src.features.llm.repository import LLMConfig

DEFAULT_REASONING_FIELD = "reasoning_content"
REASONING_FIELDS = ("reasoning_content", "reasoning")
CONTEXT_WINDOW_KEYS = ("context_length", "max_model_len", "context_window")

PROVIDER_OPTION_KEYS = (
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "reasoning_effort",
    "parallel_tool_calls",
)

TOGGLE_REASONING_EFFORT = "reasoning_effort"
TOGGLE_THINKING = "thinking"
TOGGLE_TEMPLATE_KWARGS = "chat_template_kwargs.enable_thinking"


def auth_headers(config: LLMConfig) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def reasoning_active(settings: Settings, profile: CapabilityProfile) -> bool:
    if settings.reasoning is None:
        return profile.reasoning is ReasoningKind.ALWAYS_ON
    return settings.reasoning is not False


def sampling_params(request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
    config = request.config
    settings = request.settings
    options = config.provider_options or {}
    params: Dict[str, Any] = {
        "temperature": config.temperature,
        "max_completion_tokens": config.max_tokens,
    }
    for key in PROVIDER_OPTION_KEYS:
        if options.get(key) is not None:
            params[key] = options[key]
    if settings.temperature is not None:
        params["temperature"] = settings.temperature
    if settings.top_p is not None:
        params["top_p"] = settings.top_p
    if settings.max_tokens is not None:
        params["max_completion_tokens"] = settings.max_tokens
    dropped = set(profile.sampling_rejected)
    if reasoning_active(settings, profile):
        dropped |= set(profile.sampling_ignored_with_reasoning)
    return {key: value for key, value in params.items() if key not in dropped}


def user_content(message: Message) -> Any:
    text = user_text(message)
    images = [part for part in message.parts if isinstance(part, ImagePart)]
    if not images:
        return text
    return [
        *({"type": "image_url", "image_url": {"url": image_data_uri(image)}} for image in images),
        {"type": "text", "text": text},
    ]


def render_messages(request: RenderRequest, profile: CapabilityProfile) -> List[Dict[str, Any]]:
    ids = tool_ids.wire_ids(request.messages, profile.tool_id_rule)
    field = profile.reasoning_field or DEFAULT_REASONING_FIELD
    rendered: List[Dict[str, Any]] = [{"role": "system", "content": request.system}]
    for message in request.messages:
        if message.role is Role.SYSTEM:
            rendered.append({"role": "system", "content": message.text()})
        elif message.role is Role.USER:
            rendered.append({"role": "user", "content": user_content(message)})
        elif message.role is Role.ASSISTANT:
            entry: Dict[str, Any] = {"role": "assistant", "content": message.text()}
            calls = message.tool_calls()
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": ids[call.call_id],
                        "type": "function",
                        "function": {"name": call.name, "arguments": arguments_text(call)},
                    }
                    for call in calls
                ]
            thinking = reasoning_text(message)
            if thinking:
                entry[field] = thinking
            rendered.append(entry)
        else:
            for result in message.tool_results():
                tool_entry: Dict[str, Any] = {
                    "role": "tool",
                    "content": result_text(result),
                    "tool_call_id": ids.get(result.call_id, result.call_id),
                }
                if result.name:
                    tool_entry["name"] = result.name
                rendered.append(tool_entry)
    return rendered


def tool_choice_wire(choice: ToolChoice, profile: CapabilityProfile, reasoning: bool) -> Any:
    allowed = profile.tool_choice_with_reasoning if reasoning else profile.tool_choice
    if choice.mode not in allowed or choice.mode == "auto":
        return None
    if choice.mode == "named":
        return {"type": "function", "function": {"name": choice.name}}
    return choice.mode


def apply_reasoning_toggle(body: Dict[str, Any], settings: Settings, profile: CapabilityProfile) -> None:
    value = settings.reasoning
    toggle = profile.reasoning_toggle
    if value is None or toggle is None:
        return
    if toggle == TOGGLE_REASONING_EFFORT:
        if value is False:
            body["reasoning_effort"] = "none"
        elif isinstance(value, str):
            body["reasoning_effort"] = value
    elif toggle == TOGGLE_THINKING:
        body["thinking"] = {"type": "disabled" if value is False else "enabled"}
    elif toggle == TOGGLE_TEMPLATE_KWARGS:
        kwargs = dict(body.get("chat_template_kwargs") or {})
        kwargs["enable_thinking"] = value is not False
        body["chat_template_kwargs"] = kwargs


def usage_from(raw: Dict[str, Any]) -> Usage:
    def count(value: Any) -> Optional[int]:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    completion_details = raw.get("completion_tokens_details")
    prompt_details = raw.get("prompt_tokens_details")
    cached = count((prompt_details or {}).get("cached_tokens")) if isinstance(prompt_details, dict) else None
    if cached is None:
        cached = count(raw.get("prompt_cache_hit_tokens"))
    reasoning = count((completion_details or {}).get("reasoning_tokens")) if isinstance(completion_details, dict) else None
    return Usage(
        input_tokens=count(raw.get("prompt_tokens")),
        output_tokens=count(raw.get("completion_tokens")),
        total_tokens=count(raw.get("total_tokens")),
        reasoning_tokens=reasoning,
        cached_input_tokens=cached,
        raw=raw,
    )


class _Call:
    def __init__(self, block: int) -> None:
        self.block = block
        self.wire_id: Optional[str] = None
        self.name_parts: List[str] = []
        self.started = False
        self.ended = False


class OpenAIChatDecoder:
    def __init__(
        self,
        seed: str,
        rule: ToolIdRule,
        classify: Callable[[WireFailure], HarnessError],
        label: str = "openai-chat",
    ) -> None:
        self._seed = seed
        self._rule = rule
        self._classify = classify
        self._label = label
        self._framer = SSEFramer(label)
        self._think = ThinkTagStage()
        self._next_block = 0
        self._kind: Optional[str] = None
        self._block = 0
        self._calls: List[_Call] = []
        self._by_index: Dict[Any, _Call] = {}
        self._finish_raw: Optional[str] = None
        self._saw_finish_reason = False
        self._done = False
        self._finished = False

    def feed(self, chunk: bytes) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for record in self._framer.feed(chunk):
            if self._done:
                break
            events.extend(self._record(record))
        return events

    def close(self) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for record in self._framer.close():
            if self._done:
                break
            events.extend(self._record(record))
        if not self._finished:
            events.extend(self._finish(explicit_end=False))
        self._done = True
        return events

    def _record(self, record: SSERecord) -> List[StreamEvent]:
        if record.data == DONE_SENTINEL:
            events = self._finish(explicit_end=True)
            self._done = True
            return events
        try:
            data = json.loads(record.data)
        except ValueError:
            logging.warning(f"[{self._label}] Failed to parse SSE data: {record.data[:200]}")
            return []
        if not isinstance(data, dict):
            return []
        return self._chunk(data)

    def _chunk(self, data: Dict[str, Any]) -> List[StreamEvent]:
        self._raise_in_band(data)
        events: List[StreamEvent] = []
        usage = data.get("usage")
        choices = data.get("choices")
        if isinstance(usage, dict) and usage:
            events.append(UsageReported(usage_from(usage)))
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return events
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, dict):
            events.extend(self._delta(delta))
        reason = choice.get("finish_reason")
        if reason:
            self._finish_raw = reason
            self._saw_finish_reason = True
            events.extend(self._flush_text())
            events.extend(self._end_calls())
        return events

    def _raise_in_band(self, data: Dict[str, Any]) -> None:
        error = data.get("error")
        if not error or data.get("choices"):
            return
        code = error.get("code") if isinstance(error, dict) else None
        status = code if isinstance(code, int) and not isinstance(code, bool) else 500
        body = json.dumps(data).encode("utf-8")
        raise self._classify(WireFailure(status=status, body=body, in_stream=True))

    def _delta(self, delta: Dict[str, Any]) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for key in REASONING_FIELDS:
            text = delta.get(key)
            if isinstance(text, str) and text:
                events.append(ReasoningDelta(block=self._use("reasoning"), text=text))
                break
        content = delta.get("content")
        if isinstance(content, str) and content:
            events.extend(self._pieces(self._think.feed(content)))
        fragments = delta.get("tool_calls")
        if isinstance(fragments, list):
            for fragment in fragments:
                if isinstance(fragment, dict):
                    events.extend(self._fragment(fragment))
        return events

    def _use(self, kind: str) -> int:
        if kind != self._kind:
            self._kind = kind
            self._block = self._next_block
            self._next_block += 1
        return self._block

    def _pieces(self, pieces) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for piece in pieces:
            if piece.kind == TEXT:
                events.append(TextDelta(block=self._use("text"), text=piece.text))
            elif piece.kind == THINKING:
                events.append(ReasoningDelta(block=self._use("reasoning"), text=piece.text))
            elif piece.kind == CLOSED:
                self._kind = None
        return events

    def _flush_text(self) -> List[StreamEvent]:
        return self._pieces(self._think.finish())

    def _fragment(self, fragment: Dict[str, Any]) -> List[StreamEvent]:
        function = fragment.get("function")
        function = function if isinstance(function, dict) else {}
        wire_id = fragment.get("id") or None
        name = function.get("name")
        arguments = function.get("arguments")
        call = self._call_for(fragment.get("index"), wire_id, bool(name))
        if wire_id and not call.started:
            call.wire_id = wire_id
        if isinstance(name, str) and name and not call.started:
            call.name_parts.append(name)
        events: List[StreamEvent] = []
        if arguments is not None and arguments != "":
            text = arguments if isinstance(arguments, str) else json.dumps(arguments)
            events.extend(self._start(call))
            events.append(ToolArgsDelta(block=call.block, text=text))
        return events

    def _call_for(self, index: Any, wire_id: Optional[str], has_name: bool) -> _Call:
        if index is None:
            continues = bool(self._calls) and not wire_id and (not has_name or not self._calls[-1].started)
            key: Any = ("implicit", len(self._calls) - 1 if continues else len(self._calls))
        else:
            key = index
        call = self._by_index.get(key)
        if call is not None and wire_id and call.wire_id and wire_id != call.wire_id:
            call = None
        if call is None:
            self._kind = "tool"
            call = _Call(self._next_block)
            self._next_block += 1
            self._by_index[key] = call
            self._calls.append(call)
        return call

    def _start(self, call: _Call) -> List[StreamEvent]:
        if call.started:
            return []
        call.started = True
        position = self._calls.index(call)
        if self._rule is not ToolIdRule.MINT and call.wire_id:
            call_id = call.wire_id
        else:
            call_id = tool_ids.mint_call_id(self._seed, position)
        return [ToolCallStart(block=call.block, call_id=call_id, name="".join(call.name_parts), wire_id=call.wire_id)]

    def _end_calls(self) -> List[StreamEvent]:
        events: List[StreamEvent] = []
        for call in self._calls:
            if call.ended:
                continue
            events.extend(self._start(call))
            events.append(ToolCallEnd(block=call.block))
            call.ended = True
        return events

    def _finish(self, explicit_end: bool) -> List[StreamEvent]:
        self._finished = True
        events = self._flush_text()
        if explicit_end or self._saw_finish_reason:
            events.extend(self._end_calls())
        reason = finish_reasons.with_tool_calls(
            finish_reasons.from_openai_compat(self._finish_raw), any(call.ended for call in self._calls)
        )
        events.append(Finish(reason=reason, raw_reason=self._finish_raw))
        return events


class OpenAIChatAdapter(HttpAdapter):
    name = "openai-chat"
    display_name = "OpenAI-compatible server"

    def build_body(self, request: RenderRequest, profile: CapabilityProfile) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": request.config.model,
            "messages": render_messages(request, profile),
            **sampling_params(request, profile),
        }
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        if request.tools and profile.tools:
            body["tools"] = [tool.function_schema() for tool in request.tools]
            if request.settings.tool_choice is not None:
                choice = tool_choice_wire(
                    request.settings.tool_choice, profile, reasoning_active(request.settings, profile)
                )
                if choice is not None:
                    body["tool_choice"] = choice
        apply_reasoning_toggle(body, request.settings, profile)
        return body

    def render(self, request: RenderRequest, profile: CapabilityProfile) -> WireRequest:
        return WireRequest(
            method="POST",
            url=f"{request.config.base_url.rstrip('/')}/chat/completions",
            headers=auth_headers(request.config),
            body=self.build_body(request, profile),
            stream=True,
        )

    def decoder(self, request: RenderRequest, profile: CapabilityProfile) -> OpenAIChatDecoder:
        return OpenAIChatDecoder(
            seed=tool_ids.seed_for(request.messages),
            rule=profile.tool_id_rule,
            classify=self.classify_error,
            label=self.name,
        )

    async def list_models(self, base_url: str, api_key: Optional[str] = None) -> List[DiscoveredModel]:
        payload = await fetch_json(self.display_name, base_url, "/models", bearer_headers(api_key))
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ModelListingError(f"{self.display_name} returned no model list.")
        models = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            owner = entry.get("owned_by")
            models.append(DiscoveredModel(
                id=str(entry["id"]),
                details={"owned_by": owner} if owner else None,
            ))
        return sorted_models(models)

    async def discover(self, config: LLMConfig) -> ProfileLayer:
        fields: Dict[str, Any] = {}
        try:
            payload = await fetch_json(
                self.display_name,
                config.base_url,
                "/models",
                bearer_headers(config.api_key),
                timeout=CONTEXT_WINDOW_PROBE_TIMEOUT_SECONDS,
            )
            entries = payload.get("data") if isinstance(payload, dict) else None
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict) or entry.get("id") != config.model:
                    continue
                for key in CONTEXT_WINDOW_KEYS:
                    tokens = positive_int(entry.get(key))
                    if tokens:
                        fields["context_window"] = tokens
                        break
                break
        except Exception:
            logging.debug("%s discovery failed for %s", self.display_name, config.model, exc_info=True)
        return ProfileLayer(ProfileSource.DISCOVERED, fields)
